
include(joinpath(@__DIR__, "..", "config.jl"))

RESULTS_DIR = joinpath(BASE_DIR, "argos", "full")

if !isdir(RESULTS_DIR)
    mkpath(RESULTS_DIR)
end

function build_full_space_opf(model; use_gpu=false)
    blk = if use_gpu
        model_gpu = PolarForm(model, CUDABackend())
        flp = Argos.FullSpaceEvaluator(model_gpu)
        Argos.OPFModel(Argos.bridge(flp))
    else
        flp = Argos.FullSpaceEvaluator(model)
        Argos.OPFModel(flp)
    end
    # Init model for initial LU factorization in BieglerKKTSystem
    x0 = NLPModels.get_x0(blk)
    nnzj = NLPModels.get_nnzj(blk)
    jac = zeros(nnzj)
    NLPModels.jac_coord!(blk, x0, jac)
    return blk
end

function build_madnlp(blk::Argos.OPFModel, ::Argos.FullSpace; options...)
    return MadNLP.MadNLPSolver(
        blk;
        kkt_system=MadNLP.SparseKKTSystem,
        callback=MadNLP.SparseCallback,
        options...,
    )
end

function build_madnlp(
    blk::Argos.OPFModel,
    ::Argos.BieglerReduction;
    options...
)
    KKT = Argos.BieglerKKTSystem{Float64, CuVector{Int}, CuVector{Float64}, CuMatrix{Float64}}
    return MadNLP.MadNLPSolver(
        blk;
        kkt_system=KKT,
        linear_solver=LapackGPUSolver,
        lapack_algorithm=MadNLP.CHOLESKY,
        callback=MadNLP.SparseCallback,
        options...
    )
end

function benchmark(model, kkt; use_gpu=false, ntrials=3, options...)
    blk = build_full_space_opf(model; use_gpu=use_gpu)

    ## Warm-up
    solver = build_madnlp(blk, kkt; max_iter=1, print_level=MadNLP.ERROR)
    results = MadNLP.solve!(solver)

    ## Benchmark
    t_total, t_callbacks, t_linear_solver = (0.0, 0.0, 0.0)
    n_it, obj = 0, 0.0
    for _ in 1:ntrials
        Argos.reset!(Argos.backend(blk))
        # Solve
        solver = build_madnlp(blk, kkt; options...)
        results = MadNLP.solve!(solver)
        # Save results
        t_total += solver.cnt.total_time
        t_callbacks += solver.cnt.eval_function_time
        t_linear_solver += solver.cnt.linear_solver_time
        n_it += solver.cnt.k
        obj += solver.obj_val
        # Clean memory
        use_gpu && refresh_memory()
    end

    return (
        status = get_status(results.status),
        iters = n_it / ntrials,
        obj = obj / ntrials,
        total = t_total / ntrials,
        callbacks = t_callbacks / ntrials,
        linear_solver = t_linear_solver / ntrials,
    )
end

function main(cases, kkt, ntrials; use_gpu=false, options...)
    # Setup
    dev = use_gpu ? :cuda : :cpu
    form = isa(kkt, Argos.BieglerReduction) ? :bieglerkkt : :sparsekkt

    nexp = length(cases)
    results = zeros(nexp, 6)

    for (i, case) in enumerate(cases)
        @info "Case: $case"
        datafile = joinpath(PGLIB_PATH, case)
        model = ExaPF.PolarForm(datafile)

        r = benchmark(model, kkt; ntrials=ntrials, use_gpu=use_gpu, options...)
        results[i, :] .= (r.status, r.iters, r.obj, r.total, r.callbacks, r.linear_solver)
    end

    output_file = joinpath(RESULTS_DIR, "argos-full-$(form)-$(dev).txt")
    writedlm(output_file, [cases results])
    return results
end

cases = filter!(e->(occursin("pglib_opf_case",e) && occursin("ieee",e)),readdir(PGLIB_PATH))
use_gpu = IS_GPU_AVAILABLE

main(
    cases,
    Argos.FullSpace(),
    3;
    print_level=MadNLP.ERROR,
    linear_solver=Ma57Solver,
    use_gpu=use_gpu,
    max_iter=300,
    tol=1e-5,
    dual_initialized=true,
)

if IS_GPU_AVAILABLE
    main(
        cases,
        Argos.BieglerReduction(),
        3;
        print_level=MadNLP.ERROR,
        linear_solver=LapackGPUSolver,
        use_gpu=use_gpu,
        max_iter=300,
        tol=1e-5,
        dual_initialized=true,
    )
end

