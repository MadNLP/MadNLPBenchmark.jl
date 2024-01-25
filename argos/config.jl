using DelimitedFiles

using CUDA
using CUSOLVERRF

using NLPModels

using MadNLP
using MadNLPHSL
using MadNLPGPU

using ExaPF
using Argos

const PS = ExaPF.PowerSystem

if haskey(ENV, "PGLIB_PATH")
    const PGLIB_PATH = ENV["PGLIB_PATH"]
else
    error("Unable to find path to PGLIB benchmark.\n"*
          "Please set environment variable `PGLIB_PATH` to run benchmark with PowerModels.jl")
end

const BASE_DIR = joinpath(@__DIR__, "..", "results")
const IS_GPU_AVAILABLE = CUDA.has_cuda() && CUDA.functional()


if !isdir(BASE_DIR)
    mkdir(BASE_DIR)
end

if IS_GPU_AVAILABLE
    CUDA.allowscalar(false)
end

function get_status(code::MadNLP.Status)
    if code == MadNLP.SOLVE_SUCCEEDED
        return 1
    elseif code == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
        return 2
    else
        return 3
    end
end

function refresh_memory()
    GC.gc(true)
    CUDA.has_cuda() && CUDA.reclaim()
    return
end

function build_opf(model, kkt::Union{Argos.FullSpace, Argos.BieglerReduction}; use_gpu=false)
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

function build_opf(model, kkt::Argos.DommelTinney; use_gpu=false)
    nlp = if use_gpu
        model_gpu = PolarForm(model, CUDABackend())
        nlp = Argos.ReducedSpaceEvaluator(model_gpu; nbatch_hessian=256)
    else
        nlp = Argos.ReducedSpaceEvaluator(model; nbatch_hessian=256)
    end
    return Argos.OPFModel(nlp)
end

function build_madnlp(blk::Argos.OPFModel, ::Argos.FullSpace; options...)
    return MadNLP.MadNLPSolver(
        blk;
        kkt_system=MadNLP.SparseKKTSystem,
        callback=MadNLP.SparseCallback,
        linear_solver=Ma57Solver,
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

function build_madnlp(
    blk::Argos.OPFModel,
    ::Argos.DommelTinney;
    options...
)
    return MadNLP.MadNLPSolver(
        blk;
        kkt_system=MadNLP.DenseCondensedKKTSystem,
        linear_solver=LapackGPUSolver,
        lapack_algorithm=MadNLP.CHOLESKY,
        options...,
    )
end

function benchmark(model, kkt; use_gpu=false, ntrials=3, options...)
    blk = build_opf(model, kkt; use_gpu=use_gpu)

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
    nexp = length(cases)
    results = zeros(nexp, 6)

    for (i, case) in enumerate(cases)
        @info "Case: $case"
        datafile = joinpath(PGLIB_PATH, case)
        model = ExaPF.PolarForm(datafile)

        r = benchmark(model, kkt; ntrials=ntrials, use_gpu=use_gpu, options...)
        results[i, :] .= (r.status, r.iters, r.obj, r.total, r.callbacks, r.linear_solver)
    end

    return results
end

