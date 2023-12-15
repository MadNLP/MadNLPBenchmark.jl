
using Comonicon

include(joinpath(@__DIR__, "..", "config.jl"))

RESULTS_DIR = joinpath(BASE_DIR, "argos", "full")

@main function main(; quick::Bool=false, ntrials::Int=3)
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end

    cases = if quick
        filter!(e->(occursin("pglib_opf_case",e) && occursin("ieee",e)),readdir(PGLIB_PATH))
    else
        filter!(
            e->(
                occursin("pglib_opf_case",e) &&
                (occursin("ieee",e) || occursin("goc",e) || occursin("pegase",e))
            ),
            readdir(PGLIB_PATH),
        )
    end

    use_gpu = IS_GPU_AVAILABLE
    lapack_solver = use_gpu ? LapackGPUSolver : LapackCPUSolver

    main(
        cases,
        Argos.FullSpace(),
        ntrials;
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
            ntrials;
            print_level=MadNLP.ERROR,
            linear_solver=lapack_solver,
            use_gpu=use_gpu,
            max_iter=300,
            tol=1e-5,
            dual_initialized=true,
        )
    end
end
