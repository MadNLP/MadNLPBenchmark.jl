
using Comonicon
include(joinpath(@__DIR__, "..", "config.jl"))

RESULTS_DIR = joinpath(BASE_DIR, "argos", "reduced")

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
    flag = quick ? "quick" : "full"
    dev = use_gpu ? "cuda" : "cpu"
    lapack_solver = use_gpu ? LapackGPUSolver : LapackCPUSolver

    results = main(
        cases,
        Argos.DommelTinney(),
        ntrials;
        print_level=MadNLP.ERROR,
        linear_solver=lapack_solver,
        use_gpu=use_gpu,
        max_iter=300,
        tol=1e-5,
        dual_initialized=true,
    )
    output_file = joinpath(RESULTS_DIR, "argos-$(flag)-reduced-$(dev).txt")
    writedlm(output_file, [cases results])
end

