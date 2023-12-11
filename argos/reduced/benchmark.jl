
include(joinpath(@__DIR__, "..", "config.jl"))

RESULTS_DIR = joinpath(BASE_DIR, "argos", "reduced")

if !isdir(RESULTS_DIR)
    mkpath(RESULTS_DIR)
end

cases = filter!(e->(occursin("pglib_opf_case",e) && occursin("ieee",e)),readdir(PGLIB_PATH))
use_gpu = IS_GPU_AVAILABLE
lapack_solver = use_gpu ? LapackGPUSolver : LapackCPUSolver

main(
    cases,
    Argos.DommelTinney(),
    3;
    print_level=MadNLP.INFO,
    linear_solver=lapack_solver,
    use_gpu=use_gpu,
    max_iter=300,
    tol=1e-5,
    dual_initialized=true,
)

