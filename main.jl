

using Distributed
using DelimitedFiles

@everywhere begin
    using MadNLPBenchmark
    using MadNLPHSL
end

const RESULTS_DIR = joinpath(@__DIR__, "results")

const BENCHMARK_KEYS = Dict{String, MadNLPBenchmark.AbstractBenchmarkSetting}(
    "cutest-small" => MadNLPBenchmark.CUTEstBenchmark(; max_var=1000),
    "cutest" => MadNLPBenchmark.CUTEstBenchmark(),
    "cops" => MadNLPBenchmark.COPSBenchmark(),
    "mittelmann" => MadNLPBenchmark.COPSBenchmark(; config=:mittelmann),
    "acopf" => MadNLPBenchmark.ACOPFBenchmark(),
    "acopf-polar" => MadNLPBenchmark.ACOPFBenchmark(;formulation=:polar),
    "acopf-rect" => MadNLPBenchmark.ACOPFBenchmark(;formulation=:rect),
)

const SOLVER_KEYS = Dict{String, MadNLPBenchmark.AbstractSolverSetup}(
    "ipopt" => MadNLPBenchmark.IpoptSolver(),
    "ipopt-ma27" => MadNLPBenchmark.IpoptSolver(; linear_solver="ma27"),
    "madnlp" => MadNLPBenchmark.MadNLPDefault(),
    "madnlp-ma27" => MadNLPBenchmark.MadNLPDefault(; linear_solver=Ma27Solver),
)

function parse_args(args::Vector{String})
    # Default options
    solver = MadNLPBenchmark.IpoptSolver()
    benchmark = MadNLPBenchmark.CUTEstBenchmark()
    decode_cutest = false
    automatic_scaling = false
    tol = 1e-8
    for arg in args
        if startswith(arg, "--solver=")
            solver = SOLVER_KEYS[split(arg, "=")[2]]
        elseif startswith(arg, "--benchmark=")
            benchmark = BENCHMARK_KEYS[split(arg, "=")[2]]
        elseif startswith(arg, "--tol=")
            tol = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--decode-cutest")
            decode_cutest = true
        elseif startswith(arg, "--automatic-scaling")
            automatic_scaling = true
        end
    end
    if decode_cutest
        benchmark = MadNLPBenchmark.CUTEstBenchmark()
    end
    solver.tol = tol
    solver.automatic_scaling = automatic_scaling
    return solver, benchmark, decode_cutest
end

function @main(args::Vector{String})
    solver, benchmark, decode_cutest = parse_args(args)

    if !isdir(RESULTS_DIR)
        mkdir(RESULTS_DIR)
    end

    # Launch benchmark
    probs = MadNLPBenchmark.get_instances(benchmark)
    if decode_cutest
        MadNLPBenchmark.decode_cutest(probs)
        # Early return
        return true
    end

    retvals = pmap(prob->MadNLPBenchmark.evaluate_model(benchmark,solver, prob),probs)

    # Export results
    n = length(probs)
    results = zeros(n, 8)
    for k in 1:n
        results[k, :] .= retvals[k]
    end
    names = MadNLPBenchmark.parse_name.(probs, Ref(benchmark))
    results = [reshape(MadNLPBenchmark.COLS, 1, 9); names results]

    label = MadNLPBenchmark.introduce(benchmark, solver)
    writedlm(joinpath(RESULTS_DIR, "$(label).csv"), results)
    return true
end

