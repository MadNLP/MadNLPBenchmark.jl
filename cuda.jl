
using DelimitedFiles
using MadNLPBenchmark

const RESULTS_DIR = joinpath(@__DIR__, "results")

include(joinpath(@__DIR__, "ext", "MadNLPCUDA.jl"))

const BENCHMARK_KEYS = Dict{String, MadNLPBenchmark.AbstractBenchmarkSetting}(
    "cops" => MadNLPBenchmark.COPS(; backend=CUDABackend()),
    "mittelmann" => MadNLPBenchmark.COPS(; config=:mittelmann, backend=CUDABackend()),
    "acopf" => MadNLPBenchmark.ACOPFBenchmark(; backend=CUDABackend()),
    "acopf-polar" => MadNLPBenchmark.ACOPFBenchmark(;formulation=:polar, backend=CUDABackend()),
    "acopf-rect" => MadNLPBenchmark.ACOPFBenchmark(;formulation=:rect, backend=CUDABackend()),
)

function parse_args(args::Vector{String})
    # Default options
    solver = MadNLPCUDA()
    benchmark = BENCHMARK_KEYS["acopf"]
    tol = 1e-8
    for arg in args
        if startswith(arg, "--benchmark=")
            benchmark = BENCHMARK_KEYS[split(arg, "=")[2]]
        elseif startswith(arg, "--tol=")
            tol = parse(Float64, split(arg, "=")[2])
        end
    end
    solver.tol = tol
    println(benchmark)
    return solver, benchmark
end

function @main(args::Vector{String})
    solver, benchmark = parse_args(args)
    if !isdir(RESULTS_DIR)
        mkdir(RESULTS_DIR)
    end
    # Launch benchmark
    probs = MadNLPBenchmark.get_instances(benchmark)
    results = MadNLPBenchmark.run_benchmark(benchmark, probs, solver)
    label = MadNLPBenchmark.introduce(benchmark, solver)
    writedlm(joinpath(RESULTS_DIR, "$(label).csv"), results)
    return true
end
