
using Comonicon
using Distributed

@everywhere include("config.jl")

BASE_DIR = joinpath(@__DIR__, "..", "results")
RESULTS_DIR = joinpath(BASE_DIR, "cops")

function benchmark(evalmodel, probs; warm_up_probs=[])
    nprobs = length(probs)
    status = []
    names = []
    time = Float64[]
    obj = Float64[]
    mem = Float64[]
    iter = Float64[]
    println("Solving problems")
    for (k, (instance, params)) in enumerate(probs)
        name = parse_name(instance, params)
        model = instance(params...)
        @info("Solving $name")
        s, o, t, m, i = evalmodel(model)
        push!(names, name)
        push!(status, s)
        push!(obj, o)
        push!(time, t)
        push!(mem, m)
        push!(iter, i)
    end
    results = [names status obj time mem iter]
    return results
end

@main function main(;
    instances="default",
    solver="madnlp-cpu",
    quick::Bool=false,
)
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end

    if quick
        all_cases = COPS_INSTANCES_QUICK
        flag = "quick"
    else
        if instances == "gpu"
            all_cases = COPS_INSTANCES_GPU
            flag = "gpu"
        elseif instances == "mittelmann"
            all_cases = COPS_INSTANCES_MITTELMANN
            flag = "mittelmann"
        elseif instances == "default"
            all_cases = COPS_INSTANCES_DEFAULT
            flag = "default"
        else
            println("Supported options for `instance` are `gpu`, `mittelmann` and `default`.")
            return
        end
    end

    backend = (instances == "gpu") ? :examodels : :jump

    run_madnlp_cpu = (solver == "madnlp-cpu") || (solver == "all")
    run_madnlp_cuda = (solver == "madnlp-cuda") || (solver == "all")
    run_hykkt_cuda = (solver == "hykkt-cuda") || (solver == "all")
    run_ipopt = (solver == "ipopt") || (solver == "all")

    if run_ipopt
        if backend == :examodels
            results = benchmark(solve_ipopt_ma57, all_cases)
        else
            results = benchmark(solve_ipopt_ma57_jump, all_cases)
        end
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-$(backend)-ipopt-ma57.csv")
        writedlm(output_file, results)
    end
    if run_madnlp_cpu
        if backend == :examodels
            results = benchmark(solve_madnlp_ma57, all_cases)
        else
            results = benchmark(solve_madnlp_ma57_jump, all_cases)
        end
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-$(backend)-madnlp-ma57.csv")
        writedlm(output_file, results)
    end
    if run_madnlp_cuda && backend == :examodels
        results = benchmark(solve_madnlp_sckkt_cuda, all_cases)
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-$(backend)-madnlp-cuda.csv")
        writedlm(output_file, results)
    end
    if run_hykkt_cuda && backend == :examodels
        results = benchmark(solve_madnlp_hykkt_cuda, all_cases)
        output_file = joinpath(RESULTS_DIR, "cops-$(flag)-$(backend)-hykkt-cuda.csv")
        writedlm(output_file, results)
    end
end

