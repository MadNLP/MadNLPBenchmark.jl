
using Comonicon

include("config.jl")

BASE_DIR = joinpath(@__DIR__, "..", "results")
RESULTS_DIR = joinpath(BASE_DIR, "exaopf")

function benchmark(evalmodel, probs; warm_up_probs=[])
    println("Warming up (forcing JIT compile)")
    for case in warm_up_probs
        prob = joinpath(PGLIB_PATH, case)
        evalmodel(prob)
    end

    nprobs = length(probs)
    status = []
    names = []
    time = Float64[]
    mem = Float64[]
    iter = Float64[]
    println("Solving problems")
    for (k, prob) in enumerate(probs)
        name, case = prob
        println("Solving $name")
        s, t, m, i = evalmodel(case)
        push!(names, name)
        push!(status, s)
        push!(time, t)
        push!(mem, m)
        push!(iter, i)
    end
    results = [names status time mem iter]
    return results
end

@main function main(; solver="madnlp-cpu", quick::Bool=false)
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end

    all_cases = if quick
        all_cases = filter(((n,c),)-> endswith(n, "ieee"), pglib_cases)
    else
        all_cases = filter(((n,c),)-> endswith(n, "goc") || endswith(n, "pegase"), pglib_cases)
    end
    ncases = length(all_cases)

    flag = quick ? "short" : "full"

    run_madnlp_cpu = (solver == "madnlp-cpu") || (solver == "all")
    run_madnlp_cuda = (solver == "madnlp-cuda") || (solver == "all")
    run_ipopt = (solver == "ipopt") || (solver == "all")

    if run_madnlp_cpu
        results = benchmark(madnlp_eval_cpu, all_cases; warm_up_probs=["pglib_opf_case118_ieee.m"])
        output_file = joinpath(RESULTS_DIR, "exaopf-madnlp-cpu.csv")
        writedlm(output_file, results)
    end
    if run_madnlp_cuda
        results = benchmark(madnlp_eval_cuda, all_cases; warm_up_probs=["pglib_opf_case118_ieee.m"])
        output_file = joinpath(RESULTS_DIR, "exaopf-madnlp-cuda.csv")
        writedlm(output_file, results)
    end
    if run_ipopt
        results = benchmark(ipopt_eval, all_cases; warm_up_probs=["pglib_opf_case118_ieee.m"])
        output_file = joinpath(RESULTS_DIR, "exaopf-ipopt.csv")
        writedlm(output_file, results)
    end
end

