
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
    obj = Float64[]
    mem = Float64[]
    iter = Float64[]
    println("Solving problems")
    for (k, prob) in enumerate(probs)
        name, case = prob
        println("Solving $name")
        s, o, t, m, i = evalmodel(case)
        push!(names, name)
        push!(status, s)
        push!(obj, o)
        push!(time, t)
        push!(mem, m)
        push!(iter, i)
    end
    return [names status obj time mem iter]
end

@main function main(;
    solver="madnlp-cpu",
    quick::Bool=false,
)
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end

    all_cases = if quick
        all_cases = filter(((n,c),)-> endswith(n, "ieee"), pglib_cases)
    else
        all_cases = filter(((n,c),)-> endswith(n, "goc") || endswith(n, "pegase"), pglib_cases)
    end

    flag = quick ? "quick" : "full"

    run_madnlp_cpu = (solver == "madnlp-cpu") || (solver == "all")
    run_madnlp_cuda = (solver == "madnlp-cuda") || (solver == "all")
    run_hykkt_cuda = (solver == "hykkt-cuda") || (solver == "all")
    run_ipopt = (solver == "ipopt") || (solver == "all")

    if run_madnlp_cpu
        results = benchmark(solve_madnlp_ma27, all_cases; warm_up_probs=["pglib_opf_case118_ieee.m"])
        output_file = joinpath(RESULTS_DIR, "exaopf-$(flag)-madnlp-ma27.csv")
        writedlm(output_file, results)
    end
    if run_madnlp_cuda
        results = benchmark(solve_madnlp_sckkt_cuda, all_cases; warm_up_probs=["pglib_opf_case118_ieee.m"])
        output_file = joinpath(RESULTS_DIR, "exaopf-$(flag)-madnlp-cuda.csv")
        writedlm(output_file, results)
    end
    if run_hykkt_cuda
        results = benchmark(solve_madnlp_hykkt_cuda, all_cases; warm_up_probs=["pglib_opf_case118_ieee.m"])
        output_file = joinpath(RESULTS_DIR, "exaopf-$(flag)-hykkt-cuda.csv")
        writedlm(output_file, results)
    end
    if run_ipopt
        results = benchmark(solve_ipopt_ma27, all_cases; warm_up_probs=["pglib_opf_case118_ieee.m"])
        output_file = joinpath(RESULTS_DIR, "exaopf-$(flag)-ipopt.csv")
        writedlm(output_file, results)
    end
end

