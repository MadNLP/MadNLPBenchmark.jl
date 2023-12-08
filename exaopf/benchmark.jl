using DelimitedFiles
using MadNLP, MadNLPHSL, MadNLPGPU
using JuMP, Ipopt, CUDA, NLPModels
using PowerModels
using Printf
using LinearAlgebra

PowerModels.silence()
CUDA.allowscalar(false);

include("model.jl")
include("config.jl")

RESULTS_DIR = joinpath(@__DIR__, "results")
SOLVER = "madnlp-cpu"

if !isdir(RESULTS_DIR)
    mkdir(RESULTS_DIR)
end

all_cases = filter(((n,c),)-> endswith(n, "goc") || endswith(n, "pegase"), pglib_cases)
all_cases = filter(((n,c),)-> endswith(n, "ieee"), pglib_cases)
ncases = length(all_cases)

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

evaluation = if SOLVER == "madnlp-cpu"
    madnlp_eval_cpu
elseif SOLVER == "madnlp-cuda"
    madnlp_eval_cuda
elseif SOLVER == "ipopt"
    ipopt_eval
end

results = benchmark(evaluation, all_cases; warm_up_probs=["pglib_opf_case118_ieee.m"])

output_file = joinpath(RESULTS_DIR, "powermodels-$(SOLVER).csv")
writedlm(output_file, results)
