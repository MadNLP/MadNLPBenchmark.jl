
using Comonicon
using Distributed

# Load CUTEst library
@everywhere include("config.jl")

BASE_DIR = joinpath(@__DIR__, "..", "results")
RESULTS_DIR = joinpath(BASE_DIR, "cutest")

@main function main(; solver="madnlp", decode::Int=1, quick::Bool=false)
    decode = Bool(decode)
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end

    probs = if quick
        readdlm(joinpath(@__DIR__, "cutest-quick-names.csv"))[:]
    else
        probs = CUTEst.select()
    end
    flag = quick ? "quick" : "complete"

    filter!(e->!(e in EXCLUDE),probs)

    run_madnlp = (solver == "madnlp") || (solver == "all")
    run_ipopt = (solver == "ipopt") || (solver == "all")

    if run_madnlp
        @info "Benchmark MadNLP"
        status,time,mem,iter = benchmark(madnlp_solver,probs;warm_up_probs = ["EIGMINA"], decode = decode)
        results = [probs status time mem iter]
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-madnlp.csv")
        writedlm(output_file, results)
    end
    if run_ipopt
        @info "Benchmark Ipopt"
        status,time,mem,iter = benchmark(ipopt_solver,probs;warm_up_probs = ["EIGMINA"], decode = decode)
        results = [probs status time mem iter]
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-ipopt.csv")
        writedlm(output_file, results)
    end
end

