using Pkg
using Comonicon
using Distributed

# Load CUTEst library
@everywhere include("config.jl")

BASE_DIR = joinpath(@__DIR__, "..", "results")
RESULTS_DIR = joinpath(BASE_DIR, "cutest")

Comonicon.@main function main(
    ;
    solver="madnlp",
    decode="false",
    quick="false",
    ipopt_rev = "main",
    madnlp_rev = "master",
    madnlp_linear_solver = "Ma57Solver",
    ipopt_linear_solver = "ma57",
    tol = "1e-6",
    )
    
    @everywhere begin
        madnlp_linear_solver = $madnlp_linear_solver
        ipopt_linear_solver = $ipopt_linear_solver
        tol = $(parse(Float64,tol))
    end

    quick = parse(Bool, quick)
    decode = parse(Bool, decode)
    
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end

    probs = if quick
        readdlm(joinpath(@__DIR__, "cutest-quick-names.csv"))[:]
    else
        probs = CUTEst.list_sif_problems()
    end

    filter!(e->!(e in EXCLUDE),probs)

    run_madnlp = (solver == "madnlp") || (solver == "all")
    run_ipopt = (solver == "ipopt") || (solver == "all")

    if run_madnlp
        @info "Benchmark MadNLP"
        @everywhere include("config_madnlp.jl")

        @everywhere println(pathof(MadNLP))
        status,time,mem,iter = benchmark(madnlp_solver,probs;warm_up_probs = ["EIGMINA"], decode = decode)
        results = [probs status time mem iter]

        output_file = joinpath(RESULTS_DIR, "cutest-madnlp-$madnlp_rev-$madnlp_linear_solver")
        writedlm(output_file * ".csv", results)
        write(output_file * ".txt", "MadNLP rev: $madnlp_rev\nLinear solver: $madnlp_linear_solver\nTol: $tol\n")
    end
    
    if run_ipopt
        @info "Benchmark Ipopt"
        @everywhere include("config_ipopt.jl")
        status,time,mem,iter = benchmark(ipopt_solver,probs;warm_up_probs = ["EIGMINA"], decode = decode)
        results = [probs status time mem iter]
        output_file = joinpath(RESULTS_DIR, "cutest-ipopt-$ipopt_rev-$ipopt_linear_solver")
        writedlm(output_file * ".csv", results)
        write(output_file * ".txt", "Ipopt rev: $ipopt_rev\nLinear solver: $ipopt_linear_solver\nTol: $tol\n")
    end
end

