
using Comonicon
using Distributed

# Load CUTEst library
@everywhere include(joinpath(@__DIR__, "..", "config.jl"))

BASE_DIR = joinpath(@__DIR__, "..", "..", "results")
RESULTS_DIR = joinpath(BASE_DIR, "cutest", "lbfgs")

@everywhere function madnlp_solver_lbfgs(nlp)
    return madnlp(
        nlp;
        callback = MadNLP.SparseCallback,
        kkt_system = MadNLP.SparseKKTSystem,
        hessian_approximation=MadNLP.CompactLBFGS,
        linear_solver=Ma57Solver,
        max_wall_time=900.0,
        print_level=MadNLP.ERROR,
        tol=1e-5,
    )
end

@everywhere function ipopt_solver_lbfgs(nlp)
    return ipopt(
        nlp;
        linear_solver="ma57",
        hessian_approximation="limited-memory",
        limited_memory_max_history=30,
        max_cpu_time=900.0,
        print_level=0,
        tol=1e-5,
    )
end

@main function main(; solver="madnlp", decode=true, quick::Bool=false)
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end

    probs = if quick
        probs = readdlm(joinpath(@__DIR__, "cutest-lbfgs-names.csv"))[:]
    else
        probs = CUTEst.select()
    end
    flag = quick ? "quick" : "complete"

    filter!(e->!(e in EXCLUDE),probs)

    run_madnlp = (solver == "madnlp") || (solver == "all")
    run_ipopt = (solver == "ipopt") || (solver == "all")

    if run_madnlp
        @info "Benchmark MadNLP-LBFGS"
        status,time,mem,iter = benchmark(madnlp_solver_lbfgs,probs;warm_up_probs = ["EIGMINA"], decode = decode)
        results = [probs status time mem iter]
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-madnlp-lbfgs.csv")
        writedlm(output_file, results)
    end
    if run_ipopt
        @info "Benchmark Ipopt-LBFGS"
        status,time,mem,iter = benchmark(ipopt_solver_lbfgs,probs;warm_up_probs = ["EIGMINA"], decode = decode)
        results = [probs status time mem iter]
        output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-ipopt-lbfgs.csv")
        writedlm(output_file, results)
    end
end


