
using Distributed

# Load CUTEst library
@everywhere include(joinpath(@__DIR__, "..", "config.jl"))

BASE_DIR = joinpath(@__DIR__, "..", "..", "results")
RESULTS_DIR = joinpath(BASE_DIR, "cutest", "lbfgs")
QUICK_BENCHMARK = true
DECODE = true
SOLVER = "madnlp"

if !isdir(RESULTS_DIR)
    mkpath(RESULTS_DIR)
end

function madnlp_solver_lbfgs(nlp)
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

function ipopt_solver_lbfgs(nlp)
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

exclude = [
    # MadNLP running into error
    # Ipopt running into error
    "EG3", # lfact blows up
    # Problems that are hopelessly large
    "TAX213322","TAXR213322","TAX53322","TAXR53322",
    "YATP1LS","YATP2LS","YATP1CLS","YATP2CLS",
    "CYCLOOCT","CYCLOOCF",
    "LIPPERT1",
    "GAUSSELM",
    "BA-L52LS","BA-L73LS","BA-L21LS"
]

probs = readdlm(joinpath(@__DIR__, "cutest-lbfgs-names.csv"))[:]

filter!(e->!(e in exclude),probs)

solver = if SOLVER == "madnlp"
    madnlp_solver_lbfgs
elseif SOLVER == "ipopt"
    ipopt_solver_lbfgs
end

status,time,mem,iter = benchmark(solver,probs;warm_up_probs = ["EIGMINA"], decode = DECODE)

results = [probs status time mem iter]
flag = "short"
output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-$(SOLVER).csv")
writedlm(output_file, results)

