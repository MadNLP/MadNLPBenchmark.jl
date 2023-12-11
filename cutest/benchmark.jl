
using Distributed

# Load CUTEst library
@everywhere include("config.jl")

BASE_DIR = joinpath(@__DIR__, "..", "results")
RESULTS_DIR = joinpath(BASE_DIR, "cutest")
QUICK_BENCHMARK = true
DECODE = true
SOLVER = "madnlp"

if !isdir(BASE_DIR)
    mkdir(BASE_DIR)
end
if !isdir(RESULTS_DIR)
    mkdir(RESULTS_DIR)
end

function madnlp_solver(nlp)
    return madnlp(
        nlp;
        linear_solver=Ma57Solver,
        max_wall_time=900.0,
        print_level=MadNLP.ERROR,
        tol=1e-6,
    )
end

function ipopt_solver(nlp)
    return ipopt(
        nlp;
        linear_solver="ma57",
        max_cpu_time=900.0,
        print_level=0,
        tol=1e-6,
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

if QUICK_BENCHMARK
    probs = readdlm(joinpath(@__DIR__, "cutest-quick-names.csv"))[:]
else
    probs = CUTEst.select()
end

filter!(e->!(e in exclude),probs)

solver = if SOLVER == "madnlp"
    madnlp_solver
elseif SOLVER == "ipopt"
    ipopt_solver
end

status,time,mem,iter = benchmark(solver,probs;warm_up_probs = ["EIGMINA"], decode = DECODE)

results = [probs status time mem iter]
flag = "short"
output_file = joinpath(RESULTS_DIR, "cutest-$(flag)-$(SOLVER).csv")
writedlm(output_file, results)

