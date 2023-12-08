using Distributed

@everywhere include("config.jl")

RESULTS_DIR = joinpath(@__DIR__, "results")
QUICK_BENCHMARK = true
SOLVER = "madnlp"

if !isdir(RESULTS_DIR)
    mkdir(RESULTS_DIR)
end

function benchmark(solver, probs; warm_up_probs=[], gc_off=true)
    println("Warming up (forcing JIT compile)")
    warm_up_pms = [
        instantiate_model(joinpath(PGLIB_PATH,case),type,PowerModels.build_opf)
        for (case,type) in warm_up_probs]
    println(get_name.(warm_up_pms))
    rs = [remotecall.(solver,i,warm_up_pms) for i in procs() if i!= 1]
    ws = [wait.(r) for r in rs]
    fs= [fetch.(r) for r in rs]

    println("Solving problems")
    retvals = pmap(prob->evalmodel(prob,solver;gcoff=gc_off),probs)

    status = [status for (status,time,mem,iter) in retvals]
    time   = [time for (status,time,mem,iter) in retvals]
    mem    = [mem for (status,time,mem,iter) in retvals]
    iter   = [iter for (status,time,mem,iter) in retvals]

    return status, time, mem, iter
end

if QUICK_BENCHMARK
    cases = filter!(e->(occursin("pglib_opf_case",e) && occursin("pegase",e)),readdir(PGLIB_PATH))
    types = [ACPPowerModel]
else
    cases = filter!(e->occursin("pglib_opf_case",e),readdir(PGLIB_PATH))
    types = [ACPPowerModel, ACRPowerModel, ACTPowerModel,
             DCPPowerModel, DCMPPowerModel, NFAPowerModel,
             DCPLLPowerModel,LPACCPowerModel, SOCWRPowerModel,
             QCRMPowerModel,QCLSPowerModel]
end
probs = [(case,type) for case in cases for type in types]
name =  ["$case-$type" for case in cases for type in types]

solver = if SOLVER == "madnlp"
    madnlp_solver
elseif SOLVER == "ipopt"
    ipopt_solver
end

status,time,mem,iter = benchmark(solver,probs;warm_up_probs = [
    ("pglib_opf_case1888_rte.m", ACPPowerModel)
])

results = [name status time mem iter]
flag = QUICK_BENCHMARK ? "short" : "full"
output_file = joinpath(RESULTS_DIR, "powermodels-$(flag)-$(SOLVER).csv")
writedlm(output_file, results)

