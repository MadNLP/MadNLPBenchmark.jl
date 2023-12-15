
using Comonicon
using Distributed

@everywhere include("config.jl")

BASE_DIR = joinpath(@__DIR__, "..", "results")
RESULTS_DIR = joinpath(BASE_DIR, "powermodels")


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

@main function main(; solver="madnlp", quick::Bool=false)
    if !isdir(RESULTS_DIR)
        mkpath(RESULTS_DIR)
    end

    if quick
        cases = filter!(e->(occursin("pglib_opf_case",e) && occursin("pegase",e)),readdir(PGLIB_PATH))
        types = [ACPPowerModel]
    else
        cases = filter!(e->occursin("pglib_opf_case",e),readdir(PGLIB_PATH))
        types = [ACPPowerModel, ACRPowerModel, ACTPowerModel,
                DCPPowerModel, DCMPPowerModel, NFAPowerModel,
                DCPLLPowerModel,LPACCPowerModel, SOCWRPowerModel,
                QCRMPowerModel,QCLSPowerModel]
    end
    flag = quick ? "short" : "full"

    probs = [(case,type) for case in cases for type in types]
    name =  ["$case-$type" for case in cases for type in types]

    run_madnlp = (solver == "madnlp") || (solver == "all")
    run_ipopt = (solver == "ipopt") || (solver == "all")

    if run_madnlp
        @info "Benchmark MadNLP"
        status,time,mem,iter = benchmark(madnlp_solver,probs;warm_up_probs = [
            ("pglib_opf_case1888_rte.m", ACPPowerModel)
        ])
        results = [probs status time mem iter]
        output_file = joinpath(RESULTS_DIR, "powermodels-$(flag)-madnlp.csv")
        writedlm(output_file, results)
    end
    if run_ipopt
        @info "Benchmark Ipopt"
        status,time,mem,iter = benchmark(ipopt_solver,probs;warm_up_probs = [
            ("pglib_opf_case1888_rte.m", ACPPowerModel)
        ])
        results = [probs status time mem iter]
        output_file = joinpath(RESULTS_DIR, "powermodels-$(flag)-ipopt.csv")
        writedlm(output_file, results)
    end
end
