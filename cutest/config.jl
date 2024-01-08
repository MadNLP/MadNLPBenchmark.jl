
using DelimitedFiles
using CUTEst
using NLPModelsIpopt

using HSL
using MadNLP
using MadNLPHSL

EXCLUDE = [
    # MadNLP running into error
    # Ipopt running into error
    "EG3", # lfact blows up
    # Problems that are hopelessly large
    "TAX213322",
    "TAXR213322",
    "TAX53322",
    "TAXR53322",
    "YATP1LS",
    "YATP2LS",
    "YATP1CLS",
    "YATP2CLS",
    "CYCLOOCT",
    "CYCLOOCF",
    "LIPPERT1",
    "GAUSSELM",
    "BA-L52LS",
    "BA-L73LS",
    "BA-L21LS",
]

function decodemodel(name)
    println("Decoding $name")
    finalize(CUTEstModel(name))
end

function evalmodel(name, solver; gcoff=false)
    println("Solving $name")
    nlp = CUTEstModel(name; decode=false)
    try
        gcoff && GC.enable(false);
        mem = @allocated begin
            t = @elapsed begin
                retval = solver(nlp)
            end
        end
        gcoff && GC.enable(true);
        finalize(nlp)
        return (status=get_status(retval.status),time=t,mem=mem,iter=retval.iter)
    catch e
        finalize(nlp)
        return (status=3, time=0.,mem=0,iter=0)
    end
    println("Solved $name")
end

#=
    MadNLP
=#

function madnlp_solver(nlp)
    return madnlp(
        nlp;
        linear_solver=Ma57Solver,
        max_wall_time=900.0,
        print_level=MadNLP.ERROR,
        tol=1e-6,
    )
end

function get_status(code::MadNLP.Status)
    if code == MadNLP.SOLVE_SUCCEEDED
        return 1
    elseif code == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
        return 2
    else
        return 3
    end
end


#=
    Ipopt
=#

function get_status(code::Symbol)
    if code == :first_order
        return 1
    elseif code == :acceptable
        return 2
    else
        return 3
    end
end

function ipopt_solver(nlp)
    return ipopt(
        nlp;
        hsllib=HSL.HSL_jll.libhsl_path,
        linear_solver="ma57",
        max_cpu_time=900.0,
        print_level=0,
        tol=1e-6,
    )
end

#=
    main
=#

function benchmark(solver, probs; warm_up_probs=[], decode=false, gc_off=true)
    println("Warming up (forcing JIT compile)")
    decode && broadcast(decodemodel,warm_up_probs)
    r = [remotecall.(prob->evalmodel(prob,solver;gcoff=gc_off),i,warm_up_probs) for i in procs() if i!= 1]
    fetch.(r)

    println("Decoding problems")
    decode && broadcast(decodemodel,probs)

    println("Solving problems")
    retvals = pmap(prob->evalmodel(prob,solver;gcoff=gc_off),probs)
    status = [retval.status for retval in retvals]
    time   = [retval.time for retval in retvals]
    mem    = [retval.mem for retval in retvals]
    iter   = [retval.iter for retval in retvals]
    return status, time, mem, iter
end
