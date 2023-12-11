
using DelimitedFiles
using CUTEst
using NLPModelsIpopt

using MadNLP
using MadNLPHSL

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
