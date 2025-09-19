
using DelimitedFiles
using CUTEst

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
    # Failure
    "CHARDIS0"
]

function decodemodel(name)
    sifdecoder(name)
    build_libsif(name)
end

function evalmodel(name, solver; gcoff=false)
    println("Solving $name")
    nlp = CUTEstModel{Float64}(name; decode=false)
    try
        gcoff && GC.enable(false);
        mem = @allocated begin
            t = @elapsed begin
                retval = solver(nlp)
            end
        end
        gcoff && GC.enable(true);
        # N.B. *ALWAYS* call explicitly garbage collector
        #      to avoid annoying memory leak as GC was disable before.
        GC.gc(true)
        println("Solved $name")
        finalize(nlp)
        return (status=get_status(retval.status),time=t,mem=mem,iter=retval.iter)
    catch e
        finalize(nlp)
        GC.gc(true)
        return (status=3, time=0.,mem=0,iter=0)
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




