
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

function ipopt_solver(nlp)
    return ipopt(
        nlp;
        linear_solver="ma57",
        max_cpu_time=900.0,
        print_level=0,
        tol=1e-6,
    )
end

function get_status(code::Symbol)
    if code == :first_order
        return 1
    elseif code == :acceptable
        return 2
    else
        return 3
    end
end

