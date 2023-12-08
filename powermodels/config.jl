
using DelimitedFiles
using PowerModels
using MathOptInterface
using JuMP

using Ipopt
using MadNLP
using MadNLPHSL

if haskey(ENV, "PGLIB_PATH")
    const PGLIB_PATH = ENV["PGLIB_PATH"]
else
    error("Unable to find path to PGLIB benchmark.\n"*
        "Please set environment variable `PGLIB_PATH` to run benchmark with PowerModels.jl")
end

PowerModels.silence()

function evalmodel(prob, solver; gcoff=false)
    case, type = prob
    pm = instantiate_model(joinpath(PGLIB_PATH,case),type,PowerModels.build_opf)
    println("Solving $(get_name(pm))")
    gcoff && GC.enable(false);
    retval = solver(pm)
    gcoff && GC.enable(true);
    return retval
end

function get_status(code::MOI.TerminationStatusCode)
    if code == MOI.LOCALLY_SOLVED
        return 1
    elseif code == MOI.ALMOST_OPTIMAL
        return 2
    else
        return 3
    end
end

get_name(pm) = "$(pm.data["name"])-$(typeof(pm))"

#=
    MadNLP solver
=#
function madnlp_solver(pm)
    set_optimizer(
        pm.model, ()-> MadNLP.Optimizer(
            linear_solver=Ma57Solver,
            max_wall_time=900.0,
            tol=1e-6,
            print_level=MadNLP.ERROR,
        )
    )
    mem= @allocated begin
        t= @elapsed begin
            optimize_model!(pm)
        end
    end
    return get_status(termination_status(pm.model)), t, mem, barrier_iterations(pm.model)
end

#=
    Ipopt solver
=#

const ITER = [-1]

function ipopt_callback(
    alg_mod::Cint,iter_count::Cint,obj_value::Float64,
    inf_pr::Float64,inf_du::Float64,mu::Float64,d_norm::Float64,
    regularization_size::Float64,alpha_du::Float64,alpha_pr::Float64,ls_trials::Cint)
    ITER[] += 1
    return true
end

function ipopt_solver(pm)
    ITER[] = 0
    set_optimizer(pm.model, Ipopt.Optimizer)
    set_optimizer_attributes(
        pm.model,
        "linear_solver"=>"ma57",
        "max_cpu_time"=>900.0,
        "tol"=>1e-6,
        "print_level"=>0,
    )
    MOI.set(pm.model, Ipopt.CallbackFunction(), ipopt_callback)
    mem = @allocated begin
        t = @elapsed begin
            optimize_model!(pm)
        end
    end
    return get_status(termination_status(pm.model)), t, mem, ITER[]
end

