
abstract type AbstractSolverSetup end

function get_name end
function get_linear_solver end
function solve_instance end

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
    Ipopt
=#

@kwdef mutable struct IpoptSolver <: AbstractSolverSetup
    linear_solver::String = "ma57"
    max_iter::Int = 1000
    tol::Float64 = 1e-8
    max_wall_time::Float64 = 900.0
    automatic_scaling::Bool = false
end

get_solver(solver::IpoptSolver) = "ipopt"
get_linear_solver(solver::IpoptSolver) = solver.linear_solver

function solve_model(solver::IpoptSolver, nlp::NLPModels.AbstractNLPModel; warmup=true, options...)
    # ExaModels requires a proper warm-up for every new model
    if warmup
        NLPModelsIpopt.ipopt(nlp; hsllib=HSL_jll.libhsl, linear_solver=solver.linear_solver, max_iter=1, print_level=0)
    end

    elapsed = @elapsed begin
        stats = NLPModelsIpopt.ipopt(
            nlp;
            hsllib=HSL_jll.libhsl,
            linear_solver=solver.linear_solver,
            tol=solver.tol,
            max_iter=solver.max_iter,
            max_wall_time=solver.max_wall_time,
            ma57_automatic_scaling=solver.automatic_scaling ? "yes" : "no",
            print_level=0,
            options...
        )
    end
    return (
        NLPModels.get_nvar(nlp),
        NLPModels.get_ncon(nlp),
        NLPModels.get_nnzj(nlp),
        NLPModels.get_nnzh(nlp),
        get_status(stats.status),
        stats.objective,
        stats.iter,
        elapsed,
    )
end

#=
    MadNLP
=#

@kwdef mutable struct MadNLPDefault <: AbstractSolverSetup
    linear_solver = MadNLPHSL.Ma57Solver
    max_iter::Int = 1000
    tol::Float64 = 1e-8
    max_wall_time::Float64 = 900.0
    automatic_scaling::Bool = false
end

get_solver(solver::MadNLPDefault) = "madnlp"
get_linear_solver(solver::MadNLPDefault) = string(solver.linear_solver)

function solve_model(solver::MadNLPDefault, nlp::NLPModels.AbstractNLPModel; warmup=true, options...)
    # ExaModels requires a proper warm-up for every new model
    if warmup
        MadNLP.madnlp(nlp; linear_solver=solver.linear_solver, max_iter=1, print_level=MadNLP.ERROR)
    end

    stats = MadNLP.madnlp(
        nlp;
        linear_solver=solver.linear_solver,
        tol=solver.tol,
        max_iter=solver.max_iter,
        max_wall_time=solver.max_wall_time,
        ma57_automatic_scaling=solver.automatic_scaling,
        print_level=MadNLP.ERROR,
        options...
    )
    return (
        NLPModels.get_nvar(nlp),
        NLPModels.get_ncon(nlp),
        NLPModels.get_nnzj(nlp),
        NLPModels.get_nnzh(nlp),
        Int(stats.status),
        stats.objective,
        stats.iter,
        stats.counters.total_time,
    )
end

