
using DelimitedFiles
using MadNLP, MadNLPHSL, MadNLPGPU
using JuMP, Ipopt, CUDA, NLPModels
using PowerModels
using Printf
using LinearAlgebra

include("model.jl")

if haskey(ENV, "PGLIB_PATH")
    const PGLIB_PATH = ENV["PGLIB_PATH"]
else
    error("Unable to find path to PGLIB benchmark.\n"*
        "Please set environment variable `PGLIB_PATH` to run benchmark with PowerModels.jl")
end

CUDA.allowscalar(false);
PowerModels.silence()

pglib_cases = map(
    v -> (
        split(split(v, "case")[2], ".")[1],
        joinpath(PGLIB_PATH, v)
    ),
    filter(startswith("pglib_opf"), readdir(PGLIB_PATH))
)

function ipopt_stats(fname)
    output = read(fname, String)
    iter = parse(Int, split(split(output, "Number of Iterations....:")[2], "\n")[1])
    tot = parse(Float64,split(split(output, "Total seconds in IPOPT                               =")[2], "\n")[1])
    return iter, tot
end

function varcon(n)
    if n <= 1000
        "$n"
    elseif n <= 1000000'
        @sprintf("%5.1fk", n/1000)
    else
        @sprintf("%5.1fm", n/1000000)
    end
end

fmt(t) = @sprintf("%5.2f", t)
efmt(t) = @sprintf("%1.8e", t)
percent(t) = @sprintf("%5.1f", t * 100) * "\\%"

function termination_code(status::TerminationStatusCode)
    if status == LOCALLY_SOLVED
        return "o"
    elseif status == ALMOST_LOCALLY_SOLVED
        return "a"
    elseif status == INFEASIBLE_OR_UNBOUNDED
        return "i"
    else
        return "f"
    end
end

function termination_code(status::MadNLP.Status)
    if status == MadNLP.SOLVE_SUCCEEDED
        return "o"
    elseif status == MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL
        return "a"
    elseif status == MadNLP.DIVERGING_ITERATES || status == MadNLP.DIVERGING_ITERATES
        return "i"
    else
        return "f"
    end
end

function evaluate(m::AbstractNLPModel, result)
    constraints = similar(result.solution, m.meta.ncon)
    NLPModels.cons!(m, result.solution, constraints)
    return result.objective, max(
        norm(min.(result.solution .- m.meta.lvar, 0), Inf),
        norm(min.(m.meta.uvar .- result.solution, 0), Inf),
        norm(min.(constraints .- m.meta.lcon, 0), Inf),
        norm(min.(m.meta.ucon .- constraints, 0), Inf)
    )
end

function evaluate(m::JuMP.Model, m2 = nothing)
    if m2!= nothing
        @assert value.(all_variables(m)) == m.moi_backend.optimizer.model.inner.x
        m.moi_backend.optimizer.model.inner.x .= value.(all_variables(m2))
    end
    model = m.moi_backend.optimizer.model
    lvar = model.variables.lower
    uvar = model.variables.upper
    x = model.inner.x
    g = model.inner.g

    model.inner.eval_g(x,g)
    lcon, ucon = copy(model.qp_data.g_L), copy(model.qp_data.g_U)
    for bound in model.nlp_data.constraint_bounds
        push!(lcon, bound.lower)
        push!(ucon, bound.upper)
    end

    return model.inner.eval_f(x), max(
        norm(min.(x .- lvar, 0), Inf),
        norm(min.(uvar .- x, 0), Inf),
        norm(min.(g .- lcon, 0), Inf),
        norm(min.(ucon .- g, 0), Inf)
    )
end

function evaluate(fname::String)
    output = read(fname, String)
    o = parse(Float64, split(split(split(output, "Objective...............:   ")[2], "    ")[2],"\n")[1])
    c = parse(Float64, split(split(split(output, "Constraint violation....:   ")[2], "    ")[2],"\n")[1])
    return o, c
end

function madnlp_eval_cuda(case)
    model = ac_power_model(case; backend=CUDABackend())
    mem = @allocated begin
        result = madnlp(
            model;
            disable_garbage_collector=true,
            tol=1e-4,
            dual_initialized = true,
            print_level=MadNLP.ERROR,
        )
    end
    GC.gc()
    tot_time = result.counters.total_time
    status = termination_code(result.status)
    iter = result.counters.k
    return (status, tot_time, mem, iter)
end

function madnlp_eval_cpu(case)
    model = ac_power_model(case)
    mem = @allocated begin
        result = madnlp(
            model;
            disable_garbage_collector=true,
            tol=1e-4,
            dual_initialized = true,
            linear_solver=Ma57Solver,
            print_level=MadNLP.ERROR,
        )
    end
    GC.gc()
    tot_time = result.counters.total_time
    status = termination_code(result.status)
    iter = result.counters.k
    return (status, tot_time, mem, iter)
end

function ipopt_eval(case)
    model = jump_ac_power_model(case)
    set_optimizer(model, Ipopt.Optimizer)
    set_optimizer_attribute(model, "linear_solver", "ma57")
    set_optimizer_attribute(model, "tol", 1e-4)
    set_optimizer_attribute(model, "bound_relax_factor", 1e-4)
    set_optimizer_attribute(model, "output_file", "jump_output")
    set_optimizer_attribute(model, "dual_inf_tol", 10000.0)
    set_optimizer_attribute(model, "constr_viol_tol", 10000.0)
    set_optimizer_attribute(model, "compl_inf_tol", 10000.0)
    set_optimizer_attribute(model, "honor_original_bounds", "no")
    mem = @allocated begin
        JuMP.optimize!(model)
    end
    status = termination_code(JuMP.termination_status(model))
    it, tot = ipopt_stats("jump_output")
    return (status, tot, mem, it)
end

