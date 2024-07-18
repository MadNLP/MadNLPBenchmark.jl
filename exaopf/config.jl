
using DelimitedFiles
using Printf
using LinearAlgebra
using NLPModels
using NLPModelsIpopt

using HSL_jll
using MadNLP
using MadNLPHSL
using MadNLPGPU
using HybridKKT

using CUDA
using JuMP
using PowerModels

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

function ipopt_status(status)
    if status == :first_order
        return 1
    elseif status == :acceptable
        return 2
    else
        return 3
    end
end

function madnlp_status(status::MadNLP.Status)
    return Int(status)
end

function solve_madnlp_hykkt_cuda(case)
    model = ac_power_model(case; backend=CUDABackend())
    mem = @allocated begin
        solver = MadNLP.MadNLPSolver(
            model;
            disable_garbage_collector=true,
            tol=1e-6,
            dual_initialized = true,
            print_level=MadNLP.ERROR,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL,
            kkt_system=HybridKKT.HybridCondensedKKTSystem,
            equality_treatment=MadNLP.EnforceEquality,
            fixed_variable_treatment=MadNLP.MakeParameter,
        )
        solver.kkt.gamma[] = 1e7
        result = MadNLP.solve!(solver)
    end
    GC.gc()
    tot_time = result.counters.total_time
    status = madnlp_status(result.status)
    iter = result.counters.k
    obj = result.objective
    return (status, obj, tot_time, mem, iter)
end

function solve_madnlp_sckkt_cuda(case)
    model = ac_power_model(case; backend=CUDABackend())
    mem = @allocated begin
        result = madnlp(
            model;
            disable_garbage_collector=true,
            tol=1e-6,
            dual_initialized = true,
            print_level=MadNLP.ERROR,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL,
            kkt_system=MadNLP.SparseCondensedKKTSystem,
            equality_treatment=MadNLP.RelaxEquality,
            fixed_variable_treatment=MadNLP.RelaxBound,
        )
    end
    GC.gc()
    tot_time = result.counters.total_time
    status = madnlp_status(result.status)
    iter = result.counters.k
    obj = result.objective
    return (status, obj, tot_time, mem, iter)
end

function solve_madnlp_ma27(case)
    model = ac_power_model(case)
    mem = @allocated begin
        result = madnlp(
            model;
            disable_garbage_collector=true,
            tol=1e-6,
            dual_initialized = true,
            linear_solver=Ma27Solver,
            print_level=MadNLP.ERROR,
        )
    end
    GC.gc()
    tot_time = result.counters.total_time
    status = madnlp_status(result.status)
    iter = result.counters.k
    obj = result.objective
    return (status, obj, tot_time, mem, iter)
end

function solve_ipopt_ma27(case)
    model = ac_power_model(case)
    mem = @allocated begin
        result = ipopt(
            model;
            hsllib=HSL_jll.libhsl_path,
            linear_solver="ma27",
            max_cpu_time=900.0,
            print_level=0,
            tol=1e-6,
        )
    end
    status = ipopt_status(result.status)
    tot_time = result.elapsed_time
    it = result.iter
    obj = result.objective
    return (status, obj, tot_time, mem, it)
end

