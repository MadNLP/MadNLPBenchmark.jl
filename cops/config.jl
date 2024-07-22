
using DelimitedFiles
using Printf
using LinearAlgebra
using Ipopt
using NLPModelsIpopt

using ExaModels

using HSL_jll
using MadNLP
using MadNLPHSL
using MadNLPGPU
using HybridKKT

using CUDA
using JuMP

using COPSBenchmark

CUDA.allowscalar(false);

const COPS_INSTANCES_DEFAULT = [
    (COPSBenchmark.bearing_model, (50, 50)),
    (COPSBenchmark.chain_model, (800,)),
    (COPSBenchmark.camshape_model, (1000,)),
    (COPSBenchmark.catmix_model, (100,)),
    (COPSBenchmark.channel_model, (200,)),
    (COPSBenchmark.elec_model, (50,)),
    (COPSBenchmark.gasoil_model, (100,)),
    (COPSBenchmark.glider_model, (100,)),
    (COPSBenchmark.marine_model, (100,)),
    (COPSBenchmark.methanol_model, (100,)),
    (COPSBenchmark.minsurf_model, (50, 50)),
    (COPSBenchmark.minsurf_model, (50, 75)),
    (COPSBenchmark.minsurf_model, (50, 100)),
    (COPSBenchmark.pinene_model, (100,)),
    (COPSBenchmark.polygon_model, (100,)),
    (COPSBenchmark.robot_model, (200,)),
    (COPSBenchmark.rocket_model, (400,)),
    (COPSBenchmark.steering_model, (200,)),
    (COPSBenchmark.tetra_duct15_model, ()),
    (COPSBenchmark.tetra_duct20_model, ()),
    (COPSBenchmark.tetra_foam5_model, ()),
    (COPSBenchmark.tetra_gear_model, ()),
    (COPSBenchmark.tetra_hook_model, ()),
    (COPSBenchmark.torsion_model, (50, 50)),
    (COPSBenchmark.dirichlet_model, (20,)),
    (COPSBenchmark.henon_model, (10,)),
    (COPSBenchmark.lane_emden_model, (20,)),
    (COPSBenchmark.triangle_deer_model, ()),
    (COPSBenchmark.triangle_pacman_model, ()),
    (COPSBenchmark.triangle_turtle_model, ()),
]

const COPS_INSTANCES_QUICK = [
    (COPSBenchmark.bearing_model, (50, 50)),
    (COPSBenchmark.camshape_model, (1000,)),
    (COPSBenchmark.elec_model, (50,)),
    (COPSBenchmark.gasoil_model, (100,)),
    (COPSBenchmark.marine_model, (100,)),
    (COPSBenchmark.pinene_model, (100,)),
    (COPSBenchmark.robot_model, (200,)),
    (COPSBenchmark.steering_model, (200,)),
]

const COPS_INSTANCES_MITTELMANN = [
    (COPSBenchmark.bearing_model, (400, 400)),
    (COPSBenchmark.camshape_model, (6400,)),
    (COPSBenchmark.dirichlet_model, (120,)),
    (COPSBenchmark.elec_model, (400,)),
    (COPSBenchmark.gasoil_model, (3200,)),
    (COPSBenchmark.henon_model, (120,)),
    (COPSBenchmark.lane_emden_model, (120,)),
    (COPSBenchmark.marine_model, (1600,)),
    (COPSBenchmark.pinene_model, (3200,)),
    (COPSBenchmark.robot_model, (1600,)),
    (COPSBenchmark.rocket_model, (12800,)),
    (COPSBenchmark.steering_model, (12800,)),
]

# N.B.: subset of instances supported by ExaModels.
const COPS_INSTANCES_GPU = [
    (COPSBenchmark.bearing_model, (400, 400)),
    (COPSBenchmark.camshape_model, (6400,)),
    (COPSBenchmark.elec_model, (400,)),
    (COPSBenchmark.gasoil_model, (3200,)),
    (COPSBenchmark.marine_model, (1600,)),
    (COPSBenchmark.pinene_model, (3200,)),
    (COPSBenchmark.robot_model, (1600,)),
    (COPSBenchmark.rocket_model, (12800,)),
    (COPSBenchmark.steering_model, (12800,)),
    (COPSBenchmark.bearing_model, (800, 800)),
    (COPSBenchmark.camshape_model, (12800,)),
    (COPSBenchmark.elec_model, (800,)),
    (COPSBenchmark.gasoil_model, (12800,)),
    (COPSBenchmark.marine_model, (12800,)),
    (COPSBenchmark.pinene_model, (12800,)),
    (COPSBenchmark.robot_model, (12800,)),
    (COPSBenchmark.rocket_model, (51200,)),
    (COPSBenchmark.steering_model, (51200,)),
]

function parse_name(func, params)
    id = split(string(func), '_')[1]
    k = prod(params)
    return "$(id)_$(k)"
end

function ipopt_status(status)
    if status == :first_order
        return 1
    elseif status == :acceptable
        return 2
    else
        return 3
    end
end

function moi_status(status)
    if status == MOI.LOCALLY_SOLVED
        return 1
    elseif status == MOI.ALMOST_LOCALLY_SOLVED
        return 2
    else
        return 3
    end
end

function madnlp_status(status::MadNLP.Status)
    return Int(status)
end

function _benchmark_madnlp(nlp; gamma=NaN, options...)
    # Warm-up
    solver = MadNLP.MadNLPSolver(nlp; max_iter=1, options...)
    if isfinite(gamma)
        solver.kkt.gamma[] = 1e7
    end
    MadNLP.solve!(solver)
    # Solution
    mem = @allocated begin
        solver = MadNLP.MadNLPSolver(nlp; options...)
        if isfinite(gamma)
            solver.kkt.gamma[] = 1e7
        end
        result = MadNLP.solve!(solver)
    end
    # Refresh memory
    GC.gc(true)
    CUDA.reclaim()
    # Extract solution
    tot_time = result.counters.total_time
    status = madnlp_status(result.status)
    iter = result.counters.k
    obj = result.objective
    return (status, obj, tot_time, mem, iter)
end

function _benchmark_ipopt(nlp; options...)
    # Warm-up
    ipopt(nlp; max_iter=1, options...)
    # Solution
    mem = @allocated begin
        result = ipopt(nlp; options...)
    end
    # Refresh memory
    GC.gc(true)
    # Extract results
    status = ipopt_status(result.status)
    tot_time = result.elapsed_time
    it = result.iter
    obj = result.objective
    return (status, obj, tot_time, mem, it)
end

function solve_madnlp_hykkt_cuda(model::JuMP.Model)
    nlp = ExaModels.ExaModel(model; backend=CUDABackend())
    return _benchmark_madnlp(
        nlp;
        gamma=1e7,
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
end

function solve_madnlp_sckkt_cuda(model::Model)
    nlp = ExaModels.ExaModel(model; backend=CUDABackend())
    return _benchmark_madnlp(
        nlp;
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

function solve_madnlp_ma57(model::Model)
    nlp = ExaModels.ExaModel(model)
    return _benchmark_madnlp(
        nlp;
        disable_garbage_collector=true,
        tol=1e-6,
        dual_initialized = true,
        linear_solver=Ma57Solver,
        print_level=MadNLP.ERROR,
    )
end

function solve_madnlp_ma57_jump(model::Model)
    JuMP.set_optimizer(model, MadNLP.Optimizer)
    JuMP.set_optimizer_attribute(model, "disable_garbage_collector", true)
    JuMP.set_optimizer_attribute(model, "tol", 1e-6)
    JuMP.set_optimizer_attribute(model, "linear_solver", Ma57Solver)
    JuMP.set_optimizer_attribute(model, "print_level", MadNLP.ERROR)
    mem = @allocated begin
        JuMP.optimize!(model)
    end

    st = MOI.get(model, MOI.TerminationStatus())
    status = moi_status(st)
    tot_time = MOI.get(model, MOI.SolveTimeSec())
    iter = 0 # TODO
    obj = MOI.get(model, MOI.ObjectiveValue())
    return (status, obj, tot_time, mem, iter)
end

function solve_ipopt_ma57(model::JuMP.Model)
    nlp = ExaModels.ExaModel(model)
    return _benchmark_ipopt(
        nlp;
        hsllib=HSL_jll.libhsl_path,
        linear_solver="ma57",
        max_cpu_time=900.0,
        print_level=5,
        tol=1e-6,
    )
end

function solve_ipopt_ma57_jump(model::Model)
    JuMP.set_optimizer(model, Ipopt.Optimizer)
    JuMP.set_optimizer_attribute(model, "tol", 1e-6)
    JuMP.set_optimizer_attribute(model, "hsllib", HSL_jll.libhsl_path)
    JuMP.set_optimizer_attribute(model, "linear_solver", "ma57")
    JuMP.set_optimizer_attribute(model, "print_level", 0)
    mem = @allocated begin
        JuMP.optimize!(model)
    end

    st = MOI.get(model, MOI.TerminationStatus())
    status = moi_status(st)
    tot_time = MOI.get(model, MOI.SolveTimeSec())
    iter = 0 # TODO
    obj = MOI.get(model, MOI.ObjectiveValue())
    return (status, obj, tot_time, mem, iter)
end

