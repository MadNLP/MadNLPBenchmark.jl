
using MadNLPBenchmark
using NLPModels
using MadNLPGPU
import MadNLPGPU: CUDABackend


@kwdef mutable struct MadNLPCUDA <: MadNLPBenchmark.AbstractSolverSetup
    linear_solver = MadNLPGPU.CUDSSSolver
    max_iter::Int = 1000
    tol::Float64 = 1e-6
    max_wall_time::Float64 = 900.0
end

MadNLPBenchmark.get_solver(solver::MadNLPCUDA) = "madnlp-cuda"
MadNLPBenchmark.get_linear_solver(solver::MadNLPCUDA) = string(solver.linear_solver)

function MadNLPBenchmark.solve_model(solver::MadNLPCUDA, nlp::NLPModels.AbstractNLPModel; warmup=true, options...)
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


