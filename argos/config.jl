using DelimitedFiles

using CUDA

using NLPModels

using MadNLP
using MadNLPHSL
using MadNLPGPU

using ExaPF
using Argos
using ArgosCUDA

const PS = ExaPF.PowerSystem

if haskey(ENV, "PGLIB_PATH")
    const PGLIB_PATH = ENV["PGLIB_PATH"]
else
    error("Unable to find path to PGLIB benchmark.\n"*
          "Please set environment variable `PGLIB_PATH` to run benchmark with PowerModels.jl")
end

const BASE_DIR = joinpath(@__DIR__, "..", "results")
const IS_GPU_AVAILABLE = CUDA.has_cuda() && CUDA.functional()


if !isdir(BASE_DIR)
    mkdir(BASE_DIR)
end

if IS_GPU_AVAILABLE
    CUDA.allowscalar(false)
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

function refresh_memory()
    GC.gc(true)
    CUDA.has_cuda() && CUDA.reclaim()
    return
end

