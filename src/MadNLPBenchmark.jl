module MadNLPBenchmark

import Random
import NLPModels
import ExaModels
# Solvers
import MadNLP
import NLPModelsIpopt
import HSL_jll
import MadNLPHSL
# Benchmarks
import CUTEst
import ExaModelsPower
import COPSBenchmark

include("Benchmarks/Benchmarks.jl")
include("Solvers/Solvers.jl")

include("routines.jl")

end
