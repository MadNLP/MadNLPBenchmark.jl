
using Test

include("config.jl")

function test_solver(case, solver)
    prob = joinpath(PGLIB_PATH, case)
    results = solver(prob)
    @test results[1] == "o"
    @test results[4] > 0
end

case = "pglib_opf_case14_ieee.m"
test_solver(case, madnlp_eval_cpu)
test_solver(case, ipopt_eval)

if CUDA.has_cuda()
    test_solver(case, madnlp_eval_cuda)
end

