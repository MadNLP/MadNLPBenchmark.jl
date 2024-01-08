
using Test

include("config.jl")

function test_solver(name, solver)
    type = ACPPowerModel
    results = evalmodel((name, type), solver)
    @test results[1] == 1
    @test results[4] > 0
end

name = "pglib_opf_case14_ieee.m"

test_solver(name, madnlp_solver)
test_solver(name, ipopt_solver)

