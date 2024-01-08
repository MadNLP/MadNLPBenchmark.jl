using Test

include("config.jl")

function test_solver(name, solver)
    nlp = CUTEstModel(name)
    results = solver(nlp)
    @test get_status(results.status) == 1
    @test results.iter > 0
    finalize(nlp)
end

test_solver("HS76", madnlp_solver)
test_solver("HS76", ipopt_solver)

