
using Test

include("config.jl")

function test_kkt(case, kkt; use_gpu=false, options...)
    datafile = joinpath(PGLIB_PATH, case)
    model = ExaPF.PolarForm(datafile)
    ntrials = 1

    results = benchmark(model, kkt; use_gpu=use_gpu, ntrials=1, options...)
    @test results.status == 1
    @test results.iters > 0
end

case = "pglib_opf_case14_ieee.m"
ntrials = 1

test_kkt(
    case,
    Argos.FullSpace();
    print_level=MadNLP.ERROR,
    linear_solver=Ma57Solver,
)

if CUDA.has_cuda()
    test_kkt(
        case,
        Argos.BieglerReduction();
        use_gpu=true,
        print_level=MadNLP.ERROR,
        linear_solver=LapackGPUSolver,
    )
    # test_kkt(
    #     case,
    #     Argos.DommelTinney();
    #     use_gpu=true,
    #     print_level=MadNLP.ERROR,
    #     linear_solver=LapackGPUSolver,
    # )
end
