
const COLS = String[
    "instance",
    "n",
    "m",
    "nnzj",
    "nnzh",
    "status",
    "obj",
    "it",
    "solve_time",
]

function introduce(benchmark::AbstractBenchmarkSetting, solver::AbstractSolverSetup)
    return "$(get_tag(benchmark))-$(get_solver(solver))-$(get_linear_solver(solver))"
end

function evaluate_model(
    benchmark::AbstractBenchmarkSetting,
    solver::AbstractSolverSetup,
    instance;
    options...,
)
    @info instance
    model = load_model(instance, benchmark)
    results = solve_model(solver, model; options...)
    finalize(model)
    # N.B. *ALWAYS* call explicitly garbage collector
    #      to avoid annoying memory leak as GC was disable before.
    GC.gc(true)
    return results
end

function run_benchmark(benchmark::AbstractBenchmarkSetting, solver::AbstractSolverSetup; options...)
    return run_benchmark(benchmark, get_instances(benchmark), solver; options...)
end
function run_benchmark(benchmark::AbstractBenchmarkSetting, instances::Vector, solver::AbstractSolverSetup; options...)
    m, n = length(instances), 8
    results = zeros(m, n)
    for (k, instance) in enumerate(instances)
        results[k, :] .= evaluate_model(benchmark, solver, instance; options...)
    end
    names = parse_name.(instances, Ref(benchmark))
    return [reshape(COLS, 1, n+1); names results]
end

