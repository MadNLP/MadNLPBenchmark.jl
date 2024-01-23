
using Comonicon
using DelimitedFiles
using BenchmarkProfiles
using Plots

begin const
    # Ordering of the different columns in the result file
    INSTANCE_COLUMN = 1
    STATUS_COLUMN = 2
    TIME_COLUMN = 3
    MEMORY_COLUMN = 4
    ITER_COLUMN = 5
    # Optimal status
    OPTIMAL_STATUS = 1
    # Threshold
    THRESHOLD = 0.01
end

function process_results(res, col)
    nprobs = size(res, 1)
    values = zeros(nprobs)
    for k in 1:nprobs
        # We keep the result only if optimum is found and the value returned
        # is nonzero, otherwise we set -1 to indicate failure.
        values[k] = if (res[k, STATUS_COLUMN] == OPTIMAL_STATUS) && (res[k, col] != 0)
            res[k, col]
        else
            -1.0
        end
    end
    return values
end

@main function main(file1, file2; type="iter", name="pprof", format="pdf")

    results_1 = readdlm(file1)
    results_2 = readdlm(file2)

    if size(results_1, 1) != size(results_2, 1)
        println("The files $(file1) and $(file2) have different sizes." *
                "Invalid benchmark")
        return
    end

    col = if type == "iter"
        ITER_COLUMN
    elseif type == "mem"
        MEMORY_COLUMN
    else
        TIME_COLUMN # by default, we use the solution time as the main metric.
    end

    n_problems = size(results_1, 1)

    # We select only the problems having a solution time above THRESHOLD.
    # Incorporating smaller problems can lead to inconsistent performance
    # profile, as we are not guaranteed to discard side effect in the
    # solver's initialization.
    large = results_1[:, TIME_COLUMN] .> THRESHOLD

    all_results = zeros(n_problems, 2)
    all_results[:, 1] .= process_results(results_1, col)
    all_results[:, 2] .= process_results(results_2, col)

    all_results = all_results[large, :]

    display(all_results)

    (label, flag) = if col == TIME_COLUMN
        ("total time spent in solver", "time")
    elseif col == ITER_COLUMN
        ("total number of iterations", "iter")
    elseif col == MEMORY_COLUMN
        ("total memory used", "memory")
    end

    performance_profile(
        PlotsBackend(),
        copy(all_results),
        [basename(file1), basename(file2)];
        title="Benchmark $(label)",
    )

    # Output file.
    savefig("$(name).$(format)")
end

