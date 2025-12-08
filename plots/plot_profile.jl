
using Comonicon
using DelimitedFiles
using BenchmarkProfiles
using Plots

begin const
    # Ordering of the different columns in the result file
    INSTANCE_COLUMN = 1
    STATUS_COLUMN = 2
    ITER_COLUMN = 3
    TIME_COLUMN = 5
    MEMORY_COLUMN = 6
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

# Comonicon.@main function main(files...; type="iter", name="pprof", format="pdf")

    files = [
        "results/cutest/cutest-madnlp-master-Ma57Solver.csv",
        "results/cutest/cutest-madnlp-adaptive-Ma57Solver.csv",
    ]

    type = "time"
    results = [readdlm(file) for file in files]

    if !all(size(results[1], 1) == size(result, 1) for result in results)
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

    n_problems = size(results[1], 1)

    # We select only the problems having a solution time above THRESHOLD.
    # Incorporating smaller problems can lead to inconsistent performance
    # profile, as we are not guaranteed to discard side effect in the
    # solver's initialization.
    # large = results[1][:, TIME_COLUMN] .> THRESHOLD

    all_results = zeros(n_problems, length(results))
    for (i,result) in enumerate(results)
        all_results[:, i] .= process_results(result, col)
    end

    # all_results = all_results[large, :]

    # display(all_results)

    (label, flag) = if col == TIME_COLUMN
        ("total time spent in solver", "time")
    elseif col == ITER_COLUMN
        ("total number of iterations", "iter")
    elseif col == MEMORY_COLUMN
        ("total memory used", "memory")
    end

    labels = [
        "MadNLP-master",
        "MadNLP-PR-480",
    ]

    performance_profile(
        PlotsBackend(),
        copy(all_results),
        labels,
        title="Performance profile comparing the wall time on CUTEst (tol=1e-6)",
    )
    plot!(titlefontsize=10)

    # Output file.
    # savefig("$(name).$(format)")
# end

