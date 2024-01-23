
using Comonicon
using Printf
using DelimitedFiles

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

function diff_string(str1, str2)
    n1, n2 = length(str1), length(str2)
    str1_ = codeunits(str1)
    str2_ = codeunits(str2)
    start_idx = findfirst(str1_[i] != str2_[i] for i in 1:min(n1, n2))
    stop_idx = findfirst(str1_[n1 - i + 1] != str2_[n2 - i + 1] for i in 1:min(n1, n2))
    return (start_idx, n1 - stop_idx + 1, n2 - stop_idx + 1)
end

@main function main(file1, file2; type="iter", output_total="benchmark.txt", output_fail="failures.txt", result_dir=joinpath(@__DIR__, "..", "results", "scan"))
    res1 = readdlm(file1)
    res2 = readdlm(file2)

    flag1 = split(basename(file1), ".")[1]
    flag2 = split(basename(file2), ".")[1]

    # Find names of each solver by processing the basename
    (start, stop1, stop2) = diff_string(flag1, flag2)
    solver_1 = flag1[start:stop1]
    solver_2 = flag2[start:stop2]

    if size(res1, 1) != size(res2, 1)
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

    ofile1 = joinpath(result_dir, output_total)
    ofile2 = joinpath(result_dir, output_fail)

    @assert size(res1, 1) == size(res2, 1)
    nprobs = size(res1, 1)

    io = open(ofile1, "w")
    @printf(io, "%-10s %13s %13s\n", "", solver_1, solver_2)
    println(io, " "^10, " ", "-"^13, " ", "-"^13)
    for k in 1:nprobs
        instance = res1[k, INSTANCE_COLUMN]
        status1 = res1[k, STATUS_COLUMN]
        status2 = res2[k, STATUS_COLUMN]
        r1 = res1[k, col]
        r2 = res2[k, col]
        @printf(io, "%-10s %6d %6d %6d %6d\n", instance, status1, r1, status2, r2)
    end
    close(io)

    io = open(ofile2, "w")
    @printf(io, "%-10s %13s %13s\n", "", solver_1, solver_2)
    println(io, " "^10, " ", "-"^13, " ", "-"^13)
    for k in 1:nprobs
        instance = res1[k, INSTANCE_COLUMN]
        status1 = res1[k, STATUS_COLUMN]
        status2 = res2[k, STATUS_COLUMN]
        r1 = res1[k, col]
        r2 = res2[k, col]
        if (status1 == 1 && status2 == 3)
            @printf(io, "%-10s %6d %6d %6d %6d\n", instance, status1, r1, status2, r2)
        end
    end
    close(io)
end

