# MadNLPBenchmark

This package implements a set of benchmarks for the nonlinear solver MadNLP.
MadNLPBenchmark.jl is using Makefile as an interface to install and run the
different benchmarks. We recommend using it on a UNIX platform.

MadNLP is benchmarked against Ipopt in the different benchmark.
Both Ipopt and MadNLP requires a proper installation of `HSL.jl`
to run the benchmark. They are both using the linear solver HSL ma57 as a reference.

## Installation
To install all the dependencies, run in your shell:
```shell
make install

```
By default, MadNLPBenchmark.jl is set up to use the `master` branch
of MadNLP and its dependencies (MadNLPHSL, MadNLPGPU).

To udpate all the dependencies, run:
```shell
make update

```
You can test if the benchmark is set up correctly by running:
```shell
make test

```


## Quickstart
If you want to run the benchmark in the directory `bench`, run:

```shell
julia --project=bench bench/benchmark.jl

```
The results are stored as a text file in the folder
`results/bench/*`.

Run all the benchmarks with:
```shell
make all

```

## Benchmarks

### CUTEst
Benchmark MadNLP and Ipopt on the CUTEst benchmark, using the instances provided by the package
[CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl).
```shell
julia --project=cutest -p 4 cutest/benchmark.jl --solver all

```
The argument `-p` sets the number of processors used for the benchmark.
The argument `--solver` specifies the solver to benchmark (`madnlp` launches
MadNLP, `ipopt` Ipopt and `all` both MadNLP and Ipopt).


#### LBFGS
Alternatively, you can benchmark the performance of the LBFGS
algorithms implemented in MadNLP and Ipopt as:
```shell
julia --project=cutest -p 4 cutest/lbfgs/benchmark.jl --solver all

```


### Optimal power flow: PowerModels
Benchmark MadNLP and Ipopt on the PGLIB benchmark,
using the different AC-OPF formulations provided by
the `PowerModels.jl` package (built on top of JuMP).

The path to the PGLIB benchmark should be stored inside
the environment variable `PGLIB_PATH`. E.g.
```shell
export PGLIB_PATH=/home/user/path/to/pglib-opf

```
Once `PGLIB_PATH` defined, one can run the OPF benchmark as
```shell
julia --project=powermodels -p 4 powermodels/benchmark.jl --solver all

```

### Optimal power flow: ExaOPF
Benchmark MadNLP and Ipopt on the PGLIB benchmark, using ExaModels.jl as a backend (instead of JuMP).
ExaModels can evaluate the derivatives either on the CPU or on a NVIDIA GPU.
On the contrary to PowerModels.jl, only the polar AC-OPF formulation of
optimal power flow is implemented with ExaModels.jl.
Once the `PGLIB_PATH` set in the environment, one can run the benchmark with:
```shell
julia --project=exaopf exaopf/benchmark.jl --solver all

```
As it is designed to run on the GPU, we cannot run the ExaOPF benchmark
on more than one process.


## Plots
The directory `plots/` stores various scripts to plot the results of the different benchmark,
including performance profiles. The performance profiles are plotted
using the [BenchmarkProfiles.jl](https://github.com/JuliaSmoothOptimizers/BenchmarkProfiles.jl) package, kindly provided by JuliaSmoothOptimizers.

The directory has two scripts:
1. `plots/plot_profile.jl` shows the result of the benchmark as a performance profile.
    ```shell
    julia --project=plots plots/plot_profile.jl results/cutest/cutest-quick-{ipopt,madnlp}.csv --type "iter"

    ```
1. `plots/scan.jl` scans the benchmark and return a text file showing the detailed results.
    ```shell
    julia --project=plots plots/scan.jl results/cutest/cutest-quick-{ipopt,madnlp}.csv --type "iter"

    ```
    The script writes two files in the directory `results/scan`: `results/scan/benchmark.txt` shows all the benchmark, whereas
    `results/scan/failures.txt` shows only the instance when the first solver is successful and the
    second solver failed.

