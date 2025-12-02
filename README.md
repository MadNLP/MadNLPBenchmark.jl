# MadNLPBenchmark

This package implements a set of benchmarks for the nonlinear solver MadNLP.
The main entry point is provided as a `Makefile`.

By default, MadNLP is benchmarked against Ipopt.
Both Ipopt and MadNLP requires a proper installation of the library HSL (here provided by `HSL_jll`).

## Installation
To install all the dependencies, run in your shell:
```shell
make install

```
By default, MadNLPBenchmark.jl is set up to use the `master` branch of MadNLP and its dependencies (MadNLPHSL, MadNLPGPU).

To udpate all the dependencies, run:
```shell
make update

```

## Quickstart

For `solver={madnlp|ipopt}` and `benchmark={cops|mittelmann|cutest|acopf|acopf-rect}`, you can
run the benchmark in parallel using `NPROCS` processes using
```shell
julia -p $NPROCS --project=. --solver=$solver --benchmark=$benchmark

```

Alternatively, you can run all the benchmarks with:
```shell
make all

```
The results are stored as a text file in the folder
`results/*`.

## Benchmarks

- **CUTEst:** uses the instances provided in [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl)
- **ACOPF:** uses the ACOPF instances from the [PGLIB benchmark](https://github.com/power-grid-lib/pglib-opf), formulated using [ExaModelsPower.jl](https://github.com/exanauts/ExaModelsPower.jl)
- **COPS:** uses the instances from the [COPS benchmark](https://www.mcs.anl.gov/~more/cops/), formulated using [ExaModels.jl](https://github.com/exanauts/ExaModels.jl)


## Plots
The directory `plots/` stores various scripts to plot the results of the different benchmark,
including performance profiles. The performance profiles are plotted
using the [BenchmarkProfiles.jl](https://github.com/JuliaSmoothOptimizers/BenchmarkProfiles.jl) package, kindly provided by JuliaSmoothOptimizers.

