
JULIAEXEC=julia

all: bench_cutest bench_examodels bench_powermodels

bench_cutest:
	$(JULIAEXEC) --project=cutest cutest/benchmark.jl

bench_examodels: 
	$(JULIAEXEC) --project=exaopf exaopf/benchmark.jl

bench_powermodels: 
	$(JULIAEXEC) --project=powermodels powermodels/benchmark.jl
