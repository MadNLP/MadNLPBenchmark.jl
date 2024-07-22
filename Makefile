
JULIAEXEC=julia
NPROCS=1
BENCHMARKS= cutest powermodels exaopf argos cops

install:
	for benchmark in $(BENCHMARKS) ; do \
		echo Install $${benchmark} ; \
		$(JULIAEXEC) --project=$${benchmark} -e "using Pkg; Pkg.instantiate()" ; \
	done

update:
	for benchmark in $(BENCHMARKS) ; do \
		echo Update benchmark $${benchmark} ; \
		$(JULIAEXEC) --project=$${benchmark} -e "using Pkg; Pkg.update()" ; \
	done

test: 
	for benchmark in $(BENCHMARKS) ; do \
		echo $${benchmark} ; \
		$(JULIAEXEC) --project=$${benchmark} $${benchmark}/test.jl ; \
	done

cutest:
	$(JULIAEXEC) --project=cutest -p $(NPROCS) cutest/benchmark.jl --solver all 

lbfgs:
	$(JULIAEXEC) --project=cutest -p $(NPROCS) cutest/lbfgs/benchmark.jl --solver all 

argos:
	$(JULIAEXEC) --project=argos argos/full/benchmark.jl
	$(JULIAEXEC) --project=argos argos/reduced/benchmark.jl

cops-cpu:
	$(JULIAEXEC) --project=cops cops/benchmark.jl --instances default

cops-gpu:
	$(JULIAEXEC) --project=cops cops/benchmark.jl --instances gpu

mittelmann:
	$(JULIAEXEC) --project=cops cops/benchmark.jl --instances mittelmann

all: cutest lbfgs exaopf argos cops-cpu mittelmann

