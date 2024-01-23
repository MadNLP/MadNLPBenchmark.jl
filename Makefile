
JULIAEXEC=julia
NPROCS=1
BENCHMARKS= cutest powermodels exaopf argos

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

all: 
	$(JULIAEXEC) --project=cutest -p $(NPROCS) cutest/benchmark.jl --solver all 
	$(JULIAEXEC) --project=cutest -p $(NPROCS) cutest/lbfgs/benchmark.jl --solver all 
	$(JULIAEXEC) --project=powermodels -p $(NPROCS) powermodels/benchmark.jl --solver all 
	$(JULIAEXEC) --project=exaopf exaopf/benchmark.jl --solver all 
	$(JULIAEXEC) --project=argos argos/full/benchmark.jl
	$(JULIAEXEC) --project=argos argos/reduced/benchmark.jl

