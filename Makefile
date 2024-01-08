
JULIAEXEC=julia
NPROCS=1

install:
	$(JULIAEXEC) --project=cutest -e "using Pkg; Pkg.instantiate()"
	$(JULIAEXEC) --project=powermodels -e "using Pkg; Pkg.instantiate()"
	$(JULIAEXEC) --project=exaopf -e "using Pkg; Pkg.instantiate()"
	$(JULIAEXEC) --project=argos -e "using Pkg; Pkg.instantiate()"

all: 
	$(JULIAEXEC) --project=cutest -p $(NPROCS) cutest/benchmark.jl --solver all 
	$(JULIAEXEC) --project=cutest -p $(NPROCS) cutest/lbfgs/benchmark.jl --solver all 
	$(JULIAEXEC) --project=powermodels -p $(NPROCS) powermodels/benchmark.jl --solver all 
	$(JULIAEXEC) --project=exaopf exaopf/benchmark.jl --solver all 
	$(JULIAEXEC) --project=argos argos/full/benchmark.jl
	$(JULIAEXEC) --project=argos argos/reduced/benchmark.jl

