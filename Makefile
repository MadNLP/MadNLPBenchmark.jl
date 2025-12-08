SOLVER=madnlp
JULIAEXEC=julia
NPROCS=10
BENCHMARKS= cutest powermodels exaopf cops

MADNLP_REV=master
IPOPT_REV=main

.PHONY: install update latest cutest cops mittelmann all

install:
	$(JULIAEXEC) --project=. -e "using Pkg; Pkg.instantiate()" 

update:
	$(JULIAEXEC) --project=. -e "using Pkg; Pkg.update()" 

latest: 
	$(JULIAEXEC) --project=. -e 'import Pkg; Pkg.add(name="MadNLP", rev="$(MADNLP_REV)"); Pkg.add(name="MadNLPHSL", rev="$(MADNLP_REV)"); Pkg.add(name="Ipopt_jll", rev="$(IPOPT_REV)")'

cutest:
	$(JULIAEXEC) -p $(NPROCS) --project=. main.jl --solver=ipopt --tol=1e-6 --automatic-scaling --benchmark=cutest
	$(JULIAEXEC) -p $(NPROCS) --project=. main.jl --solver=madnlp --tol=1e-6 --automatic-scaling --benchmark=cutest

acopf:
	$(JULIAEXEC) -p $(NPROCS) --project=. main.jl --solver=ipopt-ma27 --benchmark=acopf
	$(JULIAEXEC) -p $(NPROCS) --project=. main.jl --solver=madnlp-ma27 --benchmark=acopf
	$(JULIAEXEC) -p $(NPROCS) --project=. main.jl --solver=ipopt-ma27 --benchmark=acopf-rect
	$(JULIAEXEC) -p $(NPROCS) --project=. main.jl --solver=madnlp-ma27 --benchmark=acopf-rect

cops:
	$(JULIAEXEC) -p $(NPROCS) --project=. main.jl --solver=ipopt --benchmark=cops
	$(JULIAEXEC) -p $(NPROCS) --project=. main.jl --solver=madnlp --benchmark=cops

mittelmann:
	$(JULIAEXEC) -p $(NPROCS) --project=. main.jl --solver=ipopt --benchmark=mittelmann
	$(JULIAEXEC) -p $(NPROCS) --project=. main.jl --solver=madnlp --benchmark=mittelmann

all: cutest cops mittelmann acopf

