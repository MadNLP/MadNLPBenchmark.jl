SOLVER=madnlp
JULIAEXEC=julia
NPROCS=1
BENCHMARKS= cutest powermodels exaopf cops

CUTEST_TOL=1e-6
CUTEST_MADNLP_LINEAR_SOLVER=Ma57Solver
CUTEST_IPOPT_LINEAR_SOLVER=ma57
CUTEST_MADNLP_REV=master
CUTEST_QUICK=false
CUTEST_DECODE=false

LBFGS_TOL=1e-6
LBFGS_MADNLP_LINEAR_SOLVER=Ma57Solver
LBFGS_IPOPT_LINEAR_SOLVER=ma57
LBFGS_MADNLP_REV=master
LBFGS_QUICK=false
LBFGS_DECODE=false

 .PHONY: install update test cutest lbfgs cops-cpu cops-gpu mittelmann all

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
	$(JULIAEXEC) --project=cutest -p $(NPROCS) cutest/benchmark.jl --solver $(SOLVER) --tol $(CUTEST_TOL) --madnlp-linear-solver $(CUTEST_MADNLP_LINEAR_SOLVER) --ipopt-linear-solver $(CUTEST_IPOPT_LINEAR_SOLVER) --madnlp-rev $(CUTEST_MADNLP_REV) --quick $(CUTEST_QUICK) --decode $(CUTEST_DECODE)

lbfgs:
	$(JULIAEXEC) --project=lbfgs -p $(NPROCS) lbfgs/lbfgs/benchmark.jl --solver $(SOLVER) --tol $(LBFGS_TOL) --madnlp-linear-solver $(LBFGS_MADNLP_LINEAR_SOLVER) --ipopt-linear-solver $(LBFGS_IPOPT_LINEAR_SOLVER) --madnlp-rev $(LBFGS_MADNLP_REV) --quick $(LBFGS_QUICK) --decode $(CUTEST_DECODE)

cops-cpu:
	$(JULIAEXEC) --project=cops cops/benchmark.jl --instances default

cops-gpu:
	$(JULIAEXEC) --project=cops cops/benchmark.jl --instances gpu

mittelmann:
	$(JULIAEXEC) --project=cops cops/benchmark.jl --instances mittelmann

all: cutest lbfgs exaopf argos cops-cpu mittelmann

