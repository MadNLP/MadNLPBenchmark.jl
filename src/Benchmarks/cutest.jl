
@kwdef struct CUTEstBenchmark <: AbstractBenchmarkSetting
    min_var::Int = 100
    max_var::Any = Inf
    min_con::Int = 1
end

function get_tag(bench::CUTEstBenchmark)
    return isinf(bench.max_var) ? "cutest" : "cutest-maxvar$(bench.max_var)"
end

const CUTEST_EXCLUDE = [
    # MadNLP running into error
    # Ipopt running into error
    "EG3", # lfact blows up
    # Problems that are hopelessly large
    "TAX213322",
    "TAXR213322",
    "TAX53322",
    "TAXR53322",
    "YATP1LS",
    "YATP2LS",
    "YATP1CLS",
    "YATP2CLS",
    "CYCLOOCT",
    "CYCLOOCF",
    "LIPPERT1",
    "GAUSSELM",
    "BA-L52LS",
    "BA-L73LS",
    "BA-L21LS",
    "BA-L52",
    "NONSCOMPNE",
    "LOBSTERZ",
    # Failure
    "CHARDIS0"
]

function get_instances(bench::CUTEstBenchmark)
    probs = CUTEst.select_sif_problems(; min_var=bench.min_var, min_con=bench.min_con, max_var=bench.max_var)
    filter!(e -> !(e in CUTEST_EXCLUDE), probs)
    return probs
end

function parse_name(instance, ::CUTEstBenchmark)
    return instance
end

function _decode_cutest_model(instance)
    @info "Decode $(instance)"
    CUTEst.sifdecoder(instance)
    CUTEst.build_libsif(instance)
    return
end

function decode_cutest(instances)
    _decode_cutest_model.(instances)
    return
end

function load_model(instance::String, ::CUTEstBenchmark)
    return CUTEst.CUTEstModel(instance; decode=false)
end

