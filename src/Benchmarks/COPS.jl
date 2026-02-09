
@kwdef struct COPS <: AbstractBenchmarkSetting
    config::Symbol = :cops
    backend::Any = nothing
end

get_tag(bench::COPS) = string(bench.config)

function get_instances(bench::COPS)
    if bench.config == :cops
        return [
            (COPSBenchmark.bearing_model, (50, 50)),
            (COPSBenchmark.chain_model, (800,)),
            (COPSBenchmark.camshape_model, (1000,)),
            (COPSBenchmark.catmix_model, (100,)),
            (COPSBenchmark.elec_model, (50,)),
            (COPSBenchmark.gasoil_model, (100,)),
            (COPSBenchmark.marine_model, (100,)),
            (COPSBenchmark.methanol_model, (100,)),
            (COPSBenchmark.minsurf_model, (50, 50)),
            (COPSBenchmark.minsurf_model, (50, 75)),
            (COPSBenchmark.minsurf_model, (50, 100)),
            (COPSBenchmark.pinene_model, (100,)),
            (COPSBenchmark.robot_model, (200,)),
            (COPSBenchmark.rocket_model, (400,)),
            (COPSBenchmark.steering_model, (200,)),
            (COPSBenchmark.dirichlet_model, (20,)),
            (COPSBenchmark.henon_model, (10,)),
            (COPSBenchmark.lane_emden_model, (20,)),
        ]
    elseif bench.config == :mittelmann
        return [
            (COPSBenchmark.bearing_model, (400, 400)),
            (COPSBenchmark.camshape_model, (6400,)),
            (COPSBenchmark.dirichlet_model, (120,)),
            (COPSBenchmark.elec_model, (400,)),
            (COPSBenchmark.gasoil_model, (3200,)),
            (COPSBenchmark.henon_model, (120,)),
            (COPSBenchmark.lane_emden_model, (120,)),
            (COPSBenchmark.marine_model, (1600,)),
            (COPSBenchmark.pinene_model, (3200,)),
            (COPSBenchmark.robot_model, (1600,)),
            (COPSBenchmark.rocket_model, (12800,)),
            (COPSBenchmark.steering_model, (12800,)),
        ]
    end
end

function parse_name(instance, ::COPS)
    func = instance[1]
    id = split(string(func), '_')[1]
    k = prod(instance[2])
    return "$(id)_$(k)"
end

function load_model(instance, benchmark::COPS)
    func = instance[1]
    return func(instance[2]..., COPSBenchmark.ExaModelsBackend(); backend=benchmark.backend)
end

