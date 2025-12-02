
for file in readdir(joinpath(@__DIR__, "COPS"))
    if endswith(file, ".jl")
        include(joinpath(@__DIR__, "COPS", file))
    end
end

@kwdef struct COPSBenchmark <: AbstractBenchmarkSetting
    config::Symbol = :cops
end

get_tag(bench::COPSBenchmark) = string(bench.config)

function get_instances(bench::COPSBenchmark)
    if bench.config == :cops
        return [
            (bearing_model, (50, 50)),
            (chain_model, (800,)),
            (camshape_model, (1000,)),
            (catmix_model, (100,)),
            (elec_model, (50,)),
            (gasoil_model, (100,)),
            (marine_model, (100,)),
            (methanol_model, (100,)),
            (minsurf_model, (50, 50)),
            (minsurf_model, (50, 75)),
            (minsurf_model, (50, 100)),
            (pinene_model, (100,)),
            (robot_model, (200,)),
            (rocket_model, (400,)),
            (steering_model, (200,)),
            (dirichlet_model, (20,)),
            (henon_model, (10,)),
            (lane_emden_model, (20,)),
        ]
    elseif bench.config == :mittelmann
        return [
            (bearing_model, (400, 400)),
            (camshape_model, (6400,)),
            (dirichlet_model, (120,)),
            (elec_model, (400,)),
            (gasoil_model, (3200,)),
            (henon_model, (120,)),
            (lane_emden_model, (120,)),
            (marine_model, (1600,)),
            (pinene_model, (3200,)),
            (robot_model, (1600,)),
            (rocket_model, (12800,)),
            (steering_model, (12800,)),
        ]
    end
end

function parse_name(instance, ::COPSBenchmark)
    func = instance[1]
    id = split(string(func), '_')[1]
    k = prod(instance[2])
    return "$(id)_$(k)"
end

function load_model(instance, benchmark::COPSBenchmark)
    func = instance[1]
    return func(instance[2]...)
end

