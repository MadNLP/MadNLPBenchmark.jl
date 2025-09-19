#=
    Ipopt
=#

ipopt_linear_solver = eval(Symbol(ipopt_linear_solver))

using NLPModelsIpopt
using HSL_jll

function get_status(code::Symbol)
    if code == :first_order
        return 1
    elseif code == :acceptable
        return 2
    else
        return 3
    end
end

function ipopt_solver(nlp)
    return ipopt(
        nlp;
        hsllib=HSL_jll.libhsl_path,
        linear_solver=ipopt_linear_solver,
        max_cpu_time=900.0,
        print_level=0,
        tol=tol,
    )
end

