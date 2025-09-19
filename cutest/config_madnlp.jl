#=
    MadNLP
=#

using MadNLPHSL

madnlp_linear_solver = eval(Symbol(madnlp_linear_solver))

function madnlp_solver(nlp)
    return madnlp(
        nlp;
        linear_solver=madnlp_linear_solver,
        max_wall_time=900.0,
        print_level=MadNLP.ERROR,
        tol=tol,
    )
end

function get_status(code::MadNLP.Status)
    return Int(code)
end

