using Documenter
using MultiPhaseVlasov

ENV["GKSwstype"] = "100"

makedocs(
    sitename = "MultiPhaseVlasov.jl",
    authors = "Mehdi Badsi and Pierre Navaro",
    modules = [MultiPhaseVlasov],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = MathJax(
            Dict(
                :TeX =>
                    Dict(:equationNumbers => Dict(:autoNumber => "AMS"), :Macros => Dict()),
            ),
        ),
    ),
    doctest = false,
    pages = [
        "Documentation" => "index.md",
    ],
    remotes = nothing
)

deploydocs(
    devbranch = "main",
    repo = "gitlab.inria.fr/mingus/code-vp-multistream.git",
)
