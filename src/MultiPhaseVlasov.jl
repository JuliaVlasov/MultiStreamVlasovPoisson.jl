module MultiPhaseVlasov

using DocStringExtensions
import DispersionRelations
using FFTW
using .Threads

include("grid.jl")
include("mesh.jl")
include("initial_condition.jl")
include("compute_rho.jl")
include("semi_lagrangian.jl")
include("non_linear_poisson_solver.jl")
include("compute_physical_quantity.jl")
include("single_fluid_solution.jl")
include("compute_f.jl")

end
