using FFTW
import Statistics: mean

function compute_rho!(
        rho::Vector{Float64},
        meshv::UniformMesh,
        f::Array{Float64, 2}
    )

    dv = meshv.dx
    rho .= dv .* vec(sum(f, dims = 1))
    rho .-= mean(rho)
end

export PoissonSolver

struct PoissonSolver

    modes::Vector{Float64}
    rhok::Vector{ComplexF64}

    function PoissonSolver(meshx)

        nx = meshx.nx
        modes = 2π / (meshx.xmax - meshx.xmin) * collect(fftfreq(nx, nx))
        modes[1] = 1.0
        rhok = zeros(ComplexF64, nx)
        return new(modes, rhok)

    end

end

function compute_e!(e::Vector{Float64}, solver::PoissonSolver, rho::Vector{Float64})
    solver.rhok .= -1im .* fft(rho) ./ solver.modes
    e .= real(ifft(solver.rhok))
end

export poisson!

function poisson!(
    phi::Vector{Float64},
    mesh::UniformMesh,
    rho_tot::Vector{Float64},
    ϵ::Float64,
)

    rho_tot_f=fft(rho_tot .- 1)
    rho_tot_f[1]=0
    kkx=mesh.kx
    kkx[1]=1
    ff_P = (ϵ * ϵ)*kkx .* kkx
    phi.=+real(ifft((rho_tot_f ./ (ff_P))))

end
