export compute_elec_energy
export compute_total_mass
export compute_momentum
export compute_kinetic_energy

"""
$(SIGNATURES)
    
Compute the electric energy.
"""
function compute_elec_energy(phi::Vector{Float64}, mesh::AbstractMesh; ϵ = 1.0)::Float64
    e = 0.0
    nx, dx = mesh.nx, mesh.dx
    for i in eachindex(phi)
        ir = mod1(i + 1, nx)
        e += 0.5 * ϵ * ϵ * dx * (phi[ir] - phi[i])^2 / (dx * dx)
    end
    return sqrt(e)
end

function compute_total_mass(rho::Matrix{Float64}, mesh::AbstractMesh, grid::AbstractGrid)

    mass = mesh.dx * sum(rho .* grid.w')

end

function compute_momentum(rho, u, mesh::AbstractMesh, grid_v::AbstractGrid)
    momentum = 0.0
    nx, dx = mesh.nx, mesh.dx
    nv = grid_v.nv
    for l in 1:nv, i in 1:nx
        momentum += dx * grid_v.w[l] * rho[i,l] * u[i,l]
    end
    return momentum
end

function compute_kinetic_energy(rho, u, mesh::AbstractMesh, grid_v::AbstractGrid)
    kinetic_energy = 0.0
    nx, dx = mesh.nx, mesh.dx
    nv = grid_v.nv
    for l in 1:nv, i in 1:nx
        kinetic_energy += dx * grid_v.w[l] * 0.5 * rho[i,l] * u[i,l] * u[i,l]
    end
    return kinetic_energy
end
