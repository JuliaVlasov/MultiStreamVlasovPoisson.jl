export compute_elec_energy

"""
$(SIGNATURES)
    
Compute the electric energy.
"""
function compute_elec_energy(phi::Vector{Float64}, mesh::AbstractMesh, eps)::Float64
    e = 0.0
    nx, dx = mesh.nx, mesh.dx
    for i in eachindex(phi)
        ir = mod1(i + 1, nx + 1)
        e += 0.5 * eps * eps * dx * (phi[ir] - phi[i])^2 / (dx * dx)
    end
    return e
end
