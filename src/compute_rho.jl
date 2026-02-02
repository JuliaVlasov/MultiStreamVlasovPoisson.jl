export compute_rho_total!

"""
$(SIGNATURES)
    
Compute the total density from the density matrix.
"""
function compute_rho_total!(rho_tot::Vector{Float64}, mesh::GaussHermiteMesh, rho)

    fill!(rho_tot, 0.0)

    for j in axes(rho, 2), i in axes(rho, 1)
        alpha = mesh.x[j]
        rho_tot[i] += mesh.w[j] * rho[i, j] * mean_f0(alpha) * exp(alpha * alpha)
    end

    return
end


function compute_rho_total!(rho_tot::Vector{Float64}, mesh::UniformMesh, rho)

    fill!(rho_tot, 0.0)
    ng = mesh.ng
    vmin, vmax = mesh.vmin, mesh.vmax

    for j in axes(rho, 2)
        alpha = vmin + (j - 1) * (vmax - vmin) / (ng - 1)
        for i in axes(rho, 1)
            x_i = (i - 1) * mesh.dx
            rho_tot[i] += rho[i, j] * mean_f0(alpha) / mesh.sf0
        end
    end

    return
end
