export compute_initial_condition
export compute_rho_total!
export compute_norm_dx_u

function compute_initial_condition(test_case::InitialCondition, mesh::UniformMesh, grid::UniformGrid)

    k = test_case.k
    nx, nv = mesh.nx, grid.nv
    rho = zeros(nx, nv)
    u = zeros(nx, nv)
    for j in 1:nv
        v = grid.v[j]
        for i in 1:nx
            x = mesh.x[i]
            rho[i,j] = f0(test_case, x, v) / mean_f0(test_case, mesh, v)
            u[i,j] = v
        end
    end
    return rho, u
end


function compute_rho_total!(rho_tot::Vector{Float64}, grid_v::AbstractGrid, rho)
    fill!(rho_tot, 0.0)

    for j in eachindex(grid_v.w), i in eachindex(rho_tot)
        rho_tot[i] += grid_v.w[j] * rho[j][i]
    end
end

function compute_norm_dx_u(mesh_x::AbstractMesh, grid_v::AbstractGrid, u)

    nv = grid_v.nv
    nx = mesh_x.nx
    dx = mesh_x.dx

    dx_u = 0.0
    for l in 1:nv, i in 1:nx
        du = abs(u[l][i + 1] - u[l][i]) / dx
        dx_u = max(dx_u, du)
    end

    return dx_u
end
