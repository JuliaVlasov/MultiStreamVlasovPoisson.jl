export compute_initial_condition
export compute_rho_total!
export compute_norm_dx_u

function compute_initial_condition(test_case::InitialCondition, mesh_x::AbstractMesh, grid_v::AbstractGrid)

    k = test_case.k
    nx, nv = mesh_x.nx, grid_v.nv
    rho = [zeros(nx + 1) for i in 1:nv]
    u = [zeros(nx + 1) for i in 1:nv]
    rho_tot = zeros(nx + 1)
    for j in 1:nv
        alpha = grid_v.v[j]
        for i in 1:(nx + 1)
            x_i = mesh_x.x[i]
            rho[j][i] = f0(test_case, x_i, alpha) / mean_f0(test_case, mesh_x, alpha)
            u[j][i] = alpha
            rho_tot[i] += grid_v.w[j] * rho[j][i]
        end
    end
    return rho, u, rho_tot
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
