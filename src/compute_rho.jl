export compute_initial_condition
export compute_rho_total!
export compute_norm_dx_u

function compute_initial_condition(mesh_x::AbstractMesh, grid_v::AbstractGrid, k::Float64, T::Float64, test_case::String)

    nx, nv = mesh_x.nx, grid_v.nv
    rho = zeros(nx + 1, nv)
    u = zeros(nx + 1, nv)
    rho_tot = zeros(nx + 1)
    for j in 1:nv
        alpha = grid_v.v[j]
        for i in 1:(nx + 1)
            x_i = mesh_x.x[i]
            rho[i, j] = f0(x_i, alpha, k, T, u_ini(x_i, k, test_case), test_case) / mean_f0(alpha, T, mesh_x, test_case)
            u[i, j] = alpha
            rho_tot[i] += grid_v.w[j] * rho[i, j]
        end
    end
    return rho, u, rho_tot
end


function compute_rho_total!(rho_tot::Vector{Float64}, grid_v::AbstractGrid, rho)
    fill!(rho_tot, 0.0)

    for j in axes(rho, 2), i in axes(rho, 1)
        rho_tot[i] += grid_v.w[j] * rho[i, j]
    end
    return rho_tot
end

function compute_norm_dx_u(mesh_x::AbstractMesh, grid_v::AbstractGrid, u::Matrix{Float64})

    nv = grid_v.nv
    nx = mesh_x.nx
    dx = mesh_x.dx

    dx_u = 0.0
    for l in 1:nv, i in 1:nx
        du = abs(u[i + 1, l] - u[i, l]) / dx
        dx_u = max(dx_u, du)
    end

    return dx_u
end
