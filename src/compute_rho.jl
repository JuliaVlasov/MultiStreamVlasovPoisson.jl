export compute_initial_condition
export compute_rho_total!
export compute_f

function compute_initial_condition(mesh_x::AbstractMesh,grid_v::AbstractGrid,k::Float64,T::Float64)

    nx, nv = mesh_x.nx, grid_v.nv
    rho = zeros(nx + 1, nv)
    u = zeros(nx + 1, nv)
    rho_tot = zeros(nx + 1)
    for j in 1:nv
        alpha = grid_v.v[j]
        for i in 1:(nx + 1)
            x_i = mesh_x.x[i]
            rho[i, j] = f0(x_i, alpha,k,T) / mean_f0(alpha,T)
            u[i, j] = alpha
            rho_tot[i] += grid_v.w[j] * rho[i, j]
        end
    end
    return rho, u, rho_tot
end

function compute_f(mesh_x::AbstractMesh,grid_v::AbstractGrid,rho::Matrix{Float64},u::Matrix{Float64})
    nx = mesh_x.nx
    nv = grid_v.nv
    f = zeros(nx+1,nv)

    #Ask Pierre
    for i in 1:(nx+1)
        loc_v = 0.0
        for j in 1:nv
            loc_v = u[i,j]
            m_loc = 0.0
            for k in 1:nv
                if( abs(loc_v - u[i,k]) < 1e-15)
                    m_loc += grid_v.w[k] * rho[i,k]
                end
            end
            f[i,j] = m_loc
        end  
    end  
    return f
end


function compute_rho_total!(rho_tot::Vector{Float64}, grid_v::AbstractGrid, rho)
    fill!(rho_tot, 0.0)
    for j in axes(rho, 2)
        for i in axes(rho, 1)
            rho_tot[i] += grid_v.w[j] * rho[i, j] 
        end
    end
    return rho_tot
end
