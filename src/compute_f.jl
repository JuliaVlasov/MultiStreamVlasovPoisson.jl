export interpolate_f_on_grid

function interpolate_f_on_grid(mesh_x::AbstractMesh, grid_v::AbstractGrid, rho::Matrix{Float64}, u::Matrix{Float64})

    nx = mesh_x.nx
    nv = grid_v.nv
    dv = grid_v.dv
    f_grid = zeros(nx, nv)
    for j in 1:nv
        v_j = grid_v.v[j]
        for l in 1:nv, i in 1:nx
           f_grid[i, j] += grid_v.w[l] * rho[i, l] * Spline(v_j - u[i, l], dv) 
        end
    end

    return f_grid

end
#
#Spline of order 3 supp S_3 = [-2 h , 2 h]
function Spline(x::Float64, h::Float64)
    if (h < abs(x) < 2 * h)
        S = (1.0 / 6.0) * ((2 - abs(x / h)) * (2 - abs(x / h)) * (2 - abs(x / h)))
    elseif (abs(x) < h)
        S = (1.0 / 6.0) * (4.0 - 6.0 * (abs(x / h) * abs(x / h)) + 3.0 * (abs(x / h) * abs(x / h) * abs(x / h)))
    else
        S = 0
    end
    return S / h
end

export remap_f!

#This works only for uniform grid for the moment
function remap_f!(mesh_x::AbstractMesh, grid_v::UniformGrid, rho::Matrix{Float64}, u::Matrix{Float64})
    nv = grid_v.nv
    dv = grid_v.dv
    nx = mesh_x.nx
    new_SF = 0.0
    new_weights = zeros(nv)
    new_rho = zeros(nx + 1, nv)
    new_u = zeros(nx + 1, nv)

    #Reinitiaize rho and u on uniform grid
    for l in 1:nv, i in 1:(nx + 1)
        new_u[i, l] = grid_v.v[l]
        new_rho[i, l] = evaluate_f_on(i, l, grid_v, rho, u) / evaluate_mean_f_on(l, mesh_x, grid_v, rho, u)
    end
    #Compute the new weights
    for l in 1:nv
        new_weights[l] = evaluate_mean_f_on(l, mesh_x, grid_v, rho, u) * dv
        new_SF += new_weights[l]
    end
    #Update the weights
    for l in 1:nv
        new_weights[l] *= (1.0 / new_SF)
        grid_v.w[l] = new_weights[l]
    end
    rho .= new_rho
    return u .= new_u
end

function remap_f!(rho, u, mesh_x::AbstractMesh, grid_v::AbstractGrid)

    nv, dv = grid_v.nv, grid_v.dv
    nx, dx, xmin, xmax = mesh_x.nx, mesh_x.dx, mesh_x.xmin, mesh_x.xmax
    L = xmax - xmin

    new_SF = 0.0
    new_weights = zeros(nv)
    new_rho = zeros(nx, nv)
    new_u   = zeros(nx, nv)

    new_u .= grid_v.v'

    @threads for l in 1:nv
        v_j = grid_v.v[l]
        m_f = 0.0
        for i = 1:nx
            f = 0.0
            for j = 1:nv
                df = grid_v.w[j] * rho[i, j] * Spline(v_j-u[i, j], dv)
                m_f += df * dx / L
                f += df
            end
            new_rho[i,l] = f 
        end
        new_weights[l] = m_f * dv
        new_rho[:,l] ./= m_f
    end

    new_SF = sum(new_weights)

    grid_v.w .= new_weights ./ new_SF

    rho .= new_rho
    u .= new_u
end

#Evaluate f on (x_i,v_j)
function evaluate_f_on(i::Int, j::Int, grid_v::UniformGrid, rho::Matrix{Float64}, u::Matrix{Float64})
    nv = grid_v.nv
    dv = grid_v.dv
    v_j = grid_v.v[j]
    f = 0.0
    for l in 1:nv
        f += grid_v.w[l] * rho[i, l] * Spline(v_j - u[i, l], dv)
    end
    return f
end

#Compute mean_f at v_j on the torus of size L = xmax - xmin
function evaluate_mean_f_on(j::Int, mesh_x::AbstractMesh, grid_v::UniformGrid, rho::Matrix{Float64}, u::Matrix{Float64})
    mean_f = 0.0
    nv = grid_v.nv
    dv = grid_v.dv
    nx = mesh_x.nx
    dx = mesh_x.dx
    v_j = grid_v.v[j]
    for  l in 1:nv
        for i in 1:(nx + 1) #Rectangle formula
            mean_f += (grid_v.w[l] * rho[i, l] * Spline(v_j - u[i, l], dv) * dx) / (mesh_x.L)
        end
    end
    return mean_f
end

