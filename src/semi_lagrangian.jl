using FastInterpolations

export SemiLagrangian

abstract type AbstractSolver end

struct SemiLagrangian <: AbstractSolver 

    nx :: Int
    nv :: Int
    dx :: Float64
    xq :: Vector{Vector{Float64}}
    enew :: Vector{Float64}
    dudx :: Vector{Float64}

    function SemiLagrangian( mesh, grid )

        xq = [zeros(mesh.nx) for i in 1:grid.nv]
        enew = zeros(mesh.nx)
        dudx = zeros(mesh.nx)

        new( mesh.nx, grid.nv, mesh.dx, xq, enew, dudx )

    end

end

export compute_dx

function compute_dx(v::AbstractVector, mesh::AbstractMesh)
    kx = mesh.kx
    dx_v = real(ifft(1im * kx .* fft(v)))
    return dx_v
end

export compute_dx!

function compute_dx!(dv, mesh::AbstractMesh, u, v_hat)
    v_hat .= fft(u, 1)
    v_hat .*= 1im .* mesh.kx
    dv .= real(ifft(v_hat, 1))
end



export update_rho_predictor!

function update_rho_predictor!(
        rho_pred::Matrix{Float64}, 
        solver::SemiLagrangian,
        mesh::AbstractMesh, 
        rho::Matrix{Float64}, 
        v::Vector{Float64},
        dv::Matrix{Float64},
        dt::Float64
    )

    xi = 1.0:mesh.nx
    bc = PeriodicBC(endpoint = :exclusive)

    for j in eachindex(solver.xq)
        v .= cubic_interp(xi, view(dv, :, j), solver.xq[j], bc = bc)
        rho_pred[:, j] .= cubic_interp(xi, view(rho,:,j), solver.xq[j], bc = bc)
        rho_pred[:, j] .*= exp.(-dt .* v)
    end

end

export update_u!

function update_u!(
        u::Matrix{Float64}, 
        solver::SemiLagrangian,
        mesh::AbstractMesh, 
        e::AbstractVector, 
        e_pred::AbstractVector,
        dt::Float64
    )

    xi = 1.0:mesh.nx
    bc = PeriodicBC(endpoint = :exclusive)

    for j in eachindex(solver.xq)

        u[:, j] .= cubic_interp(xi, view(u, :, j), solver.xq[j], bc = bc)
        solver.enew .= cubic_interp(xi, e, solver.xq[j], bc = bc)
        solver.enew .+= e_pred
        u[:, j] .+= 0.5dt .* solver.enew

    end

end

export update_rho_corrector!

function update_rho_corrector!(
        rho::Matrix{Float64}, 
        solver::SemiLagrangian,
        mesh::AbstractMesh, 
        dv::Matrix{Float64},
        dv_plus::Matrix{Float64},
        dt::Float64, 
    )

    xi = 1.0:mesh.nx
    bc = PeriodicBC(endpoint = :exclusive)

    for j in eachindex(solver.xq)

        rho[:,j] .= cubic_interp(xi, view(rho, :, j), solver.xq[j], bc = bc)
        solver.dudx .= cubic_interp(xi, view(dv, :, j), solver.xq[j], bc = bc)
        solver.dudx .+= view(dv_plus, :, j)
        rho[:,j] .*= exp.(-0.5dt .* solver.dudx)

    end

end

export compute_xfeet!

function compute_xfeet!(xq, xi, u, mesh, dt, bc)

    nx, dx = mesh.nx, mesh.dx
    @threads for j in eachindex(xq)
        itp = cubic_interp(xi, view(u, :, j), bc = bc)
        it = 0
        err = 1.0
        xnew = copy(xi)
        while err > 1e-5
            xold = copy(xnew)
            xnew = mod1.(xi .- dt .* itp(xnew) ./ dx, nx)
            err = maximum(abs.(xold .- xnew))
            if it > 50
                @info "err = $err,  it = $it, x_feet = $x_feet, FIXED-POINT DOES NOT CONVERGE"
                break
            end
            it += 1
        end
        xq[j] .= xnew
    end
end

export compute_x_feet_mesh!

"""
$(SIGNATURES)

Use Fixed-Point methodod to compute the feet of the characteristic : X(t-dt) = X(t) + d we search for d
"""
function compute_x_feet_mesh!(solver::SemiLagrangian, u::Matrix{Float64}, e::Vector{Float64}, dt::Float64)

    nx, dx, nv = solver.nx, solver.dx, solver.nv
    v = u .- 0.5dt .* e
    xi = 1.0:nx
    @threads for j in 1:nv
        itp = cubic_interp(xi, view(v, :, j), bc = PeriodicBC( endpoint = :exclusive))
        it = 0
        err = 1.0
        xnew = copy(xi)
        while err > 1e-6
            xold = copy(xnew)
            xnew = mod1.(xi .- dt .* itp(xnew) ./ dx, nx)
            err = maximum(abs.(xold .- xnew))
            if it > 50
                @info "err = $err,  it = $it, x_feet = $x_feet, FIXED-POINT DOES NOT CONVERGE"
                break
            end
            it += 1
        end
        solver.xq[j] .= xnew
    end


end

