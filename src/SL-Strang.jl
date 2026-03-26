export compute_feet_char!
export compute_x_feet_mesh_Strang!
export advect_x_sol_Strang!
export advect_v_sol_Strang!
"""
$(SIGNATURES)
Compute the characteristic feet when given the velocity field v:
Solve by fixed point the equation of unknown x,  y = x  + 0.5 * dt *v(x)
x_k+1 = (y-0.5*dt*v(x_k))
"""
function compute_feet_char!(y::Float64,dt::Float64,mesh::AbstractMesh,v::AbstractVector)
    it = 0
    err = 1.0
    x_feet = y
    L = mesh.L
    while( err > 1E-6 && it < 100 )
        x_old = x_feet
        x_feet = mod(y - 1.0 * dt * interpolate_cubic_on_mesh(x_feet,mesh,v),L) ## Probleme avec l'interpolation sur domaine periodique
        err = abs(x_old-x_feet)
        if( it > 50)
            println("err = $err,  it = $it, x_feet = $x_feet, FIXED-POINT DOES NOT CONVERGE")
        end
        it+=1
    end
    x_feet = mod(x_feet,L)
    return x_feet
end

"""
$(SIGNATURES)
Collect of all the feets for every phase
"""
function compute_x_feet_mesh_Strang!(dt::Float64,mesh::AbstractMesh,x_feet_mesh::AbstractVector,u::AbstractVector)
    nx = mesh.nx
    for i in 1:(nx+1)
        x_feet_mesh[i] = compute_feet_char!(mesh.x[i],dt,mesh,u)
    end
    return
end

function advect_x_sol_Strang!(
        mesh::AbstractMesh, new_rho::AbstractVector, rho::AbstractVector, new_u::AbstractVector, u::AbstractVector,
	dt::Float64, x_feet_mesh::AbstractVector)
    nx = mesh.nx
    dx_u = compute_dx!(u,mesh)
    for i in 1:(nx+1)
        jac =    (1+ 1.0 * dt * interpolate_cubic_on_mesh(x_feet_mesh[i],mesh,dx_u) )
        #println("jac = $jac")
        new_rho[i] = interpolate_cubic_on_mesh(x_feet_mesh[i],mesh,rho)/jac
        new_u[i] = interpolate_cubic_on_mesh(x_feet_mesh[i],mesh,u)
    end
    return
end

function advect_v_sol_Strang!(
        mesh::AbstractMesh, new_u::AbstractVector, u::AbstractVector, E::AbstractVector,
	dt::Float64)
    nx = mesh.nx
    for i in 1:(nx+1)
        new_u[i] =  u[i] + 0.5 * dt * E[i]
    end
    return
end
