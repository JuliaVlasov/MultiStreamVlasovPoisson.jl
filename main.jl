using DispersionRelations
using LinearAlgebra
using MultiStreamVlasovPoisson
using Plots
using .Threads

global k = 0.2               #Wave number
global test_case::String      #test_case =  {landau_damping,two_streams,mono_kinetic}
global T = 1.0                #Temperature of the Maxwellian
global L  = 2π / k            #Size of the domain
global eps = 1.0              #Debye length
global solver::String         #SOLVER = {FV,SL,SL-Strang} First Order Implicit AP Finite Volume Schem or Cubic Implicit Semi-Lagragian scheme
function main(hermite_quad)
    test_case = "two_streams"
    solver    = "SL-Strang"
    nx, xmin, xmax = 256, 0.0, L
    mesh_x = UniformMesh(xmin,xmax,nx)
    nv, vmin, vmax = 256, -6.0, 6.0
    grid_v = UniformGrid(vmin, vmax, nv, T,mesh_x,test_case)
    rho, u, rho_tot = compute_initial_condition(mesh_x,grid_v,k,T,test_case)
    phi = zeros(nx+1)
    poisson!(phi, mesh_x, rho_tot, eps)
    E = -1.0*compute_dx!(phi,mesh_x)
    if(solver == "SL" || solver == "SL-Strang")
        x_feet_mesh = zeros(nx+1,nv)
        rho_pred    = zeros(nx+1,nv)
        u_pred      = zeros(nx+1,nv)
        phi_pred    = zeros(nx+1)
    end

    #Initialize the streams
    u_at_step_n   = zeros(nx + 1,nv)    
    rho_at_step_n = zeros(nx + 1,nv)    

    #Set the CFL number and the final time
    dt =  0.1 #1.0*mesh_x.dx
    tfinal = 50
    time = [0.0]
    remap_time = 0.0

    #Array of physical quantities 
    elec_energy     = [compute_elec_energy(phi, mesh_x, eps)]
    mass            = [compute_total_mass(rho_tot,mesh_x)]
    momentum        = [compute_momentum(rho,u,mesh_x,grid_v)]
    total_energy    = [compute_elec_energy(phi, mesh_x, eps) + compute_kinetic_energy(rho,u,mesh_x,grid_v)]

    #Temporal loop
    n = 0
    #anim = Animation()
    sum_norm_dx_u = 0.0 
    remap_time = 0.0
  while n * dt <= tfinal
	    iter = 0
        err=1e-10
	    maxiter=50
        sum_norm_dx_u += dt*compute_norm_dx_u(mesh_x,grid_v,u)
        threshold = 0.2 #Numerical remapping threshold : you may change it or not depending on the test case
        if(sum_norm_dx_u > threshold )
            println("Remapping f at time = $(n*dt), solver = $solver")
            remap_time = n * dt
            rho, u = remap_f_on_uniform_grid(mesh_x,grid_v,rho,u)
            sum_norm_dx_u = 0.0
        end
        
        if(solver=="FV")
            copyto!(rho_at_step_n, rho)
            copyto!(u_at_step_n, u)	
            old_u = zeros(nx + 1,nv)
            #Fixed point loop to solve the non linear MultiStream pressureless Euler-Poisson system
            while norm(u .- old_u, Inf) / norm(u, Inf) > err && iter < maxiter
                #Update rho : the streams are packed into groups that are solved on each threads
                @threads for j in 1:nv	 
                    update_rho!(mesh_x, view(rho, :, j), view(u, :, j), view(rho_at_step_n, :, j), dt)
                end
                #Assemble rho
	            compute_rho_total!(rho_tot,grid_v,rho)
                #Solve Poisson
	            poisson!(phi, mesh_x, rho_tot,eps)	  
                copyto!(old_u, u)
                #Update u : the streams are packed into groups that are solved on each threads
	            @threads for j in 1:nv
	  	            update_u!(mesh_x, view(rho, :, j), view(u, :, j), phi,
		            view(rho_at_step_n, :, j), view(u_at_step_n, :, j), dt, maxiter)
                end
                iter += 1
	        end
        elseif(solver=="SL")

            copyto!(rho_at_step_n, rho)
            copyto!(u_at_step_n, u)	
            @threads for j in 1:nv
                compute_x_feet_mesh!(dt,mesh_x,view(x_feet_mesh,:,j), view(u_at_step_n, :, j ),E)
                update_rho_predictor_SL!(mesh_x,view(rho_pred,:,j), view(rho_at_step_n, :, j),view(u_at_step_n, :, j),dt,
                view(x_feet_mesh, :, j) )
            end

            #Assemble rho
	        compute_rho_total!(rho_tot,grid_v,rho_pred)
            #Solve Poisson 
	        poisson!(phi_pred, mesh_x, rho_tot,eps)
            E_pred = -1.0.*compute_dx!(phi_pred,mesh_x)
            @threads for j in 1:nv
                update_u_SL!(mesh_x, E, E_pred, view(u,:,j), view(u_at_step_n,:,j), dt,view(x_feet_mesh,:,j))
                update_rho_corrector_SL!(mesh_x, view(rho,:,j), view(rho_at_step_n, : ,j), view(u_at_step_n,:,j),view(u,:,j), dt, view(x_feet_mesh,:,j))
            end
            #Assemble rho
	        compute_rho_total!(rho_tot,grid_v,rho)
            #Solve Poisson 
	        poisson!(phi, mesh_x, rho_tot,eps)
            E = -1.0*compute_dx!(phi,mesh_x)   
        elseif(solver =="SL-Strang")
            copyto!(rho_at_step_n, rho) #(rho^n = rho  u^n = u)
            copyto!(u_at_step_n, u)	


            #Advection in v
            @threads for j in 1:nv
                advect_v_sol_Strang!(mesh_x,view(u_pred,:,j),view(u_at_step_n, :, j),E,dt)
            end	

            #Advection in x
            copyto!(u_at_step_n,u_pred)
            @threads for j in 1:nv
                compute_x_feet_mesh_Strang!(dt,mesh_x,view(x_feet_mesh,:,j), view(u_pred, :, j ))
                advect_x_sol_Strang!(mesh_x, view(rho,:,j), view(rho_at_step_n, :, j),view(u, :, j), view(u_at_step_n, :, j),dt,
                view(x_feet_mesh, :, j))
            end

            copyto!(u_at_step_n,u) #u^n = u
            #Advection in v
            @threads for j in 1:nv
                advect_v_sol_Strang!(mesh_x,view(u,:,j),view(u_at_step_n, :, j),E,dt)
            end	
            #Assemble rho
	        compute_rho_total!(rho_tot,grid_v,rho)
            #Solve Poisson 
	        poisson!(phi, mesh_x, rho_tot,eps)
            E = -1.0*compute_dx!(phi,mesh_x)   
        end
        #compute_rho_total!(rho_tot, grid_v, rho)
    
        push!(elec_energy, compute_elec_energy(phi, mesh_x, eps))
        #push!(mass,compute_total_mass(rho_tot,mesh_x))
        #push!(momentum,compute_momentum(rho,u,mesh_x,grid_v))
        #push!(total_energy,compute_elec_energy(phi, mesh_x, eps) + compute_kinetic_energy(rho,u,mesh_x,grid_v))
        n += 1
        push!(time, n * dt)
        #println("iteration = $n, time = $(n * dt) , tr = $remap_time:")
        #println("||E|| = $(last(elec_energy)), Mass = $(last(mass)), int_[tr,time] ||dxU(t)||dt = $sum_norm_dx_u")
        #Make the animation : evolution of the surface plot of the distribution function
        #Plot every "per" 
        #Comment this part if you do not want the animation :  it is the time consuming part of the code.
        #per = 100000
        #if(mod(n,per) == 1)
        #    f_on_grid =  interpolate_f_on_grid(mesh_x,grid_v,rho,u)
        #    X = []
        #    Y = []
        #    Z = []
        #    for i in 1:nx+1
        #        for j in 1:nv
        #            X = push!(X,mesh_x.x[i])
        #            Y = push!(Y,grid_v.v[j])
        #            Z = push!(Z,f_on_grid[i,j])
        #        end
        #    end
        #    p = plot(X,Y,Z,st = [:surface],camera = (0,90),xlabel = "x", ylabel="v",)
        #    plot!(p; ylims=(-6.,6.))
        #    frame(anim)
        #end

    end
    #gif(anim, "fig/$(test_case)_$(solver).gif", fps = 15)
    #Plot the final distribution function
    f_on_grid =  interpolate_f_on_grid(mesh_x,grid_v,rho,u)
    X = []
    Y = []
    Z = []
    for i in 1:nx+1
        for j in 1:nv
            X = push!(X,mesh_x.x[i])
            Y = push!(Y,grid_v.v[j])
            Z = push!(Z,f_on_grid[i,j])
        end
    end
    plot_f = plot(X,Y,Z,st = [:surface],camera = (0,90),xlabel = "x", ylabel="v")
    return time, elec_energy, mass, momentum, total_energy, grid_v, mesh_x, u, rho_tot, plot_f, phi, E

end
@time time, elec_energy, mass, momentum, total_energy, grid_v, mesh_x, u, rho_tot, plot_f,phi, E = main(true)
plot(time,log.(elec_energy))


#These are the reference solution electric energy evolution for the linear landau damping
# Copy and paste in the Julia Terminal the solution you are interested in.
#When k=0.5, we have from Eric's book:
#E_k(t)=4.*a.*0.3677.*exp.(-0.1533.*t).*sqrt.(2pi).*abs.(cos(1.4156.*t-0.536245))
#plot(time,log.(elec_energy))
#plot!(time,log.(0.01.*exp.(-0.1533.*time).*sqrt(2*pi).*abs.(cos.(1.4156.*time.-0.536245))))           for a=0.01
#plot!(time,log.(0.0025.*0.3677.*exp.(-0.1533.*time).*sqrt(2*pi).*abs.(cos.(1.4156.*time.-0.536245))))  for a=0.001

#This is a tool to compute the slope of the electric energy evolution
# line, ω, = fit_complex_frequency(t, elec_energy, use_peaks = 1:2)
# plot!(time, line; yaxis = :log)
# title!("ω = $(imag(ω))")