using DispersionRelations
using LinearAlgebra
using MultiStreamVlasovPoisson
using Plots
using .Threads

function main(hermite_quad)
    eps = 1.0
    nx = 1000
    k = 0.5
    T = 1.0 
    xmin, xmax = 0.0, 2π / k
    #Construct the mesh in space
    mesh_x = UniformMesh(xmin,xmax,nx)
    #Construct the grid in velocity
    nv = 500
    vmin, vmax = -nv^0.5, nv^0.5 #-7.0, 7.0
    #if hermite_quad
    #grid_v = GaussHermiteGrid(nv,T)
    #else
    grid_v = UniformGrid(vmin, vmax, nv, T)
    #nd
    #grid_v = MonteCarloGrid(nv)

    rho, u, rho_tot = compute_initial_condition(mesh_x,grid_v,k,T)
    rho_tot_init=copy(rho_tot)
    #For every x_i f[x_i,.] is vector of size the cardinality of u[x_i,.] 
    # f is not a rectangular Matrix the number of column varies for each x_i.
    #f = compute_f(mesh_x,grid_v,rho,u) work in progress
   # l = @layout [a{0.7w} b; c{0.2h}]
   # pf = plot(mesh_x.x,u, f,st = [:surface, :contourf])
    #display(plot(mesh_x.x, rho_tot))
    #Solve the Poisson equation initially
    phi = -log.(rho_tot)
    poisson!(phi, mesh_x, rho_tot)
    #poisson = NonLinearPoissonSolver(eps, nx)
    #solve!(phi, poisson, rho_tot)

    
    
    u_at_step_n   = zeros(nx + 1,nv)    
    rho_at_step_n = zeros(nx + 1,nv)    

    dt = 2*mesh_x.dx
    tfinal =  100*dt  # Final time
    time = [0.0]

    @show elec_energy = [compute_elec_energy(phi, mesh_x, eps)]

    n = 0
    while n * dt <= tfinal
	iter = 0
    err=1e-10
	maxiter=50

        copyto!(rho_at_step_n, rho)
        copyto!(u_at_step_n, u)	
        old_u = zeros(nx + 1,nv)
    
       #Fixed point iterations to solve the non linear MultiStream pressureless Euler-Poisson system
       ##########################
       while norm(u .- old_u, Inf) / norm(u, Inf) > err && iter < maxiter


	  #@show norm(u .- old_u, Inf) / norm(u, Inf)

          @threads for j in 1:nv	 
              update_rho!(mesh_x, view(rho, :, j), view(u, :, j), view(rho_at_step_n, :, j), dt)
          end

	  compute_rho_total!(rho_tot,grid_v,rho)
	  #solve!(phi, poisson, rho_tot)
	  poisson!(phi, mesh_x, rho_tot)	  

          copyto!(old_u, u)

	  @threads for j in 1:nv
	  	   update_u!(mesh_x, view(rho, :, j), view(u, :, j), phi,
		             view(rho_at_step_n, :, j), view(u_at_step_n, :, j), dt, maxiter)
          end

         iter += 1
	end


        compute_rho_total!(rho_tot, grid_v, rho)
        push!(elec_energy, compute_elec_energy(phi, mesh_x, eps))
        n += 1
        push!(time, n * dt)
        println("iteration: $n , time = $(n * dt), elec energy = $(last(elec_energy))")

    end

    return time, elec_energy

end
@time time, elec_energy = main(true)
plot(time, elec_energy, yaxis = :log, label = "UniformGrid")

# @time t, elec_energy = main(true)
# plot!(t, elec_energy, yaxis = :log, label = "hermite")

# line, ω, = fit_complex_frequency(t, elec_energy, use_peaks = 1:2)
# plot!(time, line; yaxis = :log)
# title!("ω = $(imag(ω))")


#for k=0.4, we have from Eric's book (a=0.001)
#E_k(t)=0.002.*0.424666.*exp.(-0.0661.*t).*abs.(cos.(1.285.*t.−0.3357725)))
#plot(t,log.(elec_energy))
#plot!(t,log.(0.002.*0.42466.*exp.(-0.0661.*t).*abs.(cos.(1.285.*t.-0.33577))))
#plot!(t,log.(0.0075.*0.42466.*exp.(-0.0661.*t).*abs.(cos.(1.285.*t.-0.33577))))   for a=0.001

#for k=0.5, we have from Eric's book
#E_k(t)=4.*a.*0.3677.*exp.(-0.1533.*t).*sqrt.(2pi).*abs.(cos(1.4156.*t-0.536245))
#plot(t,log.(elec_energy))
#plot!(t,log.(0.01.*exp.(-0.1533.*t).*sqrt(2*pi).*abs.(cos.(1.4156.*t.-0.536245))))           for a=0.01
#plot!(t,log.(0.0025.*0.3677.*exp.(-0.1533.*t).*sqrt(2*pi).*abs.(cos.(1.4156.*t.-0.536245))))  for a=0.001