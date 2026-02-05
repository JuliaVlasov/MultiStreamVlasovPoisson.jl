using DispersionRelations
using LinearAlgebra
using MultiStreamVlasovPoisson
using Plots
using .Threads


function main(hermite_quad)

    eps = 1.0
    nx = 200
    k = 0.5
    xmin, xmax = 0.0, 2π / k
    vmin, vmax = -6.0, 6.0
    ng = 50

    if hermite_quad
        mesh = GaussHermiteMesh(nx, ng)
    else
        mesh = UniformMesh(xmin, xmax, nx, vmin, vmax, ng)
    end

    rho, u, rho_tot, vp = compute_initial_condition(mesh, k)

    rho_tot_init=copy(rho_tot)

    poisson = NonLinearPoissonSolver(eps, nx)

    phi = -log.(rho_tot)
    #display(plot(mesh.x, rho_tot))
    poisson!(phi, mesh, rho_tot)

    u_at_step_n   = zeros(nx + 1,ng)    
    rho_at_step_n = zeros(nx + 1,ng)    

    dt = mesh.dx
    tfinal =  500*dt  # Final time
    time = [0.0]

    @show elec_energy = [compute_elec_energy(phi, mesh, eps)]

    n = 0
    while n * dt <= tfinal

        # Update phi
        #solve!(phi, poisson, rho_tot)
        #poisson!(phi, mesh, rho_tot)

	iter = 0
        err=1e-10
	maxiter=50

        copyto!(rho_at_step_n, rho)
        copyto!(u_at_step_n, u)	
        old_u = zeros(nx + 1,ng)
    
       #loop for fixed point in u
       ##########################
       while norm(u .- old_u, Inf) / norm(u, Inf) > err && iter < maxiter


	  #@show norm(u .- old_u, Inf) / norm(u, Inf)

          @threads for j in 1:ng	 
              update_rho!(mesh, view(rho, :, j), view(u, :, j), view(rho_at_step_n, :, j), dt)
          end

	  compute_rho_total!(rho_tot, mesh, rho)
	  #solve!(phi, poisson, rho_tot)
	  poisson!(phi, mesh, rho_tot)	  

          copyto!(old_u, u)

	  @threads for j in 1:ng
	  	   update_u!(mesh, view(rho, :, j), view(u, :, j), phi,
		             view(rho_at_step_n, :, j), view(u_at_step_n, :, j), dt, maxiter)
          end

#        @threads for j in 1:ng
#            update!(mesh, poisson, view(rho, :, j), view(u, :, j), phi, dt)
#        end

         iter += 1

	 #@show iter
	end


        compute_rho_total!(rho_tot, mesh, rho)
        push!(elec_energy, compute_elec_energy(phi, mesh, eps))
        n += 1
        push!(time, n * dt)
        println("iteration: $n , time = $(n * dt), elec energy = $(last(elec_energy))")

    end

    return time, elec_energy, rho_tot_init, vp

end



@time t, elec_energy, rho_tot_init, vp = main(false)
plot(t, elec_energy, yaxis = :log, label = "uniform")

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