using Aqua
using MultiStreamVlasovPoisson
using Test

@testset "GaussHermite mesh" begin

    nx = 100
    ng = 3
    eps = 1.0
    mesh = GaussHermiteMesh(nx, ng)

    f(x) = x^4

    I = sum(mesh.w .* f.(mesh.x))

    @test I ≈ 3(√π) / 4

end


@testset "Distribution function" begin

    eps = 1.0
    nx = 100
    ng = 100

    mesh = GaussHermiteMesh(nx, ng)

    rho, u, rho_tot = compute_initial_condition(mesh)

    @test true

end

@testset "Non linear Poisson solver" begin

    eps = 1.0
    nx = 100

    solver = NonLinearPoissonSolver(eps, nx)

    @test true

end


@testset "Aqua.jl" begin
  Aqua.test_all(MultiStreamVlasovPoisson)
end
