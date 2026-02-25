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
    nv = 100
    xmin, xmax = 0, 2π
    vmin, vmax = -8.0, 8.0
    k = 0.5

    mesh = UniformMesh(xmin, xmax, nx, vmin, vmax, nv)

    rho, u, rho_tot = compute_initial_condition(mesh, k)

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
