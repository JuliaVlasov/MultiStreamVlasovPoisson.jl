# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Julia 1.12
#     language: julia
#     name: julia-1.12
# ---

using MultiPhaseVlasov
using Plots

# +
k = 0.3
L = 2π / k
vth = 1.0
nx, xmin, xmax = 96, 0.0, L
mesh_x = UniformMesh(xmin, xmax, nx)
nv, vmin, vmax = 256, -9.0, 9.0
test_case = TwoStreams(k = k, vth = vth)
grid_v = UniformGrid(vmin, vmax, nv, mesh_x, test_case)
rho, u, rho_tot = compute_initial_condition(test_case, mesh_x, grid_v)

plot(mesh_x.x, rho_tot)
# -

test_case = LandauDamping()
grid_v = UniformGrid(vmin, vmax, nv, mesh_x, test_case)
rho, u, rho_tot = compute_initial_condition(test_case, mesh_x, grid_v)
plot(mesh_x.x, rho_tot)

test_case = BumpOnTail()
grid_v = UniformGrid(vmin, vmax, nv, mesh_x, test_case)
rho, u, rho_tot = compute_initial_condition(test_case, mesh_x, grid_v)
plot(mesh_x.x, rho_tot)

test_case = MonoKinetic()
grid_v = UniformGrid(vmin, vmax, nv, mesh_x, test_case)
rho, u, rho_tot = compute_initial_condition(test_case, mesh_x, grid_v)
plot(mesh_x.x, rho_tot)
