export TwoStreams

"""
$(TYPEDEF)

## The Two-Stream Distribution Function

The standard choice is a **double-shifted Maxwellian**:

```math
f_0(x, v) = \\frac{n_0}{2\\sqrt{2\\pi}v_{th}} \\left[ \\exp\\!\\left(-\frac{(v - v_0)^2}{2v_{th}^2}\right) 
+ \\exp\\!\\left(-\\frac{(v + v_0)^2}{2v_{th}^2}\\right) \\right] \\cdot \\left(1 + \\epsilon \\cos(kx)\\right)
```

where:
- ``\\pm v_0`` are the **drift velocities** of the two beams (equal and opposite)
- ``v_{th}`` is the **thermal velocity** of each beam
- ``\\epsilon \\ll 1`` is the **perturbation amplitude** (typically ``\\epsilon \\sim 0.01–0.05``)
- ``k = 2\\pi/L`` is the **wavenumber** of the perturbation
- ``L`` is the domain length
"""
@with_kw struct TwoStreams <: InitialCondition

    n0::Float64 = 1.0
    α::Float64 = 0.001
    k::Float64 = 0.5
    v0::Float64 = 4.5
    vth::Float64 = 1.0

end

function f0(test_case::TwoStreams, x::Float64, v::Float64)::Float64

    n0 = test_case.n0
    α = test_case.α # Perturbation amplitude
    k = test_case.k # Perturbation wave number
    vth = test_case.vth # thermal velocity
    v0 = test_case.v0 # drift velocity

    f0 = (n0 / sqrt(2π * vth)) * (
        0.5 * exp(-0.5 * (v - v0) * (v - v0) / vth)
            + 0.5 * exp(-0.5 * (v + v0) * (v + v0) / vth)
    ) * (1.0 + α * cos(k * x))

    return f0

end
