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
@with_kw struct TwoStreams 

    n0::Float64 = 1.0
    α::Float64 = 0.01
    k::Float64 = 0.2
    v0::Float64 = 2.4
    vth::Float64 = 1.0

end

