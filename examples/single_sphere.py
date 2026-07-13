"""Single sphere: compute extinction efficiency and scattering matrix."""

import numpy as np
from pymstm import MSTM

# Size parameter x = 2*pi*r/lambda = 5.0, refractive index n = 1.5
m = MSTM()
m.set_spheres(
    radii=[5.0],
    positions=[[0, 0, 0]],
    orders=[12],
    ref_re=[1.5],
    ref_im=[0.0],
)
m.set_medium_ref(1.0, 0.0)
m.set_incident(alpha_deg=0, beta_deg=0)
m.prepare()

result = m.solve()
print(f"Q_ext  = {result['qext_tot']:.6f}")
print(f"Q_abs  = {result['qabs_tot']:.6f}")
print(f"Q_sca  = {result['qsca_tot']:.6f}")
print(f"Status = {result['status']} (iterations: {result['iterations']})")

# Scattering matrix
sm = m.get_scattering_angle(costheta=1.0, phi=0.0)
print(f"\nMueller matrix S11 at forward: {sm[0]:.6f}")

m.finalize()
