"""Single sphere: efficiencies and the forward scattering matrix element,
with the high-level MSTM model and with the low-level bindings."""

from pymstm import MSTM, MstmBindings

# Size parameter x = 2*pi*r/lambda = 5.0, refractive index 1.5
result = MSTM(radii=[5.0], positions=[(0.0, 0.0, 0.0)], ref_re=[1.5], ref_im=[0.0]).solve()
print(f"Q_ext = {result.q_ext_total:.6f}, Q_sca = {result.q_sca_total:.6f}")

m = MstmBindings()
m.set_spheres(radii=[5.0], positions=[[0, 0, 0]], orders=[12], ref_re=[1.5], ref_im=[0.0])
m.prepare()
r = m.solve()
print(f"Q_ext = {r['qext_tot']:.6f}, status {r['status']}, {r['iterations']} iterations")
sm = m.get_scattering_angle(costheta=1.0, phi=0.0)
print(f"S11 (forward) = {sm[0]:.6f}")
m.finalize()
