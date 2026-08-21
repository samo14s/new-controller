"""demo.py — tout ce que le modele sait faire, SANS aucun correcteur.

    python demo.py

Cinq demonstrations, dans l'ordre ou on les utilise en pratique :
  1. proprietes modales et validation
  2. reponse en frequence aux deux entrees (force de coupe, tension pastille)
  3. coefficients de coupe alpha3(t), alpha4(t) sur une periode de dent
  4. limites de stabilite : une position, puis le long de la passe, puis lobes
  5. simulation temporelle d'une passe complete
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from chebyshev_plate import ChebyshevPlate
from milling_dynamics import (alpha4_series, alpha4_average, N_TEETH, AE_NOM,
                              FZ_NOM)
from stability_fdm import is_stable, stability_limit
from time_domain import TimeSim, amplitude_spectrum, pass_duration
from schedule_im import chatter_frequencies

F_MEASURED = [540, 1068, 2787, 3351, 4122]     # Table 4 de la reference
RPM = 4900

print("=" * 72)
print(" 1. MODELE ET VALIDATION")
print("=" * 72)
p = ChebyshevPlate(PX=14, PZ=14)
p.add_piezo_patch()
print("  plaque nue        :", np.round(p.freq_n, 1), "Hz")
p.calibrate_frequencies(F_MEASURED)
D_obs = p.D_row(p.lp, p.hp)
H = np.asarray(p.H_Pe_modal, float)
print("  apres calibration :", np.round(p.freq_n, 1), "Hz")
print("  amortissements    :", np.round(p.zeta_modes * 100, 2), "%")
print("  D_obs (capteur au coin sup. droit) :", np.round(D_obs, 3))
print("  H_Pe (couplage pastille, N/V)      :", np.round(H, 5))
print("  produits modaux D_obs*H            :", np.round(D_obs * H, 4))
print("  -> signes alternes : c'est la propriete structurante du systeme")

print("\n" + "=" * 72)
print(" 2. REPONSES EN FREQUENCE (au coin, ou se trouve le capteur)")
print("=" * 72)
f = np.linspace(1, 5000, 20000)
om = 2 * np.pi * f
den = p.omega_n[:, None]**2 - om[None, :]**2 \
    + 2j * p.zeta_modes[:, None] * p.omega_n[:, None] * om[None, :]
Gf = np.sum((D_obs**2)[:, None] / den, axis=0)      # m/N
Gu = np.sum((D_obs * H)[:, None] / den, axis=0)     # m/V
db = lambda v: 20 * np.log10(np.abs(v))
print(f"  souplesse statique : {np.abs(Gf[0])*1e6:.2f} um/N   ({db(Gf[0]):.1f} dB re 1 m/N)")
print(f"  gain statique tension : {np.abs(Gu[0])*1e6:.4f} um/V ({db(Gu[0]):.1f} dB re 1 m/V)")
for k in range(5):
    pk = D_obs[k]**2 / (2 * p.zeta_modes[k] * p.omega_n[k]**2)
    print(f"   mode {k+1} ({p.freq_n[k]:6.0f} Hz) : pic {db(pk):7.1f} dB re 1 m/N")

print("\n" + "=" * 72)
print(" 3. COEFFICIENTS DE COUPE SUR UNE PERIODE DE DENT")
print("=" * 72)
ap = 0.30e-3
a3, a4 = alpha4_series(RPM, ap, p.hp, 400, midpoint=True)
tau = 60.0 / (N_TEETH * RPM)
print(f"  {RPM} tr/min, a_p = {ap*1e3:.2f} mm, a_e = {AE_NOM*1e3:.1f} mm, "
      f"f_z = {FZ_NOM*1e3:.3f} mm/dent")
print(f"  tau = {tau*1e3:.3f} ms    alpha4 moyen = {alpha4_average(RPM, ap, p.hp):.0f} N/m")
print(f"  alpha4 : min {a4.min():.0f}  max {a4.max():.0f} N/m   "
      f"rapport crete/moyenne = {abs(a4).max()/abs(alpha4_average(RPM,ap,p.hp)):.1f}")
print(f"  fraction de la periode ou une dent coupe : "
      f"{100*np.mean(np.abs(a4) > 0.01*np.abs(a4).max()):.1f} %")

print("\n" + "=" * 72)
print(" 4. STABILITE EN BOUCLE OUVERTE")
print("=" * 72)
ok, fc, rho = is_stable(p, RPM, ap, 0.0, coeff_mode='time', coeff_scale=1.0,
                        return_freq=True)
print(f"  a {RPM} tr/min, x = 0, a_p = {ap*1e3:.2f} mm : rho = {rho:.4f}"
      f"  -> {'stable' if ok else 'INSTABLE'}")
print("\n  limite le long de la passe :")
for x in (0, 25, 50, 75, 100):
    L = stability_limit(p, RPM, x * 1e-3, hi=3e-3, tol=3e-6,
                        coeff_mode='time', coeff_scale=1.0)
    fcs, _ = chatter_frequencies(p, RPM, x * 1e-3)
    print(f"    x = {x:3d} mm : {L*1e3:.4f} mm   frequences de broutement "
          f"predites {fcs[0]:.0f} / {fcs[1]:.0f} Hz")
print("\n  lobes (minimum sur 5 positions) :")
speeds = np.arange(3000, 7001, 250)
lob = [min(stability_limit(p, r, x * 1e-3, hi=3e-3, tol=2e-5,
                           coeff_mode='time', coeff_scale=1.0)
           for x in (0, 25, 50, 75, 100)) * 1e3 for r in speeds]
for r, v in zip(speeds, lob):
    print(f"    {r:5d} tr/min : {v:.3f} mm")
np.savetxt('results/lobes_boucle_ouverte.csv',
           np.column_stack([speeds, lob]), delimiter=',',
           header='rpm,limite_mm', comments='')

print("\n" + "=" * 72)
print(" 5. SIMULATION TEMPORELLE — PASSE COMPLETE, SANS COMMANDE")
print("=" * 72)
T, v = pass_duration(RPM)
print(f"  duree de passe {T:.3f} s a {v*1e3:.2f} mm/s "
      f"(l'outil parcourt les {p.lp*1e3:.0f} mm de l'arete)")
sim = TimeSim(p, RPM, 0.03e-3, sign=1.0, n_modes=5, n_sub=82)
r = sim.run(T=0.6, controller=None, moving=True, stop_um=5e3)
y = r['y_mill'] * 1e6
print(f"  a_p = 0.030 mm (sous la limite) : |y| moyen {np.mean(np.abs(y)):.3f} um, "
      f"crete {np.abs(y).max():.3f} um")
sim = TimeSim(p, RPM, 0.30e-3, sign=1.0, n_modes=5, n_sub=82)
r2 = sim.run(T=0.6, controller=None, moving=True, stop_um=5e3)
print(f"  a_p = 0.300 mm (au-dessus)      : "
      f"{'divergence a t = %.3f s' % r2['t_div'] if r2['diverged'] else 'stable'}")
fq, A = amplitude_spectrum(r2['t'], r2['y_mill'])
i = int(np.argmax(np.where((fq > 300) & (fq < 1500), A, 0)))
print(f"  frequence dominante du broutement : {fq[i]:.0f} Hz")

fig, axs = plt.subplots(2, 2, figsize=(11.5, 7))
axs[0, 0].semilogy(f, np.abs(Gf), lw=1.2)
axs[0, 0].set_xlabel('Hz'); axs[0, 0].set_ylabel('|G| [m/N]')
axs[0, 0].set_title('(a) reponse a la force de coupe'); axs[0, 0].grid(alpha=.3)
axs[0, 1].plot(np.arange(400) / 400, a4 * 1e-3, lw=1.4)
axs[0, 1].set_xlabel('t / tau'); axs[0, 1].set_ylabel(r'$\alpha_4$ [kN/m]')
axs[0, 1].set_title('(b) coefficient regeneratif sur une periode de dent')
axs[0, 1].grid(alpha=.3)
axs[1, 0].plot(speeds, lob, lw=1.6)
axs[1, 0].set_xlabel('tr/min'); axs[1, 0].set_ylabel('limite [mm]')
axs[1, 0].set_title('(c) lobes en boucle ouverte'); axs[1, 0].grid(alpha=.3)
axs[1, 1].plot(r2['t'], r2['y_mill'] * 1e6, lw=.4, color='#c0392b')
axs[1, 1].set_xlabel('t [s]'); axs[1, 1].set_ylabel('y [um]')
axs[1, 1].set_title('(d) broutement a $a_p$ = 0.30 mm'); axs[1, 1].grid(alpha=.3)
fig.suptitle('Modele de fraisage de plaque mince — boucle ouverte', fontsize=12)
fig.tight_layout()
fig.savefig('results/demo.png', dpi=140)
print("\n  figures : results/demo.png    donnees : results/lobes_boucle_ouverte.csv")
