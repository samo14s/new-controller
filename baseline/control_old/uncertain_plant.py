"""
uncertain_plant.py — Systeme incertain du fraisage (Section 3.1-3.3 du papier)
==============================================================================
Construit, dans l'ordre exact du papier :

  Eq. (16)  modele d'etat de la plaque avec entrees force et tension ;
  Eq. (17)  troncature modale a DEUX modes + incertitude additive
            Delta_Pa(s) W_Pa(s), W_Pa = [W_Paf W_Pau] (Eqs. 18-19) ;
  Eq. (21)  modele reduit ecrit en valeurs nominales + perturbations
            (M_Pr0 + D_PrM) q" + (C_Pr0 + D_PrC) q' + (K_Pr0 + D_PrK
            + a40 D_Pr0^T D_Pr0 + D_PrD) q = F_PIr + H_Pe u_P ;
  Eq. (22)  D_PrM = L_Pm d_Pm, D_PrC = L_Pc d_Pc, D_PrKD = L_Pkd d_Pkd ;
  Eqs. (23)-(25)  a40 = 1.6 abar4, L_Pa = 1.3 abar4, D_Pr0^T D_Pr0 = moyenne
            (max+min)/2 le long du bord, L_DD = amplitude, et
            D_PrD = L_Pa L_DD delta ;
  Eq. (26)  systeme incertain complet (perturbations parametriques +
            incertitude additive) ;
  Eq. (27)  systeme de synthese avec les ponderations W_Pf, W_Pu, W_Pn ;
  Fig. 10   plant generalise G_Pmu et structure de Delta_P
            = blkdiag(Delta_Pa, Delta_Pr, Delta_Pp).

CONVENTION DE SIGNE : Eq. (13) telle que publiee (sign = +1 dans le paquet
`plant`), alpha4 etant la valeur signee rendue par milling_dynamics.

STRUCTURE DE Delta_Pr : les dix parametres reels de l'Eq. (22)
    d_Pm1, d_Pm2, d_Pc1, d_Pc2, d_Pk1, d_Pk2, d_PD1, d_PD2, d_PD3, d_PD4
sont ecrits comme dix blocs scalaires 1x1 (le produit L_Pkd d_Pkd de l'Eq. (22)
est developpe : chaque parametre recoit son propre canal entree/sortie). C'est
la meme LFT, sous forme carree, seule forme utilisable par mussv/D-K.
"""
import numpy as np

from milling_dynamics import (alpha4_average, dtd_statistics, AE_NOM)
from weights import W_Paf_ss, W_Pau_ss, synthesis_weights


# ---------------------------------------------------------------------------
def nominal_and_perturbations(plate, rpm, ap, ae=AE_NOM, n_modes=2,
                              alpha_coupling='paper', mass_pert=0.10,
                              stiff_pert=0.10, damp_pert=0.20, n_pos=201):
    """Eqs. (21)-(25) : valeurs nominales et amplitudes de perturbation.

    alpha_coupling :
      'paper' -> Eq. (25) litterale : L_PD_i = |L_Pa| * L_DD_i ;
      'exact' -> enveloppe exacte du produit alpha4(t) * D^T D(x) sur
                 alpha4 dans [0.3, 2.9] abar4 et x dans [0, l_P], c.-a-d.
                 L_PD_i = max|alpha4 DD_i - alpha40 DD0_i|. L'Eq. (25) du
                 papier neglige les termes croises alpha40*L_DD et L_Pa*DD0 ;
                 cette option les retablit (cf. README, section "ecarts").
    """
    w = plate.omega_n[:n_modes]
    z = plate.zeta_modes[:n_modes]
    M0 = np.eye(n_modes)                      # modes normalises en masse
    C0 = np.diag(2 * z * w)
    K0 = np.diag(w**2)

    abar4 = alpha4_average(rpm, ap, plate.hp, ae)          # coefficient moyen
    alpha40 = 1.6 * abar4                                   # Eq. (23)
    L_Palpha = 1.3 * abar4                                  # Eq. (23)

    st = dtd_statistics(plate, n_pos)
    DD0 = st['DD0'][:n_modes, :n_modes]                      # Eq. (24)
    LDD = st['LDD'][:n_modes, :n_modes]

    if alpha_coupling == 'paper':                            # Eq. (25)
        L_PD = np.abs(L_Palpha) * LDD
    elif alpha_coupling == 'exact':
        DtD = st['DtD'][:, :n_modes, :n_modes]
        lo, hi = 0.3 * abar4, 2.9 * abar4
        prod = np.concatenate([lo * DtD, hi * DtD], axis=0)
        L_PD = np.max(np.abs(prod - alpha40 * DD0), axis=0)
    else:
        raise ValueError(alpha_coupling)

    return dict(
        n=n_modes, omega=w, zeta=z, M0=M0, C0=C0, K0=K0,
        abar4=abar4, alpha40=alpha40, L_Palpha=L_Palpha,
        DD0=DD0, LDD=LDD, L_PD=L_PD,
        L_Pm=mass_pert * np.diag(M0).copy(),                 # 10 %
        L_Pk=stiff_pert * np.diag(K0).copy(),                # 10 %
        L_Pc=damp_pert * np.diag(C0).copy(),                 # 20 %
        K_eff=K0 + alpha40 * DD0,                            # raideur nominale
    )


# ---------------------------------------------------------------------------
class MillingUncertainSystem:
    """Plant generalise G_Pmu de la Fig. 10 et sa structure d'incertitude."""

    #  ordre des dix parametres reels de Delta_Pr
    PARAM_NAMES = ['dPm1', 'dPm2', 'dPc1', 'dPc2',
                   'dPk1', 'dPk2', 'dPD1', 'dPD2', 'dPD3', 'dPD4']

    def __init__(self, plate, rpm, ap, ae=AE_NOM, n_modes=2,
                 alpha_coupling='paper', wpf=None, wpu=None, wpn=None,
                 x_obs=None, z_obs=None, sign=1.0):
        self.plate, self.rpm, self.ap, self.ae = plate, rpm, ap, ae
        self.n = n_modes
        self.sign = float(sign)
        self.par = nominal_and_perturbations(plate, rpm, ap, ae, n_modes,
                                             alpha_coupling)
        n = self.n
        # --- geometrie des points d'action ---------------------------------
        self.x_obs = plate.lp if x_obs is None else x_obs
        self.z_obs = plate.hp if z_obs is None else z_obs
        self.D_obs = plate.D_row(self.x_obs, self.z_obs)[:n]
        self.H = np.asarray(plate.H_Pe_modal, float)[:n]
        # direction nominale de la force de coupe, compatible avec DD0 :
        # D_nom D_nom^T ~ DD0 (DD0 est de rang ~1)
        DD0 = self.par['DD0']
        s12 = np.sign(DD0[0, 1]) if DD0[0, 1] != 0 else 1.0
        self.D_nom = np.array([np.sqrt(max(DD0[0, 0], 0.0)),
                               s12 * np.sqrt(max(DD0[1, 1], 0.0))])
        # --- ponderations ---------------------------------------------------
        self.Wf = synthesis_weights(wpf, wpu, wpn)[0]
        self.Wu = synthesis_weights(wpf, wpu, wpn)[1]
        self.Wn = synthesis_weights(wpf, wpu, wpn)[2]
        self.Waf = W_Paf_ss()
        self.Wau = W_Pau_ss()
        self._build()

    # ------------------------------------------------------------------
    def _uncertainty_channels(self):
        """Retourne (Cq, Cqd, Ca_flag, Bcol) pour les dix canaux scalaires.

        Chaque canal j est defini par :
          - sa SORTIE : 'a' (acceleration i), 'v' (vitesse i) ou 'q' (position i)
          - sa ligne d'ENTREE : coefficient -L dans l'equation de a_i.
        """
        p = self.par
        n = self.n
        ch = []
        for i in range(n):                       # d_Pm : masse
            ch.append(dict(name=self.PARAM_NAMES[i], out=('a', i),
                           row=i, gain=-p['L_Pm'][i]))
        for i in range(n):                       # d_Pc : amortissement
            ch.append(dict(name=self.PARAM_NAMES[2 + i], out=('v', i),
                           row=i, gain=-p['L_Pc'][i]))
        for i in range(n):                       # d_Pk : raideur modale
            ch.append(dict(name=self.PARAM_NAMES[4 + i], out=('q', i),
                           row=i, gain=-p['L_Pk'][i]))
        # d_PD1..d_PD4 : perturbation de alpha4 D^T D (Eq. 25)
        LPD = p['L_PD']
        ch.append(dict(name='dPD1', out=('q', 0), row=0, gain=-LPD[0, 0]))
        ch.append(dict(name='dPD2', out=('q', 1), row=0, gain=-LPD[0, 1]))
        ch.append(dict(name='dPD3', out=('q', 0), row=1, gain=-LPD[1, 0]))
        ch.append(dict(name='dPD4', out=('q', 1), row=1, gain=-LPD[1, 1]))
        return ch

    def _build(self):
        n, p = self.n, self.par
        A21 = -(p['K0'] + self.sign * p['alpha40'] * p['DD0'])
        A22 = -p['C0']
        Ap = np.block([[np.zeros((n, n)), np.eye(n)], [A21, A22]])
        Bf = np.concatenate([np.zeros(n), self.D_nom])[:, None]      # force
        Bu = np.concatenate([np.zeros(n), self.H])[:, None]          # tension

        ch = self._uncertainty_channels()
        nu = len(ch)
        Bunc = np.zeros((2 * n, nu))
        for j, c in enumerate(ch):
            Bunc[n + c['row'], j] = c['gain']

        # sorties d'incertitude
        Cunc = np.zeros((nu, 2 * n))
        Dunc_unc = np.zeros((nu, nu))
        Dunc_f = np.zeros((nu, 1))
        Dunc_u = np.zeros((nu, 1))
        for j, c in enumerate(ch):
            kind, i = c['out']
            if kind == 'a':                       # acceleration : boucle algebrique
                Cunc[j, :] = np.concatenate([A21[i], A22[i]])
                Dunc_unc[j, :] = Bunc[n + i, :]
                Dunc_f[j, 0] = Bf[n + i, 0]
                Dunc_u[j, 0] = Bu[n + i, 0]
            elif kind == 'v':
                Cunc[j, n + i] = 1.0
            else:
                Cunc[j, i] = 1.0

        self.ss_plant = dict(A=Ap, Bunc=Bunc, Bf=Bf, Bu=Bu, Cunc=Cunc,
                             Dunc_unc=Dunc_unc, Dunc_f=Dunc_f, Dunc_u=Dunc_u,
                             Cy=np.concatenate([self.D_obs, np.zeros(n)])[None, :],
                             channels=ch)

    # ------------------------------------------------------------------
    def generalized_plant(self):
        """Plant generalise P (Fig. 10), en etat-espace.

        entrees  : [u_Pa (1) | u_Pr (10) | F_PIr (1) | n_P (1) | u_P (1)]
        sorties  : [y_Paf, y_Pau (2) | y_Pr (10) | z_disp, z_u (2) | y_mes (1)]
        etats    : [q, q' (4) | W_Paf (2) | W_Pau (2) | W_Pf (1) | W_Pu (1)]
        """
        sp = self.ss_plant
        n = self.n
        nx_p = 2 * n
        Aaf, Baf, Caf, Daf = self.Waf
        Aau, Bau, Cau, Dau = self.Wau
        Af, Bf_w, Cf, Df = self.Wf
        Au, Bu_w, Cu, Du = self.Wu
        An, Bn, Cn, Dn = self.Wn                       # constante : 0 etat
        nz = [nx_p, Aaf.shape[0], Aau.shape[0], Af.shape[0], Au.shape[0]]
        off = np.cumsum([0] + nz)
        N = off[-1]
        nu_unc = sp['Bunc'].shape[1]
        n_in = 1 + nu_unc + 1 + 1 + 1
        n_out = 2 + nu_unc + 2 + 1

        A = np.zeros((N, N))
        B = np.zeros((N, n_in))
        C = np.zeros((n_out, N))
        D = np.zeros((n_out, n_in))

        i_ua, i_ur, i_f, i_np, i_up = 0, 1, 1 + nu_unc, 2 + nu_unc, 3 + nu_unc
        o_af, o_au, o_ur, o_z1, o_z2, o_y = (0, 1, 2, 2 + nu_unc,
                                             3 + nu_unc, 4 + nu_unc)
        sl = [slice(off[k], off[k + 1]) for k in range(5)]

        # -- plaque ---------------------------------------------------------
        A[sl[0], sl[0]] = sp['A']
        B[sl[0], i_ur:i_ur + nu_unc] = sp['Bunc']
        B[sl[0], i_f:i_f + 1] = sp['Bf']
        B[sl[0], i_up:i_up + 1] = sp['Bu']
        # -- W_Paf (entree = F_PIr) -----------------------------------------
        A[sl[1], sl[1]] = Aaf
        B[sl[1], i_f:i_f + 1] = Baf
        C[o_af, sl[1]] = Caf
        D[o_af, i_f] = Daf[0, 0]
        # -- W_Pau (entree = u_P) -------------------------------------------
        A[sl[2], sl[2]] = Aau
        B[sl[2], i_up:i_up + 1] = Bau
        C[o_au, sl[2]] = Cau
        D[o_au, i_up] = Dau[0, 0]
        # -- canaux d'incertitude parametrique ------------------------------
        C[o_ur:o_ur + nu_unc, sl[0]] = sp['Cunc']
        D[o_ur:o_ur + nu_unc, i_ur:i_ur + nu_unc] = sp['Dunc_unc']
        D[o_ur:o_ur + nu_unc, i_f] = sp['Dunc_f'][:, 0]
        D[o_ur:o_ur + nu_unc, i_up] = sp['Dunc_u'][:, 0]
        # -- sortie physique y_PO = D_obs q + u_Pa ---------------------------
        Cy = sp['Cy']
        # -- W_Pf (performance : deplacement pondere) ------------------------
        A[sl[3], sl[3]] = Af
        A[sl[3], sl[0]] = Bf_w @ Cy
        B[sl[3], i_ua:i_ua + 1] = Bf_w
        C[o_z1, sl[3]] = Cf
        C[o_z1, sl[0]] = Df @ Cy
        D[o_z1, i_ua] = Df[0, 0]
        # -- W_Pu (performance : tension ponderee) ---------------------------
        A[sl[4], sl[4]] = Au
        B[sl[4], i_up:i_up + 1] = Bu_w
        C[o_z2, sl[4]] = Cu
        D[o_z2, i_up] = Du[0, 0]
        # -- mesure y_mu = y_PO + W_Pn n_P -----------------------------------
        C[o_y, sl[0]] = Cy
        D[o_y, i_ua] = 1.0
        D[o_y, i_np] = Dn[0, 0]

        blocks = ([('F', 2, 1)]                       # Delta_Pa : plein 1x2
                  + [('S', 1, 1)] * nu_unc            # dix scalaires reels
                  + [('P', 2, 2)])                    # bloc de performance

        # echelle initiale des canaux d'incertitude (D-scale constante, donc
        # sans effet sur mu) : rend les canaux 'acceleration', 'vitesse' et
        # 'position' comparables — indispensable au conditionnement de SB10AD
        q0, w0 = 1e-6, 2 * np.pi * 1000.0
        dscale = [1.0 / q0]                            # Delta_Pa
        for c in sp['channels']:
            kind = c['out'][0]
            dscale.append({'a': 1.0 / (q0 * w0**2), 'v': 1.0 / (q0 * w0),
                           'q': 1.0 / q0}[kind])
        return dict(A=A, B=B, C=C, D=D, nmeas=1, ncon=1,
                    blocks=blocks, n_unc_in=1 + nu_unc, n_unc_out=2 + nu_unc,
                    n_w=2, n_z=2, dscale_init=np.array(dscale),
                    names=['Delta_Pa'] + self.PARAM_NAMES + ['Delta_Pp'])

    # ------------------------------------------------------------------
    def nominal_open_loop(self):
        """(A, B_f, B_u, C_y) du systeme nominal reduit, sans ponderation."""
        sp = self.ss_plant
        return sp['A'], sp['Bf'], sp['Bu'], sp['Cy']
