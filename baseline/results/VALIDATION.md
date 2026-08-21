# FOPID contre ADRC-FOPID — comparaison equitable, optimisation PSO

Modele : plaque mince fraisee (Du, Liu, Dai & Long, *IJMS* **274** (2024) 109257,
Section 2), pastille piezoelectrique au coin inferieur droit, capteur au coin
superieur droit. Convention de couplage `SIGN_SIM = -1`.

## Protocole d'equite

| element | FOPID | ADRC-FOPID |
|---|---|---|
| modele, pastille, capteur, signe | identiques | identiques |
| operateurs d'ordre fractionnaire | Oustaloup, bande [1 Hz, 100 kHz], N = 3 | idem |
| lissage anti-repliement | Butterworth 2e ordre, 8 kHz | idem |
| fonction objectif | `objective.evaluate` | idem |
| contraintes | Ms <= 2, effort <= 450 V/N, stabilite nominale | idem |
| PSO | memes reglages, memes graines (1,2,3), meme budget | idem |
| evaluation finale | Floquet m = 200, modele complet 5 modes | idem |
| **parametres** | **5** (Kp, Ki, Kd, lambda, mu) | **7** (+ w_o, b0) |
| etats du correcteur | 16 | 19 |

Le nombre de parametres differe : c'est l'objet meme de la comparaison, et il
est rapporte plutot que masque.

## Protocole A — synthese sur le modele reduit (2 modes), evaluation sur 5

C'est le cas realiste : le correcteur est concu sur le modele simplifie du
papier, puis confronte a la structure complete.

| grandeur | boucle ouverte | FOPID | ADRC-FOPID |
|---|---|---|---|
| lobes, moyenne 3000-7000 tr/min (mm) | 0.255 | 2.106 | **3.421** |
| lobes, minimum (mm) | 0.065 | 1.078 | **2.262** |
| limite a 4900 tr/min, min sur le bord (mm) | 0.224 | 1.487 | **2.960** |
| deplacement max a a_p = 0.6 mm (um) | diverge (0.175 s) | 4.07 | **3.80** |
| tension max (V) | — | 40.1 | **17.8** |
| tension moyenne (V) | — | 6.79 | **4.88** |
| marge de module Ms | — | 1.997 | **1.881** |
| effort (V/N) | — | 392 | **160** |
| pole nominal le plus lent (s^-1) | — | -7.9 | **-21.3** |

### Robustesse

| modele d'evaluation | boucle ouverte | FOPID | ADRC-FOPID |
|---|---|---|---|
| 2 modes (celui de la synthese) | 0.158 | **3.727** | 1.997 |
| **5 modes (verite)** | 0.220 | 1.483 | **2.964** |
| derive modale +17 % / +9 % | 0.049 | **3.977** | 3.384 |
| amortissement x 0.8 | 0.127 | **3.712** | 1.982 |

Lecture : le FOPID est meilleur DANS le modele qui a servi a le regler, et
perd 60 % de sa limite des que les modes 3 a 5 apparaissent. L'ADRC-FOPID fait
l'inverse : il gagne 48 % en passant a 5 modes. L'observateur d'etat etendu
absorbe les modes non modelises comme une perturbation et les rejette, ce qui
est exactement son argument de vente ; le FOPID, lui, n'a aucun mecanisme pour
cela et son reglage PSO est specialise au modele de synthese.

Conclusion : sur le modele-verite, **l'ADRC-FOPID double la limite de coupe
du FOPID (2.96 mm contre 1.49) tout en demandant deux fois moins de tension
(17.8 V contre 40.1)** — avec deux parametres de plus.

## Reserve importante

Ces resultats correspondent au protocole A. Le protocole B (synthese ET
evaluation sur 5 modes) mesure la performance brute a modele parfait ; il est
relance dans `run_pso.py` avec la configuration actuelle
(`N_MODES = 5`) et doit etre reconduit avant toute conclusion definitive sur
la performance maximale atteignable par chaque structure. Le protocole A reste
le plus representatif de l'usage reel.
