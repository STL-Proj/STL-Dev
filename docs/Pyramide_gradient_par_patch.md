# Gradient par patch grâce au stockage en pyramide

Note destinée au développeur qui portera, pour la classe à convolution par
noyau `STL_2D_Kernel_Torch`, puis pour ses extensions 2D+1D et 2D+1D+1D, ce
qui a été fait sur la sphère dans la branche `HealpixUpdate` de STL-Dev. Elle
explique d'abord l'idée, puis comment elle est codée pour HEALPix, puis ce qui
change (et surtout ce qui se simplifie) sur une grille régulière.

## 1. Le problème : S3 et S4 relient des échelles éloignées

Les coefficients S1 et S2 vivent à une seule échelle : `|ψ_j ⋆ x|` et
`(ψ_j ⋆ x)(ψ_j ⋆ x)*` sont des moyennes de champs calculés à la résolution de
l'échelle `j`. Le gradient d'un pixel ne dépend que d'un voisinage de la
taille du noyau. Les découper en tuiles est immédiat.

S3 est différent :

    S3(j1, j3) = moyenne_n [ ψ_j3 ⋆ D^(n−d) |ψ_j1 ⋆ x| ] · conj(ψ_j3 ⋆ x)

avec `d` la résolution de `j1` (fine) et `n` celle de `j3` (grossière),
`D` la décimation (lissage puis sous-échantillonnage d'un facteur 2 par axe).
Un pixel grossier de niveau `n` dépend, à travers la chaîne
« ondelette fine → module → `n − d` décimations → ondelette grossière », de
`2^(n−d)` pixels fins par axe, multiplié par la taille du noyau. Pour
`j1 = 0` et `j3 = J − 1`, c'est tout le domaine. S4 a la même structure avec
deux modules décimés.

### Le piège : découper naïvement en patchs

La première idée est de découper le niveau grossier en patchs et de calculer,
pour chaque patch, toute la chaîne depuis les pixels fins. On l'a essayé
(`STL_main/PatchScattering.py`) : c'est exact, mais le halo nécessaire au
niveau fin vaut `halo_grossier × 2^(n−d)` par axe. À `J = 5` avec un noyau de
5, un patch grossier de quelques pixels réclame un domaine fin de plusieurs
dizaines de pour cent de l'image. La mémoire ne baisse presque pas et le temps
explose à cause du recouvrement des halos. **Ne pas refaire cela.**

## 2. L'idée : stocker la pyramide des modules décimés

Deux observations font tout le travail.

**L'ondelette est la même à tous les niveaux.** Le premier champ d'ondelettes
ne dépend que du niveau de résolution `d`, pas de l'indice d'échelle :
`z_d = ψ ⋆ maps[d]` avec `maps[d] = D^d x`.

**Après le module, la chaîne est linéaire.** Décimation et ondelette
grossière sont des opérateurs linéaires. Donc, pour toute partition du niveau
fin `d` en blocs `F_k` :

    m[d][n] = D^(n−d) |z_d| = Σ_k D^(n−d) ( |z_d| · 1_{F_k} )

Chaque bloc fin calcule sa contribution seul et la pousse vers le niveau
grossier, où les contributions s'additionnent. On **stocke** alors, pour
chaque paire de niveaux `d < n`, la carte réelle `m[d][n]` au niveau `n`.

Conséquence décisive : un bloc du niveau `n` n'a plus besoin des pixels fins.
Il lit `maps[n]`, `|z_n|` (qu'il recalcule sur son bloc + halo) et les
`m[d][n]` stockés, **tous au niveau `n`, sur son bloc + un halo de quelques
pixels de ce niveau**. Le couplage entre échelles éloignées est porté par des
cartes stockées à la résolution grossière, pas par un champ réceptif fin à
recalculer.

### Ce que coûte la pyramide

`m[d][n]` a la taille du niveau `n` : `N / 4^n` pixels en 2D (`N` = pixels du
niveau 0), fois `C × L` (canaux, orientations), en réels. Le total sur toutes
les paires vaut `Σ_n n · N / 4^n ≈ 4N/9` valeurs par `C × L`, plus autant pour
leurs cotangents dans la passe inverse. C'est moins que deux cartes pleine
résolution, et surtout **sans axe d'orientation complexe au niveau fin**.

Le reste de la mémoire est par bloc : quelques champs de taille
`(bloc + halo) × C × L` en complexe. C'est ce qui fait descendre la mémoire
avec la taille du bloc.

## 3. Deux balayages pour un gradient exact

La perte est globale, `Σ_i w_i |S_i(x) − S_i(cible)|²`, et chaque
coefficient est une moyenne de pixels à son niveau natif `n`, donc une somme
de sommes partielles par bloc divisée par le nombre total de pixels du niveau.

**Balayage montant (niveaux 0 → n_max)**, pour chaque bloc de chaque niveau :

1. calculer `z_n` sur le bloc + halo (une convolution sur bloc + 2 halos) ;
2. calculer les secondes ondelettes sur le bloc à partir de `|z_n|` (niveau
   propre) et des `m[d][n]` stockés (niveaux plus fins) ;
3. accumuler les sommes partielles de tous les coefficients de niveau natif
   `n` ;
4. pousser `|z_n| · 1_bloc` à travers la chaîne de décimation vers tous les
   niveaux plus grossiers `n'` et l'ajouter dans `m[n][n']`.

À la fin, on a toutes les statistiques, donc les cotangents
`g_i = 2 w_i (S_i − cible_i)`.

**Balayage descendant (niveaux n_max → 0)**, pour chaque bloc :

1. recalculer les champs du bloc (pas de stockage entre les deux balayages) ;
2. appliquer les cotangents des coefficients de niveau `n` ;
3. adjoint de la seconde ondelette : cotangent des modules sur le bloc + halo ;
   la part qui concerne un niveau fin `d` s'ajoute dans `gm[d][n]` (stocké),
   la part du niveau propre va directement à la dérivée du module ;
4. comme on descend, `gm[n][n']` est **complet** quand on arrive au niveau
   `n` (tous les niveaux plus grossiers sont faits) : on le tire à travers les
   transposées de la chaîne de décimation restreintes au bloc ;
5. dérivée du module, adjoint de la première ondelette, accumulation du
   cotangent de `maps[n]` sur le bloc + halo.

Enfin, les cotangents de `maps[d]` pour tous les `d` sont ramenés au niveau 0
par les transposées des décimations (`maps[d] = D^d x`), ce qui donne le
gradient par rapport à la carte.

Ce gradient est **identique** au gradient plein domaine (tests à 1e-10 sur la
sphère). Ce n'est pas une approximation par patchs : c'est le même calcul,
réordonné.

## 4. Comment c'est codé pour la sphère

Fichiers de la branche `HealpixUpdate` :

| Fichier | Rôle |
|---|---|
| `STL_main/LocalScattering.py` | le moteur complet : `LocalEngine`, `_Block`, `apply_local`, `vjp_local`, `loss_and_grad_local` |
| `STL_main/LocalGeometry.py` | blocs, halos, opérations par bloc : `blocks`, `dilate_fine`, `conv`, `down`, `decimate`, `decimate_adjoint`, `piece_forward`, `piece_adjoint`, `analysis_levels`, `analysis_vjp`, `pyramid_from_map` |
| `STL_main/TorchGeometry.py` | construction sur GPU des opérateurs partiels HEALPix (convolution et décimation) ; **sans objet en 2D** |
| `STL_main/AdjointOps.py` | VJP élémentaires : `modulus_vjp`, `root_modulus_vjp`, `covariance_vjp`, `convolution_adjoint` |
| `STL_main/PyramidScattering.py` | `build_plan` (liste des coefficients avec leur niveau natif `native_dg`), `PyramidStatistics` |
| `STL_main/LossOptim.py` | `PyramidSynthesis(geometry=..., variables="map")`, `PyramidLBFGS` |
| `tests/test_local_scattering.py` | tests d'exactitude contre le moteur plein ciel |

Lecture conseillée de `LocalScattering.py`, dans cet ordre :

- `LocalEngine.__init__` : alloue `self.m[d][n]` (réel, `[B, C, L, N_n]`) pour
  toutes les paires `d < n` ;
- `LocalEngine.forward` : le balayage montant ; la boucle « Push |z_n| on F
  to every coarser level » est le cœur de l'idée ;
- `_Block.__init__` : définit `a1 = bloc + halo`, `domain = a1 + halo`, et
  calcule `z_n` sur `a1` ;
- `_Block.modulus(d)` : `|z_n|` si `d == n`, sinon lecture de
  `engine.m[d][n]` sur `a1` ;
- `_Block.branch(d)` : seconde ondelette de `modulus(d)`, gardée sur le bloc ;
- `_Block.operands` / `partial` : les opérandes de chaque famille, et la
  somme partielle divisée par le nombre total de pixels du niveau ;
- `_Block.reverse_coefficient` : les cotangents de chaque famille ;
- `_Block.finish_reverse` : adjoint des secondes ondelettes, routage vers
  `gm`, chaîne inverse du bloc, dérivée du module, adjoint de la première
  ondelette ;
- `LocalEngine.reverse` : le balayage descendant.

Sur la sphère, la difficulté principale n'était pas l'algorithme mais la
géométrie : les opérateurs HEALPix sont des matrices creuses irrégulières, il
a fallu construire pour chaque bloc des opérateurs partiels avec un halo en
anneaux de voisins (`kth_neighbourhood`) et ne garder que les lignes dont le
stencil est complet. C'est l'essentiel de `LocalGeometry.py` et tout
`TorchGeometry.py`.

## 5. Ce qui change en 2D noyau (`STL_2D_Kernel_Torch`)

**Tout devient plus simple** : la grille est régulière, les blocs sont des
tuiles rectangulaires, la convolution est `F.conv2d` et la décimation est
`F.conv2d(stride=2)` avec le noyau de lissage (ou la moyenne 2×2). Pas de
géométrie à construire, pas de cache, pas de lignes « incomplètes ».

Les opérations à recoder (dans `WaveletOperator2Dkernel_torch`) :

- `apply(data, j)` → `_semicomplex_conv2d_circular` avec `pad = K // 2` puis
  `F.conv2d` ;
- `downsample(data, dg_out)` → `_downsample_tensor`, pas à pas,
  `pad_before = (Ks − 1) // 2`, `pad_after = Ks // 2` (noyau de lissage
  `Ks` = 2, 3 ou 6 selon `_get_smooth_kernel`), stride 2.

### Halos exacts en 2D

Pour une tuile de lignes `[i0, i1)` et colonnes `[k0, k1)` au niveau `n` :

- **convolution** : halo `h_c = K // 2` pixels du niveau de chaque côté ;
- **décimation `n → n+1`** : le pixel grossier `i` lit les pixels fins
  `2i − pad_before … 2i + pad_after` ; la tuile grossière `[i0/2, i1/2)` a donc
  besoin de la tuile fine élargie de `pad_before` avant et `pad_after` après ;
- **pièce poussée vers le haut** (un champ fin supporté sur une tuile, poussé
  à travers `D`) : les lignes grossières touchées sont
  `[(i0 − pad_after) // 2, (i1 − 1 + pad_before) // 2 + 1)` ; rien n'est
  tronqué puisqu'on part d'un champ nul hors tuile ;
- **transposée** : symétrique, on tire depuis ces lignes grossières vers la
  tuile fine.

Avec les tuiles alignées sur des puissances de 2 (taille de tuile divisible
par `2^n_max`), une tuile du niveau `d` a pour image au niveau `d+1` une
tuile de taille moitié, exactement alignée. C'est l'analogue NESTED de la
sphère, en plus simple.

### Bords du domaine

La seule subtilité 2D est le bord global :

- `pbc=True` (padding `circular`) : le halo d'une tuile au bord **s'enroule**.
  Le plus simple et le plus sûr est de lire la tuile élargie par indexation
  modulo (`torch.arange(i0 − h, i1 + h) % H`, idem en colonnes) puis de
  convoluer en mode `valid` (sans padding) : la sortie est exactement la tuile,
  identique au calcul plein domaine.
- `pbc=False` (padding `replicate`) : même chose avec `clamp(0, H − 1)` au lieu
  du modulo. C'est exactement ce que fait `F.pad(mode="replicate")` sur l'image
  entière.

Avec ces deux règles, le résultat sur la tuile est bit à bit celui du calcul
plein domaine, sans aucune renormalisation.

### Moyennes et rognage des bords (`pbc=False`)

En non périodique, `ST_Operator` rogne une bordure avant les moyennes
(`_get_crop_border_size_*` dans `STL_2D_Kernel_Torch.py`). Il faut reproduire
ce rognage : la somme partielle d'une tuile ne porte que sur ses pixels situés
dans la région rognée, et le diviseur est le nombre de pixels de la région
rognée du niveau, pas de tout le niveau. Dans la passe inverse, le cotangent
est nul hors de cette région. C'est le point le plus facile à rater ; le tester
explicitement.

### Ce qu'il faut écrire

1. un objet `LocalGrid2D` (équivalent de `LocalGeometry`) : liste des tuiles
   par niveau, lecture d'une tuile élargie (modulo ou clamp), convolution et
   adjoint sur tuile, décimation, pièce et transposée sur tuile, passage
   `maps[d] = D^d x` et son adjoint par tuiles ;
2. un `LocalEngine2D` qui reprend **tel quel** la logique de
   `LocalScattering.LocalEngine` et `_Block` : seules changent la forme des
   indices (deux axes au lieu d'un identifiant de pixel) et les opérateurs ;
3. un `build_plan` 2D (celui de `PyramidScattering.build_plan` refuse tout ce
   qui n'est pas HEALPix, mais l'énumération des coefficients est la même) ;
4. le branchement dans `PyramidSynthesis` via `geometry=`, avec
   `variables="map"` (les bandes laplaciennes conditionnent mal LBFGS).

Les adjoints s'obtiennent facilement : l'adjoint de `F.conv2d` (mode `valid`)
est `F.conv_transpose2d` avec le même noyau ; celui d'une décimation à stride 2
est `F.conv_transpose2d(stride=2)` ; ou bien `torch.autograd.grad` appliqué
**sur une seule tuile**, ce qui reste borné en mémoire et sert de référence
pendant le développement.

### Tests à écrire (critère d'acceptation)

Il n'existe pas de moteur explicite plein domaine en 2D, donc la référence est
la synthèse classique avec autograd sur une petite image :

- statistiques : `ST_Operator.apply(data)` contre le moteur par tuiles, toutes
  familles, `pbc=True` et `pbc=False`, plusieurs tailles de tuile, à 1e-10 ;
- gradient : `torch.autograd.grad(perte(apply(x)), x)` contre le gradient par
  tuiles, mêmes cas ;
- invariance à la taille de tuile, et égalité quand une seule tuile couvre tout
  le niveau (on retrouve le calcul plein domaine).

## 6. Extensions 2D+1D et 2D+1D+1D

L'argument ne dépend pas de la dimension. Il repose sur trois propriétés qu'il
faut vérifier pour chaque extension :

1. **une pyramide** : chaque niveau est une décimation locale du précédent ;
2. **une ondelette par niveau** identique d'un niveau à l'autre (sinon il faut
   indexer `m` par échelle plutôt que par niveau, ce qui reste possible mais
   coûte plus de mémoire) ;
3. **la linéarité après le module** : décimation et seconde ondelette
   linéaires.

Ce qui change :

- **les tuiles** deviennent des pavés 3D ou 4D, avec un halo par axe calculé
  comme en 2D (noyau propre à chaque axe) ;
- **si les axes ont des pyramides indépendantes** (par exemple échelles
  spatiales `j` et échelles temporelles ou spectrales `k`), un niveau devient
  un couple `(n_x, n_t)` et les modules stockés deviennent
  `m[(d_x, d_t)][(n_x, n_t)]` pour les couples `d ≤ n` composante par
  composante, avec au moins une inégalité stricte. La décimation est le produit
  des décimations par axe, toujours linéaire. Le balayage montant parcourt les
  couples par ordre croissant de `n_x + n_t` (ou tout ordre topologique du
  treillis), le descendant en sens inverse ; la règle « `gm[d][n]` est complet
  quand tous les niveaux plus grossiers que `n` sont faits » reste vraie.
  Attention au coût : le nombre de paires croît vite, à estimer avant de coder ;
- **si l'axe 1D n'est pas décimé** (seulement des convolutions le long de cet
  axe), il n'entre pas dans la pyramide : on découpe librement le long de cet
  axe avec un halo de convolution, et la pyramide reste celle de la partie
  spatiale ;
- **si les ondelettes sont séparables** (2D ⊗ 1D), la convolution sur un pavé
  se fait axe par axe et le halo est le produit des halos.

## 7. Résultats obtenus sur la sphère, pour fixer les ordres de grandeur

Cible WebSky (convergence de lentillage et kSZ), `L = 4`, toutes les familles,
LBFGS sur la carte, un GPU de 24 Go :

| nside | pixels | coefficients | temps par évaluation | pic GPU | plein domaine classique |
|---|---|---|---|---|---|
| 256 | 0,8 M | 5 882 | 5,9 s | 3,2 Go | possible |
| 1024 | 12,6 M | 11 354 | 108 s | 8,2 Go | 16 Go de géométrie seule |
| 2048 | 50 M | ≈ 14 000 | quelques minutes | ≈ 10 Go | impossible |

En 2D, l'absence de géométrie à construire supprime le poste dominant du temps
sur la sphère (environ la moitié) ; on peut s'attendre à un moteur par tuiles
proche du coût du calcul plein domaine, avec une mémoire bornée par la taille
de tuile et la pyramide stockée.

## 8. Pièges rencontrés sur la sphère (à éviter en 2D)

- Ne pas découper le niveau grossier et recalculer la chaîne depuis les pixels
  fins (section 1) : il faut la pyramide stockée.
- Optimiser **la carte**, pas les bandes d'une pyramide laplacienne :
  `PyramidSynthesis(variables="map")`.
- Comparer toujours au calcul plein domaine sur de petites tailles, toutes
  familles et les deux définitions de S3/S4 (`has_fewer_convolutions`).
- Initialisation grossière vers fine pour les grandes tailles : synthèse à
  basse résolution, réplication sur les enfants, plus un bruit blanc dont
  l'écart-type est le RMS des petites échelles manquantes de la cible
  (`benchmark_synthesis.py --init ... --init-noise auto`).
