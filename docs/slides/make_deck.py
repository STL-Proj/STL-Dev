"""Build the 4-slide deck on patch-wise S3 synthesis (python-pptx, 16:9, sober)."""

import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
INK, ACCENT, MUTED = RGBColor(0x36, 0x45, 0x4F), RGBColor(0x02, 0x80, 0x90), RGBColor(0x6B, 0x77, 0x80)
FONT = "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]


def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, spacing=6):
    """runs: str or list of (text, dict(bold/size/color))."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    paragraphs = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        pieces = para if isinstance(para, list) else [(para, {})]
        for piece, opts in pieces:
            r = p.add_run()
            r.text = piece
            r.font.name = FONT
            r.font.size = Pt(opts.get("size", size))
            r.font.bold = opts.get("bold", bold)
            r.font.color.rgb = opts.get("color", color)
    return box


def title(slide, s):
    text(slide, 0.6, 0.45, 12.1, 0.8, s, size=30, bold=True)


def picture(slide, name, x, y, w=None, h=None):
    kw = {}
    if w is not None:
        kw["width"] = Inches(w)
    if h is not None:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(os.path.join(FIG, name), Inches(x), Inches(y), **kw)


# ---- 1. Pourquoi S3 est le coefficient difficile ---------------------------------
s = prs.slides.add_slide(blank)
title(s, "S3 croise une ondelette fine et une ondelette grossière")
text(s, 0.6, 1.35, 12.1, 0.9, [
    [("S3(j1, j3) = moyenne sur les pixels grossiers de  ", {}),
     ("ψ_j3 ⋆ D^(n−d) |ψ_j1 ⋆ x|", {"bold": True, "color": ACCENT}),
     ("  ×  conj(ψ_j3 ⋆ x)", {"bold": True}),
     ("     avec d = résolution de j1, n = résolution de j3", {"size": 13, "color": MUTED})],
], size=16)
picture(s, "s3_chain.png", 0.6, 2.25, w=12.1)
text(s, 0.6, 5.85, 6.0, 1.4, [
    [("S1 et S2 vivent à une seule échelle : ", {"bold": True}), ("on les traite naturellement niveau par niveau.", {})],
    [("S3 relie deux échelles : ", {"bold": True}),
     ("le gradient d'un pixel grossier relit 4^(n−d) pixels fins à travers la chaîne, soit tout le ciel pour j1 = 0, j3 = J−1.", {})],
], size=14)
text(s, 7.0, 5.85, 5.7, 1.4, [
    [("À nside 2048 (50 M de pixels), un seul champ ψ ⋆ x complexe à L = 4 orientations pèse 3,2 Go, ", {}),
     ("et la géométrie de convolution plein ciel 250 Go.", {"bold": True})],
    [("Impossible d'avoir tout le ciel en mémoire : il faut découper.", {"color": ACCENT, "bold": True})],
], size=14)

# ---- 2. Découper la chaîne en patchs ------------------------------------------------
s = prs.slides.add_slide(blank)
title(s, "Découper la chaîne en blocs : la linéarité fait le travail")
picture(s, "pyramid.png", 0.6, 1.5, w=7.6)
text(s, 8.5, 1.45, 4.3, 5.6, [
    [("Après le module, tout est linéaire", {"bold": True, "color": ACCENT})],
    [("Décimation et ondelette grossière sont des opérateurs linéaires : la contribution de chaque bloc fin F_k se calcule seule et se somme au niveau grossier.", {})],
    [("Un bloc à la fois, avec son halo", {"bold": True, "color": ACCENT})],
    [("Le halo (quelques pixels : stencil de l'ondelette, cône de décimation) rend chaque bloc exact à l'intérieur ; les lignes de bord ne sont jamais utilisées.", {})],
    [("Même taille de bloc à chaque niveau", {"bold": True, "color": ACCENT})],
    [("Les niveaux grossiers, plus petits qu'un bloc, se traitent entiers : c'est l'ancien calcul plein ciel, là où il ne coûte rien.", {})],
    [("Géométrie recalculée sur le GPU bloc par bloc : rien à stocker, ni géométrie ni champ plein ciel.", {"size": 13, "color": MUTED})],
], size=14, spacing=8)

# ---- 3. L'optimisation -----------------------------------------------------------------
s = prs.slides.add_slide(blank)
title(s, "Un gradient exact assemblé par deux balayages")
picture(s, "sweeps.png", 0.6, 1.6, w=7.4)
text(s, 8.3, 1.45, 4.5, 5.6, [
    [("Perte globale, variable globale", {"bold": True, "color": ACCENT})],
    [("Σ |S(x) − S(cible)|² sur toutes les familles ; la carte entière est la variable ; LBFGS avec recherche linéaire, comme la synthèse classique.", {})],
    [("Montée : les statistiques", {"bold": True, "color": ACCENT})],
    [("Niveau par niveau, bloc par bloc : sommes partielles des coefficients, et modules décimés m[d][n] poussés vers les niveaux grossiers (cartes réelles, 4 fois plus petites à chaque niveau).", {})],
    [("Descente : le gradient", {"bold": True, "color": ACCENT})],
    [("Cotangents des coefficients, puis en sens inverse : transposées de la chaîne, dérivée du module, adjoint de l'ondelette, accumulés sur la carte.", {})],
    [("Identique au calcul plein ciel à 1e-10 ; même trajectoire LBFGS que la synthèse classique.", {"size": 13, "color": MUTED})],
], size=14, spacing=8)

# ---- 4. Résultats -----------------------------------------------------------------------
s = prs.slides.add_slide(blank)
title(s, "WebSky : de nside 256 à nside 2048 sur un GPU de 24 Go")
picture(s, "loss_256.png", 0.6, 1.4, w=6.2)
rows = [
    ("nside", "pixels", "coefs", "par évaluation", "pic GPU", "plein ciel classique"),
    ("256", "0,8 M", "5 882", "5,9 s", "3,2 Go", "possible"),
    ("1024", "12,6 M", "11 354", "108 s", "8,2 Go", "16 Go de géométrie seule"),
    ("2048", "50 M", "≈ 14 000", "≈ 7 min", "≈ 10 Go", "250 Go : impossible"),
]
tbl = s.shapes.add_table(len(rows), len(rows[0]), Inches(7.1), Inches(1.5), Inches(5.7), Inches(2.4)).table
for j, width in enumerate((0.7, 0.8, 0.8, 1.1, 0.8, 1.5)):
    tbl.columns[j].width = Inches(width)
for i, row in enumerate(rows):
    for j, cell_text in enumerate(row):
        cell = tbl.cell(i, j)
        cell.text = cell_text
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.name, r.font.size = FONT, Pt(12)
                r.font.bold = i == 0
                r.font.color.rgb = INK
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0xD9, 0xEE, 0xF0) if i == 0 else RGBColor(0xFF, 0xFF, 0xFF)
text(s, 7.1, 4.15, 5.7, 1.3, [
    [("L = 4 orientations, J = 7 à 9 échelles, toutes les familles S1 à S4, cible : convergence de lentillage CMB WebSky (Stein et al. 2020).", {"size": 12, "color": MUTED})],
], size=12)
text(s, 0.6, 5.5, 12.1, 1.7, [
    [("Ce que montre la courbe : ", {"bold": True}),
     ("la perte chute de six ordres de grandeur ; le gradient par blocs n'est pas une approximation, c'est le même gradient, calculé avec une mémoire bornée par la taille du bloc.", {})],
    [("Montée en résolution : ", {"bold": True}),
     ("la synthèse à 256, répliquée sur les pixels enfants plus un bruit blanc calibré sur la cible, initialise directement la synthèse à 2048, qui n'a plus qu'à créer les petites échelles.", {})],
], size=14, spacing=6)

out = os.path.join(HERE, "S3_synthese_par_blocs.pptx")
prs.save(out)
print("saved", out)
