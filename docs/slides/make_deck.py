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
    """runs: str or list of paragraphs, each a str or a list of (text, options)."""
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


# ---- 1. Why S3 is the hard coefficient -------------------------------------------
s = prs.slides.add_slide(blank)
title(s, "S3 couples a fine wavelet with a coarse one")
text(s, 0.6, 1.35, 12.1, 0.9, [
    [("S3(j1, j3) = mean over the coarse pixels of  ", {}),
     ("ψ_j3 ⋆ D^(n−d) |ψ_j1 ⋆ x|", {"bold": True, "color": ACCENT}),
     ("  ×  conj(ψ_j3 ⋆ x)", {"bold": True}),
     ("     with d the resolution of j1 and n the resolution of j3", {"size": 13, "color": MUTED})],
], size=16)
picture(s, "s3_chain.png", 0.6, 2.25, w=12.1)
text(s, 0.6, 5.85, 6.0, 1.4, [
    [("S1 and S2 live at a single scale: ", {"bold": True}), ("they are naturally computed level by level.", {})],
    [("S3 ties two scales together: ", {"bold": True}),
     ("the gradient of one coarse pixel reads 4^(n−d) fine pixels through the chain, i.e. the whole sky for j1 = 0, j3 = J−1.", {})],
], size=14)
text(s, 7.0, 5.85, 5.7, 1.4, [
    [("At nside 2048 (50 M pixels), a single complex field ψ ⋆ x with L = 4 orientations is 3.2 GB, ", {}),
     ("and the full-sky convolution geometry 250 GB.", {"bold": True})],
    [("The whole sky cannot be held in memory: it has to be split.", {"color": ACCENT, "bold": True})],
], size=14)

# ---- 2. Splitting the chain into blocks ------------------------------------------------
s = prs.slides.add_slide(blank)
title(s, "Splitting the chain into blocks: linearity does the work")
picture(s, "pyramid.png", 0.6, 1.5, w=7.6)
text(s, 8.5, 1.45, 4.3, 5.6, [
    [("After the modulus, everything is linear", {"bold": True, "color": ACCENT})],
    [("Decimation and the coarse wavelet are linear operators: the contribution of each fine block F_k is computed on its own and summed at the coarse level.", {})],
    [("One block at a time, with its halo", {"bold": True, "color": ACCENT})],
    [("The halo (a few pixels: wavelet stencil, decimation cone) makes every block exact in its interior; border rows are never used.", {})],
    [("Same block size at every level", {"bold": True, "color": ACCENT})],
    [("Coarse levels smaller than one block are processed whole: the former full-sky computation, where it costs nothing.", {})],
    [("Geometry recomputed on the GPU block by block: nothing to store, neither geometry nor full-sky field.", {"size": 13, "color": MUTED})],
], size=14, spacing=8)

# ---- 3. The optimisation ---------------------------------------------------------------
s = prs.slides.add_slide(blank)
title(s, "An exact gradient assembled by two sweeps")
picture(s, "sweeps.png", 0.6, 1.6, w=7.4)
text(s, 8.3, 1.45, 4.5, 5.6, [
    [("Global loss, global variable", {"bold": True, "color": ACCENT})],
    [("Σ |S(x) − S(target)|² over all families; the whole map is the variable; LBFGS with line search, as in the classical synthesis.", {})],
    [("Upward: the statistics", {"bold": True, "color": ACCENT})],
    [("Level by level, block by block: partial sums of the coefficients, and decimated moduli m[d][n] pushed to the coarser levels (real maps, four times smaller per level).", {})],
    [("Downward: the gradient", {"bold": True, "color": ACCENT})],
    [("Cotangents of the coefficients, then in reverse order: transposes of the chain, modulus derivative, wavelet adjoint, accumulated on the map.", {})],
    [("Identical to the full-sky computation to 1e-10; same LBFGS trajectory as the classical synthesis.", {"size": 13, "color": MUTED})],
], size=14, spacing=8)

# ---- 4. Results --------------------------------------------------------------------------
s = prs.slides.add_slide(blank)
title(s, "WebSky: from nside 256 to nside 2048 on a 24 GB GPU")
picture(s, "loss_256.png", 0.6, 1.4, w=6.2)
rows = [
    ("nside", "pixels", "coefs", "per evaluation", "GPU peak", "classical full sky"),
    ("256", "0.8 M", "5,882", "5.9 s", "3.2 GB", "possible"),
    ("1024", "12.6 M", "11,354", "108 s", "8.2 GB", "16 GB of geometry alone"),
    ("2048", "50 M", "≈ 14,000", "≈ 7 min", "≈ 10 GB", "250 GB: impossible"),
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
    [("L = 4 orientations, J = 7 to 9 scales, all families S1 to S4; target: WebSky CMB lensing convergence (Stein et al. 2020).", {"size": 12, "color": MUTED})],
], size=12)
text(s, 0.6, 5.5, 12.1, 1.7, [
    [("What the curve shows: ", {"bold": True}),
     ("the loss drops by six orders of magnitude; the block-wise gradient is not an approximation, it is the same gradient computed with memory bounded by the block size.", {})],
    [("Going up in resolution: ", {"bold": True}),
     ("the nside 256 synthesis, replicated on the child pixels plus white noise calibrated on the target, directly initialises the nside 2048 synthesis, which only has to create the small scales.", {})],
], size=14, spacing=6)

out = os.path.join(HERE, "S3_patchwise_synthesis.pptx")
prs.save(out)
print("saved", out)
