"""Download a WebSky full-sky map and store it NESTED as .npy.

python -m benchmarks.scripts.prepare_websky_kappa --product kap --out data/websky --downgrade 1024 512
python -m benchmarks.scripts.prepare_websky_kappa --product ksz --out data/websky --downgrade 1024 512

Source: https://lambda.gsfc.nasa.gov/data/simulation/cmb_mocks/ (Stein et al.
2020, arXiv:2001.08787; nside 4096, full sky, 805 MB per map). The FITS maps
are RING ordered; the conversion to NESTED uses STL's torch ``ring2nest``.
Downgraded versions average the 4**k NESTED children (plain binning).
"""

import argparse
import os
import urllib.request

import numpy as np
import torch

BASE = "https://lambda.gsfc.nasa.gov/data/simulation/cmb_mocks/"
PRODUCTS = {
    "kap": "kap.fits",  # CMB lensing convergence
    "ksz": "ksz.fits",  # kinetic SZ (uK)
    "ksz_patchy": "ksz_patchy.fits",
    "tsz": "tsz.fits",  # Compton-y
    "isw": "isw.fits",
    "cib_143": "cib_nu0143.fits",  # MJy/sr
    "cib_217": "cib_nu0217.fits",
    "cib_353": "cib_nu0353.fits",
    "cib_545": "cib_nu0545.fits",
    "cib_857": "cib_nu0857.fits",
}


def read_healpix_fits(path):
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[1]
        header = hdu.header
        ordering = header.get("ORDERING", "RING").strip().upper()
        nside = int(header["NSIDE"])
        data = hdu.data
        column = data.columns[0].name
        values = np.asarray(data[column]).reshape(-1)
    return values, nside, ordering


def ring_to_nested(values, nside, chunk=16_000_000):
    from STL_main.TorchGeometry import ring2nest

    order = int(np.log2(nside))
    out = np.empty_like(values)
    for start in range(0, values.shape[0], chunk):
        stop = min(start + chunk, values.shape[0])
        ring = torch.arange(start, stop, dtype=torch.long)
        nest = ring2nest(ring, order).numpy()
        out[nest] = values[start:stop]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/websky")
    parser.add_argument(
        "--product", default="kap", choices=sorted(PRODUCTS),
        help="WebSky map: kap (lensing convergence), ksz, tsz, isw, cib_<GHz>",
    )
    parser.add_argument("--fits", default=None, help="existing FITS file (skips download)")
    parser.add_argument("--downgrade", type=int, nargs="*", default=[1024])
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    url = BASE + PRODUCTS[args.product]
    fits_path = args.fits or os.path.join(args.out, PRODUCTS[args.product])
    if not os.path.exists(fits_path):
        print(f"downloading {url} -> {fits_path}", flush=True)
        urllib.request.urlretrieve(url, fits_path)
    values, nside, ordering = read_healpix_fits(fits_path)
    print(f"nside={nside} ordering={ordering} dtype={values.dtype} npix={values.size} "
          f"mean={values.mean():.4e} std={values.std():.4e}", flush=True)
    if ordering == "RING":
        values = ring_to_nested(values, nside)
    elif ordering != "NESTED":
        raise ValueError(f"Unknown ORDERING {ordering}")
    name = os.path.join(args.out, f"{args.product}_nside{nside}_nested.npy")
    np.save(name, values)
    print(f"saved {name} ({os.path.getsize(name) / 1e6:.0f} MB)", flush=True)
    for target in args.downgrade:
        factor = (nside // target) ** 2
        low = values.reshape(-1, factor).mean(axis=1).astype(values.dtype)
        name = os.path.join(args.out, f"{args.product}_nside{target}_nested.npy")
        np.save(name, low)
        print(f"saved {name} ({os.path.getsize(name) / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
