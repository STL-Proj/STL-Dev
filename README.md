# STL (temporary package name)

Welcome to the Scattering Transform Library!

Doc: https://stl-dev.readthedocs.io/en/latest/

## HEALPix geometry backend

HEALPix geometry is provided by `healpix-analyse`: `HealPixConv` for
convolutions, `HealPixDown` for filtered decimation, and `HEALPixSHT` for
spherical harmonic transforms. The pyramid engine reuses the same geometry
and interpolation weights for its explicit gradients.

The former `STL_main.SphericalStencil` module has been removed. For direct
geometry operations, use the corresponding operators from `healpix_analyse`;
their APIs are not drop-in replacements for the old class. STL's current
HEALPix data and scattering APIs are unchanged by this removal.

## HEALPix pyramid synthesis

An opt-in engine provides native-resolution ScatCov coefficients, explicit
gradients and synthesis in Laplacian pyramid variables, with bounded convolution
workspace. See [the guide](docs/healpix_pyramid.md) and
[the runnable example](examples/healpix_pyramid_synthesis.py).


## Pre-commit hooks

This step is **required for anyone contributing to the development of the library**.

The pre-commit hooks automatically format the code before each commit according to the project's formatting rules. This ensures that code submitted through pull requests to the `main` development branch follows the required formatting standards.

To install the hooks locally, run the following command from the root of the repository:

```bash
pre-commit install
```

Once installed, the hooks will run automatically before each commit.

If the hooks modify your files, stage the changes and commit again:

```
git add .
git commit -m "your commit message"
```
