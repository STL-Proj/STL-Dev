# STL (temporary package name)

Welcome to the Scattering Transform Library!

Doc: https://stl-dev.readthedocs.io/en/latest/


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