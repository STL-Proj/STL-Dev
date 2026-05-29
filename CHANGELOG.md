# Changelog

## [v1.0.0] - 2026-01-27
### Added
- Calculation of scattering and cross-channel statistics
  - Available in real space via the `STL_2D_Kernel_Torch` class
  - Available in Fourier space via the `STL_2D_FFT_Torch` class
  - Support for both periodic and non-periodic images (`Kernel` & `Torch`)
  - Handling of NaN values in real space (`Kernel`)
- Image synthesis 
- User example notebook demonstrating main features (only scattering stats computation, no synthesis)

### Fixed
- N/A (first stable release)

### Breaking Changes
- N/A (first stable release)


## [v1.1.0] - 2026-02-13
### Added
- Reduction of scattering cov coefficients (thanks Sebastien!)
- Power spectrum computation for the FFT class (works with both `pbc=True` and `pbc=False`)
- Refactor: factorization of `DataClass` into an abstract base data class
- Steerable bump wavelets in Fourier space

### Fixed
- Normalization fix in Fourier downgrade: ensures real-space means match downsampled maps

### Known limitations
- FFT handling for `pbc=False` is not fully supported yet


## [v1.2.0] - 2026-02-22
### Added
- Sihao reduced scattering statistics (one fewer convolution for S3 and S4). Enable this feature by setting the `has_fewer_convolutions` argument in the `st_operator` constructor.
- Support for `pbc=False` in FFT, regardless of the defined wavelet type (`gaussian`, `bump_steerable`, etc.).
- Power spectrum computation for the Kernel class (works with both `pbc=True` and `pbc=False`)
- Cleaned up user notebook for computing and comparing scattering coefficients.


## [v1.3.0] - 2026-03-31
### Added
- Added cross-spectrum calculation
- Added wavelet satisfying the Littlewood-Paley condition for cross-spectrum calculation (thanks Celia!)
- Completed the exhaustive user notebook for syntheses
- Added a user-friendly wrapper for synthesis from maps (see certification notebook for usage)
  - Can synthesize maps of different sizes using the `running_shape` argument
  - Can synthesize by averaging over the batch dimension using the `mean_field` argument
- Added a benchmark setup and an initial benchmark (Kernel vs Kernel_Sihao vs FFT vs FFT_Sihao).

### Fixed
- Added the `WType` argument to the constructor of the Wavelet operator in the `STL_2D_Kernel_Torch` dataclass. Currently available: Morlet
- Removed the `mean_ref` variable to avoid numerical instability


## [v1.4.0] - 2026-04-29
### Added
- Added the ability to perform synthesis from the statistics of a target map.
- Added and fully completed a user notebook demonstrating synthesis from statistics.
- Extended the `flatten` method to optionally preserve the batch dimension and to handle flattening of complex coefficients.

### Fixed
- Fixed incorrect synthesis behavior in mono-channel FFT and cross-channel FFT.


## [v1.5.0] - 2026-05-29
### Added
- Added user informative exceptions in the handling of the `norm` parameter of the scattering operator.
  
### Fixed
- Updated wavelet filters for both FFT & Kernel backends to improve Littlewood-Paley condition. Updated smoothing filter with better anti-aliasing propoerties for Kernel backend.
