# Changelog

All notable changes to **PaleoSTeHM Dashboard** will be documented in this file.

## [0.1.0] - 2026-11-24

### Added
- Guided multi-step Streamlit dashboard (`python -m stgp_dashboard`) for Holocene relative sea-level (RSL) spatiotemporal Gaussian Process (STGP) modeling.
- Modular `stgp_pipeline` package supporting no-GIA, 1D GIA (ICE6G `.mat` and ANU-ICE `.txt`), and combined 1D/3D GIA workflows, sea-level budget decomposition, NetCDF/CSV exports, figure generation, and sensitivity suites.
- `gia_covariance_tools` package for 1D/3D GIA ensemble loading and WRSS-weighted covariance estimation.
- Open-access US Atlantic Coast test dataset (`Data/US_Atlantic_Coast_for_ESTGP.csv`) and ready-to-run smoke-test configuration (`configs/example_us_atlantic.yaml`).
- Cross-platform `conda-forge` environment specification (`environment.yml`), automated pytest test suites, GitHub Actions CI, and Zenodo/CFF citation metadata.
