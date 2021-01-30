# Naga

[![Matrix](https://img.shields.io/badge/Matrix-%23naga%3Amatrix.org-blueviolet.svg)](https://matrix.to/#/#naga:matrix.org)
[![Crates.io](https://img.shields.io/crates/v/naga.svg?label=naga)](https://crates.io/crates/naga)
[![Docs.rs](https://docs.rs/naga/badge.svg)](https://docs.rs/naga)
[![Build Status](https://github.com/gfx-rs/naga/workflows/pipeline/badge.svg)](https://github.com/gfx-rs/naga/actions)

This is an experimental shader translation library for the needs of gfx-rs project and WebGPU.

## Supported end-points

Everything is still a work-in-progress, but some end-points are usable:

Front-end       |       Status       | Notes |
--------------- | ------------------ | ----- |
SPIR-V (binary) | :white_check_mark: |       |
WGSL            | :white_check_mark: |       |
GLSL (Vulkan)   | :ok:               |       |
Rust            |                    |       |

Back-end        |       Status       | Notes |
--------------- | ------------------ | ----- |
SPIR-V (binary) | :white_check_mark: |       |
WGSL            |                    |       |
Metal           | :white_check_mark: |       |
HLSL            |                    |       |
GLSL            | :ok:               |       |
AIR             |                    |       |
DXIL/DXIR       |                    |       |
DXBC            |                    |       |

:white_check_mark: = Primary support — :ok: = Secondary support — :construction: = Unsupported, but support in progress
