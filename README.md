# Naga

[![Matrix](https://img.shields.io/badge/Matrix-%23naga%3Amatrix.org-blueviolet.svg)](https://matrix.to/#/#naga:matrix.org)
[![Crates.io](https://img.shields.io/crates/v/naga.svg?label=naga)](https://crates.io/crates/naga)
[![Docs.rs](https://docs.rs/naga/badge.svg)](https://docs.rs/naga)
[![Build Status](https://github.com/gfx-rs/naga/workflows/pipeline/badge.svg)](https://github.com/gfx-rs/naga/actions)

This is an experimental shader translation library for the needs of gfx-rs project and WebGPU.

## Supported end-points

Front-end       |       Status       | Notes |
--------------- | ------------------ | ----- |
SPIR-V (binary) | :construction:     |       |
WGSL            | :construction:     |       |
GLSL (Vulkan)   | :construction:     |       |
Rust            |                    |       |

Back-end        |       Status       | Notes |
--------------- | ------------------ | ----- |
SPIR-V (binary) | :construction:     |       |
WGSL            |                    |       |
Metal           | :construction:     |       |
HLSL            |                    |       |
GLSL            | :construction:     |       |
AIR             |                    |       |
DXIR            |                    |       |
DXIL            |                    |       |
DXBC            |                    |       |

:heavy_check_mark: = Primary support — :white_check_mark: = Secondary support — :construction: = Unsupported, but support in progress
