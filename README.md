# Naga

[![Matrix](https://img.shields.io/badge/Matrix-%23naga%3Amatrix.org-blueviolet.svg)](https://matrix.to/#/#naga:matrix.org)
[![Crates.io](https://img.shields.io/crates/v/naga.svg?label=naga)](https://crates.io/crates/naga)
[![Docs.rs](https://docs.rs/naga/badge.svg)](https://docs.rs/naga)
[![Build Status](https://travis-ci.org/gfx-rs/naga.svg?branch=master)](https://travis-ci.org/gfx-rs/naga)

This is an experimental shader translation library for the needs of gfx-rs project and WebGPU. It's meant to provide a safe and performant way of converting to and from SPIR-V.

## Supported end-points

Front-end       |       Status       | Notes |
--------------- | ------------------ | ----- |
SPIR-V (binary) | :construction:     |       |
WGSL (Tint)     | :construction:     |       |
GLSL (Vulkan)   |                    |       |
Rust            |                    |       |

Back-end        |       Status       | Notes |
--------------- | ------------------ | ----- |
SPIR-V (binary) |                    |       |
WGSL            |                    |       |
Metal           | :construction:     |       |
HLSL            |                    |       |
GLSL            |                    |       |
AIR             |                    |       |
DXIR            |                    |       |
DXIL            |                    |       |
DXBC            |                    |       |
