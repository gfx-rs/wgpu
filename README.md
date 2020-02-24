# Naga

[![Matrix](https://img.shields.io/badge/Matrix-%23gfx%3Amatrix.org-blueviolet.svg)](https://matrix.to/#/#gfx:matrix.org)
[![Build Status](https://travis-ci.org/gfx-rs/naga.svg?branch=master)](https://travis-ci.org/gfx-rs/naga)

This is an experimental shader translation library for the needs of gfx-rs project and WebGPU. It's meant to provide a safe and performant way of converting to and from SPIR-V.

## Supported end-points

Front-end       |       Status       | Notes |
--------------- | ------------------ | ----- |
SPIR-V (binary) | :construction:     |       |
Tint            |                    |       |
GLSL (Vulkan)   |                    |       |
Rust            |                    |       |

Back-end        |       Status       | Notes |
--------------- | ------------------ | ----- |
SPIR-V (binary) |                    |       |
Tint            |                    |       |
MSL             |                    |       |
HLSL            |                    |       |
GLSL            |                    |       |
AIR             |                    |       |
DXBC            |                    |       |
DXIL            |                    |       |
