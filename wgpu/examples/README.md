## Structure

For the simplest examples without using any helping code (see `framework.rs` here), check out:
  - `hello ` for printing adapter information
  - `hello-triangle` for graphics and presentation
  - `hello-compute` for pure computing

Notably, `capture` example shows rendering without a surface/window. It reads back the contents and saves them to a file.

All the examples use [WGSL](https://gpuweb.github.io/gpuweb/wgsl.html) shaders unless specified otherwise.

All framework-based examples render to the window.

## Feature matrix
| Feature                      | boids  | bunnymark | cube   | mipmap | msaa-line | shadow | skybox | texture-arrays | water  | conservative-raster |
| ---------------------------- | ------ | --------- | ------ | ------ | --------- | ------ | ------ | -------------- | ------ | ------------------- |
| vertex attributes            | :star: |           | :star: |        | :star:    | :star: | :star: | :star:         | :star: |                     |
| instancing                   | :star: |           |        |        |           |        |        |                |        |                     |
| lines and points             |        |           |        |        | :star:    |        |        |                |        | :star:              |
| dynamic buffer offsets       |        | :star:    |        |        |           | :star: |        |                |        |                     |
| implicit layout              |        |           |        | :star: |           |        |        |                |        |                     |
| sampled color textures       | :star: | :star:    | :star: | :star: |           |        | :star: | :star:         | :star: | :star:              |
| storage textures             | :star: |           |        |        |           |        |        |                |        |                     |
| binding array                |        |           |        |        |           |        |        | :star:         |        |                     |
| comparison samplers          |        |           |        |        |           | :star: |        |                |        |                     |
| subresource views            |        |           |        | :star: |           | :star: |        |                |        |                     |
| cubemaps                     |        |           |        |        |           |        | :star: |                |        |                     |
| multisampling                |        |           |        |        | :star:    |        |        |                |        |                     |
| off-screen rendering         |        |           |        |        |           | :star: |        |                | :star: | :star:              |
| stencil testing              |        |           |        |        |           |        |        |                |        |                     |
| depth testing                |        |           |        |        |           | :star: | :star: |                | :star: |                     |
| depth biasing                |        |           |        |        |           | :star: |        |                |        |                     |
| read-only depth              |        |           |        |        |           |        |        |                | :star: |                     |
| blending                     |        | :star:    | :star: |        |           |        |        |                | :star: |                     |
| render bundles               |        |           |        |        | :star:    |        |        |                | :star: |                     |
| compute passes               | :star: |           |        |        |           |        |        |                |        |                     |
| *optional extensions*        |        |           |        |        |           |        |        | :star:         |        |                     |
| - SPIR-V shaders             |        |           |        |        |           |        |        | :star:         |        |                     |
| - binding indexing           |        |           |        |        |           |        |        | :star:         |        |                     |
| - push constants             |        |           |        |        |           |        |        | :star:         |        |                     |
| - depth clamping             |        |           |        |        |           | :star: |        |                |        |                     |
| - compressed textures        |        |           |        |        |           |        | :star: |                |        |                     |
| - polygon mode               |        |           | :star: |        |           |        |        |                |        |                     |
| - queries                    |        |           |        | :star: |           |        |        |                |        |                     |
| - conservative rasterization |        |           |        |        |           |        |        |                |        | :star:              |
| *integrations*               |        |           |        |        |           |        |        |                |        |                     |
| - staging belt               |        |           |        |        |           |        |        |                |        |                     |
| - typed arena                |        |           |        |        |           |        |        |                |        |                     |
| - obj loading                |        |           |        |        |           |        | :star: |                |        |                     |

## Hacking

You can record an API trace any of the framework-based examples by starting them as:
```sh
mkdir -p trace && WGPU_TRACE=trace cargo run --features trace --example <example-name>
```
