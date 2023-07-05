## Structure

For the simplest examples without using any helping code (see `framework.rs` here), check out:

- `hello` for printing adapter information
- `hello-triangle` for graphics and presentation
- `hello-compute` for pure computing

Notably, `capture` example shows rendering without a surface/window. It reads back the contents and saves them to a file.

All the examples use [WGSL](https://gpuweb.github.io/gpuweb/wgsl.html) shaders unless specified otherwise.

All framework-based examples render to the window and are reftested against the screenshot in the directory.

## Feature matrix

| Feature                      | boids  | bunnymark | cube   | mipmap | msaa-line | shadow | skybox | texture-arrays | water  | conservative-raster | stencil-triangles |
| ---------------------------- | ------ | --------- | ------ | ------ | --------- | ------ | ------ | -------------- | ------ | ------------------- | ----------------- |
| vertex attributes            | :star: |           | :star: |        | :star:    | :star: | :star: | :star:         | :star: |                     |                   |
| instancing                   | :star: |           |        |        |           |        |        |                |        |                     |                   |
| lines and points             |        |           |        |        | :star:    |        |        |                |        | :star:              |                   |
| dynamic buffer offsets       |        | :star:    |        |        |           | :star: |        |                |        |                     |                   |
| implicit layout              |        |           |        | :star: |           |        |        |                |        |                     |                   |
| sampled color textures       | :star: | :star:    | :star: | :star: |           |        | :star: | :star:         | :star: | :star:              |                   |
| storage textures             | :star: |           |        |        |           |        |        |                |        |                     |                   |
| comparison samplers          |        |           |        |        |           | :star: |        |                |        |                     |                   |
| subresource views            |        |           |        | :star: |           | :star: |        |                |        |                     |                   |
| cubemaps                     |        |           |        |        |           |        | :star: |                |        |                     |                   |
| multisampling                |        |           |        |        | :star:    |        |        |                |        |                     |                   |
| off-screen rendering         |        |           |        |        |           | :star: |        |                | :star: | :star:              |                   |
| stencil testing              |        |           |        |        |           |        |        |                |        |                     | :star:            |
| depth testing                |        |           |        |        |           | :star: | :star: |                | :star: |                     |                   |
| depth biasing                |        |           |        |        |           | :star: |        |                |        |                     |                   |
| read-only depth              |        |           |        |        |           |        |        |                | :star: |                     |                   |
| blending                     |        | :star:    | :star: |        |           |        |        |                | :star: |                     |                   |
| render bundles               |        |           |        |        | :star:    |        |        |                | :star: |                     |                   |
| compute passes               | :star: |           |        |        |           |        |        |                |        |                     |                   |
| error scopes                 |        |           | :star: |        |           |        |        |                |        |                     |                   |
| _optional extensions_        |        |           |        |        |           |        |        | :star:         |        |                     |                   |
| - SPIR-V shaders             |        |           |        |        |           |        |        |                |        |                     |                   |
| - binding array              |        |           |        |        |           |        |        | :star:         |        |                     |                   |
| - push constants             |        |           |        |        |           |        |        |                |        |                     |                   |
| - depth clamping             |        |           |        |        |           | :star: |        |                |        |                     |                   |
| - compressed textures        |        |           |        |        |           |        | :star: |                |        |                     |                   |
| - polygon mode               |        |           | :star: |        |           |        |        |                |        |                     |                   |
| - queries                    |        |           |        | :star: |           |        |        |                |        |                     |                   |
| - conservative rasterization |        |           |        |        |           |        |        |                |        | :star:              |                   |
| _integrations_               |        |           |        |        |           |        |        |                |        |                     |                   |
| - staging belt               |        |           |        |        |           |        | :star: |                |        |                     |                   |
| - typed arena                |        |           |        |        |           |        |        |                |        |                     |                   |
| - obj loading                |        |           |        |        |           |        | :star: |                |        |                     |                   |

## Hacking

You can record an API trace any of the framework-based examples by starting them as:

```sh
mkdir -p trace && WGPU_TRACE=trace cargo run --features trace --bin <example-name>
```
