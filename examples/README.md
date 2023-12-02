## Structure

For the simplest examples without using any helping code (see `framework.rs` here), check out:

- `hello` for printing adapter information
- `hello-triangle` for graphics and presentation
- `hello-compute` for pure computing

### Summary of examples

A summary of the basic examples as split along the graphics and compute "pathways" laid out roughly in order of building on each other. Those further indented, and thus more roughly dependent on more other examples, tend to be more complicated as well as those further down. It should be noted, though, that computing examples, even though they are mentioned further down (because rendering to a window is by far the most common use case), tend to be less complex as they require less surrounding context to create and manage a window to render to.

The rest of the examples are for demonstrating specific features that you can come back for later when you know what those features are.

#### General

- `hello` - Demonstrates the basics of the WGPU library by getting a default Adapter and debugging it to the screen

#### Graphics

- `hello-triangle` - Provides an example of a bare-bones WGPU workflow using the Winit crate that simply renders a red triangle on a green background.
- `uniform-values` - Demonstrates the basics of enabling shaders and the GPU, in general, to access app state through uniform variables. `uniform-values` also serves as an example of rudimentary app building as the app stores state and takes window-captured keyboard events. The app displays the Mandelbrot Set in grayscale (similar to `storage-texture`) but allows the user to navigate and explore it using their arrow keys and scroll wheel.
- `cube` - Introduces the user to slightly more advanced models. The example creates a set of triangles to form a cube on the CPU and then uses a vertex and index buffer to send the generated model to the GPU for usage in rendering. It also uses a texture generated on the CPU to shade the sides of the cube and a uniform variable to apply a transformation matrix to the cube in the shader.
- `bunnymark` - Demonstrates many things, but chief among them is performing numerous draw calls with different bind groups in one render pass. The example also uses textures for the icon and uniform buffers to transfer both global and per-particle states.
- `skybox` - Shows off too many concepts to list here. The name comes from game development where a "skybox" acts as a background for rendering, usually to add a sky texture for immersion, although they can also be used for backdrops to give the idea of a world beyond the game scene. This example does so much more than this, though, as it uses a car model loaded from a file and uses the user's mouse to rotate the car model in 3d. `skybox` also makes use of depth textures and similar app patterns to `uniform-values`.
- `shadow` - Likely by far the most complex example (certainly the largest in lines of code) of the official WGPU examples. `shadow` demonstrates basic scene rendering with the main attraction being lighting and shadows (as the name implies). It is recommended that any user looking into lighting be very familiar with the basic concepts of not only rendering with WGPU but also the primary mathematical ideas of computer graphics.
- `render-to-texture` - Renders to an image texture offscreen, demonstrating both off-screen rendering as well as how to add a sort of resolution-agnostic screenshot feature to an engine. This example either outputs an image file of your naming (pass command line arguments after specifying a `--` like `cargo run --bin render-to-texture -- "test.png"`) or adds an `img` element containing the image to the page in WASM.

#### Compute

- `hello-compute` - Demonstrates the basic workflow for getting arrays of numbers to the GPU, executing a shader on them, and getting the results back. The operation it performs is finding the Collatz value (how many iterations of the [Collatz equation](https://en.wikipedia.org/wiki/Collatz_conjecture) it takes for the number to either reach 1 or overflow) of a set of numbers and prints the results.
- `repeated-compute` - Mostly for going into detail on subjects `hello-compute` did not. It, too, computes the Collatz conjecture, but this time, it automatically loads large arrays of randomly generated numbers, prints them, runs them, and prints the result. It does this cycle 10 times.
- `hello-workgroups` - Teaches the user about the basics of compute workgroups; what they are and what they can do.
- `hello-synchronization` - Teaches the user about synchronization in WGSL, the ability to force all invocations in a workgroup to synchronize with each other before continuing via a sort of barrier.
- `storage-texture` - Demonstrates the use of storage textures as outputs to compute shaders. The example on the outside seems very similar to `render-to-texture` in that it outputs an image either to the file system or the web page, except displaying a grayscale render of the Mandelbrot Set. However, inside, the example dispatches a grid of compute workgroups, one for each pixel, which calculates the pixel value and stores it to the corresponding pixel of the output storage texture.

#### Combined

- `boids` - Demonstrates how to combine compute and render workflows by performing a [boid](https://en.wikipedia.org/wiki/Boids) simulation and rendering the boids to the screen as little triangles.

## Feature matrix

| Feature                      | boids  | bunnymark | conservative-raster | cube   | hello-synchronization | hello-workgroups | mipmap | msaa-line | render-to-texture | repeated-compute | shadow | skybox | stencil-triangles | storage-texture | texture-arrays | uniform-values | water  |
| ---------------------------- | ------ | --------- | ------------------- | ------ | --------------------- | ---------------- | ------ | --------- | ----------------- | ---------------- | ------ | ------ | ----------------- | --------------- | -------------- | -------------- | ------ |
| vertex attributes            | :star: |           |                     | :star: |                       |                  |        | :star:    |                   |                  | :star: | :star: |                   |                 | :star:         |                | :star: |
| instancing                   | :star: |           |                     |        |                       |                  |        |           |                   |                  |        |        |                   |                 |                |                |        |
| lines and points             |        |           | :star:              |        |                       |                  |        | :star:    |                   |                  |        |        |                   |                 |                |                |        |
| dynamic buffer offsets       |        | :star:    |                     |        |                       |                  |        |           |                   |                  | :star: |        |                   |                 |                |                |        |
| implicit layout              |        |           |                     |        |                       |                  | :star: |           |                   |                  |        |        |                   |                 |                |                |        |
| sampled color textures       | :star: | :star:    | :star:              | :star: |                       |                  | :star: |           |                   |                  |        | :star: |                   |                 | :star:         |                | :star: |
| storage textures             | :star: |           |                     |        |                       |                  |        |           |                   |                  |        |        |                   | :star:          |                |                |        |
| comparison samplers          |        |           |                     |        |                       |                  |        |           |                   |                  | :star: |        |                   |                 |                |                |        |
| subresource views            |        |           |                     |        |                       |                  | :star: |           |                   |                  | :star: |        |                   |                 |                |                |        |
| cubemaps                     |        |           |                     |        |                       |                  |        |           |                   |                  |        | :star: |                   |                 |                |                |        |
| multisampling                |        |           |                     |        |                       |                  |        | :star:    |                   |                  |        |        |                   |                 |                |                |        |
| off-screen rendering         |        |           | :star:              |        |                       |                  |        |           | :star:            |                  | :star: |        |                   |                 |                |                | :star: |
| stencil testing              |        |           |                     |        |                       |                  |        |           |                   |                  |        |        | :star:            |                 |                |                |        |
| depth testing                |        |           |                     |        |                       |                  |        |           |                   |                  | :star: | :star: |                   |                 |                |                | :star: |
| depth biasing                |        |           |                     |        |                       |                  |        |           |                   |                  | :star: |        |                   |                 |                |                |        |
| read-only depth              |        |           |                     |        |                       |                  |        |           |                   |                  |        |        |                   |                 |                |                | :star: |
| blending                     |        | :star:    |                     | :star: |                       |                  |        |           |                   |                  |        |        |                   |                 |                |                | :star: |
| render bundles               |        |           |                     |        |                       |                  |        | :star:    |                   |                  |        |        |                   |                 |                |                | :star: |
| uniform buffers              |        |           |                     |        |                       |                  |        |           |                   |                  |        |        |                   |                 |                | :star:         |        |
| compute passes               | :star: |           |                     |        | :star:                | :star:           |        |           |                   | :star:           |        |        |                   | :star:          |                |                |        |
| buffer mapping               |        |           |                     |        | :star:                | :star:           |        |           |                   | :star:           |        |        |                   | :star:          |                |                |        |
| error scopes                 |        |           |                     | :star: |                       |                  |        |           |                   |                  |        |        |                   |                 |                |                |        |
| compute workgroups           |        |           |                     |        | :star:                | :star:           |        |           |                   |                  |        |        |                   |                 |                |                |        |
| compute synchronization      |        |           |                     |        | :star:                |                  |        |           |                   |                  |        |        |                   |                 |                |                |        |
| _optional extensions_        |        |           |                     |        |                       |                  |        |           |                   |                  |        |        |                   |                 | :star:         |                |        |
| - SPIR-V shaders             |        |           |                     |        |                       |                  |        |           |                   |                  |        |        |                   |                 |                |                |        |
| - binding array              |        |           |                     |        |                       |                  |        |           |                   |                  |        |        |                   |                 | :star:         |                |        |
| - push constants             |        |           |                     |        |                       |                  |        |           |                   |                  |        |        |                   |                 |                |                |        |
| - depth clamping             |        |           |                     |        |                       |                  |        |           |                   |                  | :star: |        |                   |                 |                |                |        |
| - compressed textures        |        |           |                     |        |                       |                  |        |           |                   |                  |        | :star: |                   |                 |                |                |        |
| - polygon mode               |        |           |                     | :star: |                       |                  |        |           |                   |                  |        |        |                   |                 |                |                |        |
| - queries                    |        |           |                     |        |                       |                  | :star: |           |                   |                  |        |        |                   |                 |                |                |        |
| - conservative rasterization |        |           | :star:              |        |                       |                  |        |           |                   |                  |        |        |                   |                 |                |                |        |
| _integrations_               |        |           |                     |        |                       |                  |        |           |                   |                  |        |        |                   |                 |                |                |        |
| - staging belt               |        |           |                     |        |                       |                  |        |           |                   |                  |        | :star: |                   |                 |                |                |        |
| - typed arena                |        |           |                     |        |                       |                  |        |           |                   |                  |        |        |                   |                 |                |                |        |
| - obj loading                |        |           |                     |        |                       |                  |        |           |                   |                  |        | :star: |                   |                 |                |                |        |


## Additional notes

Note that the examples regarding computing build off of each other; repeated-compute extends hello-compute, hello-workgroups assumes you know the basic workflow of GPU computation, and hello-synchronization assumes you know what a workgroup is. Also, note that the computing examples cannot be downleveled to WebGL as WebGL does not allow storage textures. Running these in a browser will require that browser to support WebGPU.

All the examples use [WGSL](https://gpuweb.github.io/gpuweb/wgsl.html) shaders unless specified otherwise.

All framework-based examples render to the window and are reftested against the screenshot in the directory.

## Hacking

You can record an API trace for any of the framework-based examples by starting them as:

```sh
mkdir -p trace && WGPU_TRACE=trace cargo run --features trace --bin wgpu-examples <example-name>
```
