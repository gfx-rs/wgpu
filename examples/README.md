> [!NOTE]  
> These are the examples for the development version of wgpu. If you want to see the examples for the latest crates.io release
> of wgpu, go to the [latest release branch](https://github.com/gfx-rs/wgpu/tree/v26/examples#readme).

# Examples

If you are just starting your graphics programming journey entirely, we recommend going through [Learn-WGPU](https://sotrh.github.io/learn-wgpu/)
for a mode guided tutorial, which will also teach you the basics of graphics programming.

## Standalone Examples

All the standalone examples are separate crates and include all boilerplate inside the example itself. They can
be cloned out of the repository to serve as a starting point for your own projects and are fully commented.

| Name   | Description | Platforms |
|--------|-------------|-----------|
| ---    | Introductory Examples | --- |
| [1. hello compute](standalone/01_hello_compute/) | Simplest example and shows how to run a compute shader on a given set of input data and get the results back. | Native-Only |
| [2. hello window](standalone/02_hello_window/) | Shows how to create a window and render into it. | Native-Only |
| --- | Special Examples | --- |
| [custom backend](standalone/custom_backend/) | Shows how to implement and use custom wgpu context | All |

You can also use [`cargo-generate`](https://github.com/cargo-generate/cargo-generate) to easily use these as a basis for your own projects.

```sh
cargo generate gfx-rs/wgpu --branch v26
```

## Framework Examples

These examples use a common framework to handle wgpu init, window creation, and event handling. This allows the example to focus on the unique code in the example itself. Refer to the standalone examples for a more detailed look at the boilerplate code.

#### Graphics

- `hello_triangle` - Provides an example of a bare-bones WGPU workflow using the Winit crate that simply renders a red triangle on a green background.
- `uniform_values` - Demonstrates the basics of enabling shaders and the GPU, in general, to access app state through uniform variables. `uniform_values` also serves as an example of rudimentary app building as the app stores state and takes window-captured keyboard events. The app displays the Mandelbrot Set in grayscale (similar to `storage_texture`) but allows the user to navigate and explore it using their arrow keys and scroll wheel.
- `cube` - Introduces the user to slightly more advanced models. The example creates a set of triangles to form a cube on the CPU and then uses a vertex and index buffer to send the generated model to the GPU for usage in rendering. It also uses a texture generated on the CPU to shade the sides of the cube and a uniform variable to apply a transformation matrix to the cube in the shader.
- `bunnymark` - Demonstrates many things, but chief among them is performing numerous draw calls with different bind groups in one render pass. The example also uses textures for the icon and uniform buffers to transfer both global and per-particle states.
- `skybox` - Shows off too many concepts to list here. The name comes from game development where a "skybox" acts as a background for rendering, usually to add a sky texture for immersion, although they can also be used for backdrops to give the idea of a world beyond the game scene. This example does so much more than this, though, as it uses a car model loaded from a file and uses the user's mouse to rotate the car model in 3d. `skybox` also makes use of depth textures and similar app patterns to `uniform_values`.
- `shadow` - Likely by far the most complex example (certainly the largest in lines of code) of the official WGPU examples. `shadow` demonstrates basic scene rendering with the main attraction being lighting and shadows (as the name implies). It is recommended that any user looking into lighting be very familiar with the basic concepts of not only rendering with WGPU but also the primary mathematical ideas of computer graphics.
- `multiple-render-targets` - Demonstrates how to render to two texture targets simultaneously from fragment shader.
- `render_to_texture` - Renders to an image texture offscreen, demonstrating both off-screen rendering as well as how to add a sort of resolution-agnostic screenshot feature to an engine. This example either outputs an image file of your naming (pass command line arguments after specifying a `--` like `cargo run --bin wgpu-examples -- render_to_texture "test.png"`) or adds an `img` element containing the image to the page in WASM.
- `ray_cube_fragment` - Demonstrates using ray queries with a fragment shader.
- `ray_scene` - Demonstrates using ray queries and model loading
- `ray_shadows` - Demonstrates a simple use of ray queries - high quality shadows - uses a light set with push constants to raytrace through an untransformed scene and detect whether there is something obstructing the light.
- `mesh_shader` - Rrenders a triangle to a window with mesh shaders, while showcasing most mesh shader related features(task shaders, payloads, per primitive data).

#### Compute

- `hello_compute` - Demonstrates the basic workflow for getting arrays of numbers to the GPU, executing a shader on them, and getting the results back. The operation it performs is finding the Collatz value (how many iterations of the [Collatz equation](https://en.wikipedia.org/wiki/Collatz_conjecture) it takes for the number to either reach 1 or overflow) of a set of numbers and prints the results.
- `repeated_compute` - Mostly for going into detail on subjects `hello-compute` did not. It, too, computes the Collatz conjecture, but this time, it automatically loads large arrays of randomly generated numbers, prints them, runs them, and prints the result. It does this cycle 10 times.
- `hello_workgroups` - Teaches the user about the basics of compute workgroups; what they are and what they can do.
- `hello_synchronization` - Teaches the user about synchronization in WGSL, the ability to force all invocations in a workgroup to synchronize with each other before continuing via a sort of barrier.
- `storage_texture` - Demonstrates the use of storage textures as outputs to compute shaders. The example on the outside seems very similar to `render_to_texture` in that it outputs an image either to the file system or the web page, except displaying a grayscale render of the Mandelbrot Set. However, inside, the example dispatches a grid of compute workgroups, one for each pixel, which calculates the pixel value and stores it to the corresponding pixel of the output storage texture. This example either outputs an image file of your naming (pass command line arguments after specifying a `--` like `cargo run --bin wgpu-examples -- storage_texture "test.png"`) or adds an `img` element containing the image to the page in WASM.
- `big_compute_buffers` - Demonstrates how you can split _large_ datasets across multiple buffers, using `binding_array` in your `wgsl` [NOTE: native only, no WASM support].

#### Combined

- `boids` - Demonstrates how to combine compute and render workflows by performing a [boid](https://en.wikipedia.org/wiki/Boids) simulation and rendering the boids to the screen as little triangles.
- `ray_cube_compute` - Demonstrates using ray queries with a compute shader.
- `ray_traced_triangle` - A simpler example demonstrating using ray queries with a compute shader

## Running on the Web

To run the examples in a browser, run `cargo xtask run-wasm`.
Then open `http://localhost:8000` in your browser, and you can choose an example to run.
Naturally, in order to display any of the WebGPU based examples, you need to make sure your browser supports it.
