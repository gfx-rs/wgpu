/*!
# Running on the Web (WebGPU and WebGL)

`wgpu` can run in the browser by compiling to WebAssembly, targeting either
WebGPU (where available) or the WebGL2 backend as a fallback. The WebGL2
backend is still missing some features compared to the native and WebGPU
backends.

## Running the examples

### Installing the Rust WebAssembly target

To build the `wgpu` examples for execution in a browser, you must first
install the Rust toolchain for the `wasm32-unknown-unknown` target. Using
`rustup`:

```bash
rustup target add wasm32-unknown-unknown
```

### Using `cargo xtask run-wasm`

The simplest way to run the examples on the web is the `run-wasm` xtask:

```bash
cargo xtask run-wasm
```

This builds the `wgpu-examples` crate for `wasm32-unknown-unknown` (both the
WebGPU and WebGL2 variants), runs `wasm-bindgen` on the output, and serves
the result. It requires
[`wasm-bindgen-cli`](https://crates.io/crates/wasm-bindgen-cli) and
[`simple-http-server`](https://crates.io/crates/simple-http-server) to be
installed. Once it's running, open <http://127.0.0.1:8000> in your browser
and pick an example.

> **Note:** the server binds to `127.0.0.1` on purpose. WebGPU requires a
> [secure context](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts),
> which `127.0.0.1` satisfies but `0.0.0.0` does not.

### WebGPU browser support

WebGPU is available in current versions of Chrome and other Chromium-based
browsers, and is shipping or in progress elsewhere. For up-to-date
implementation status, check [webgpu.io](https://webgpu.io) or
[caniuse.com/webgpu](https://caniuse.com/webgpu). Note that `wgpu` is often
ahead of browsers in catching up with upstream WebGPU API changes.

## Manual compilation with `wasm-bindgen-cli`

If you'd rather not use the xtask, you can reproduce what it does by hand.
First install the version of `wasm-bindgen-cli` that matches the version
used by `wgpu` (check the workspace `Cargo.lock`):

```bash
cargo install -f wasm-bindgen-cli --version <matching version>
```

Then build the examples for `wasm32-unknown-unknown` and run `wasm-bindgen`
on the output. For WebGPU:

```bash
cargo build --target wasm32-unknown-unknown -p wgpu-examples --no-default-features --features webgpu
wasm-bindgen target/wasm32-unknown-unknown/debug/wgpu-examples.wasm \
    --target web --no-typescript --out-dir target/generated --out-name webgpu
```

For WebGL2, swap the `webgpu` feature for `webgl` and the `--out-name`
accordingly:

```bash
cargo build --target wasm32-unknown-unknown -p wgpu-examples --no-default-features --features webgl
wasm-bindgen target/wasm32-unknown-unknown/debug/wgpu-examples.wasm \
    --target web --no-typescript --out-dir target/generated --out-name webgl2
```

### Setting up the page

Create an `index.html` file in the `target/generated` directory that loads
the generated module:

```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  </head>
  <body>
    <script type="module">
      import init from "./webgpu.js"; // or "./webgl2.js"
      init();
    </script>
  </body>
</html>
```

### Running the code

Now run a web server locally inside the `target/generated` directory to view
the example in the browser. A secure context is required for WebGPU, so
serve from `127.0.0.1` (or `localhost`), for example with
[`simple-http-server`](https://crates.io/crates/simple-http-server):

```bash
simple-http-server target/generated -c wasm,html,js -i --coep --coop --ip 127.0.0.1
```

The `--coep` and `--coop` flags set the cross-origin isolation headers some
features require.
*/
