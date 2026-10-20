/*!
# Running on the Web with Emscripten

At the moment `wgpu` has one example meant to run under Emscripten:
`wgpu-hal/examples/raw-gles.rs`.

In the workspace's top directory, build the example as follows:

```bash
cargo clean
EMCC_CFLAGS="-g -s ERROR_ON_UNDEFINED_SYMBOLS=0 --no-entry -s FULL_ES3=1" \
    cargo build --example raw-gles --target wasm32-unknown-emscripten --features gles,emscripten,webgl
```

Then copy the HTML page that loads the compiled example into the target
directory alongside the WebAssembly:

```bash
cp wgpu-hal/examples/raw-gles.em.html target/wasm32-unknown-emscripten/debug/examples/
```

Finally, start an HTTP server publishing the contents of that directory:

```bash
cd target/wasm32-unknown-emscripten/debug/examples/
python -m http.server
```

Then visit `localhost:8000/` in your browser.

If you get other, more interesting examples working with Emscripten, please
update this page!
*/
