export async function start() {
  let url = new URL(window.location.href);
  let name = url.searchParams.get("name");
  let wasm = url.searchParams.get("wasm");

  let wasm_paths = await (await fetch("./wasm_paths.json")).json();
  let wasm_script = wasm_paths[wasm];

  if (wasm_script == null) {
    throw new Error("can't find wasm file");
  }

  const { default: init, run_test, run_gpu_report } = await import(wasm_script);

  init().then(async () => {
    if (name == null) {
      await run_gpu_report();
    } else {
      await run_test(name);
    }
  });
}
