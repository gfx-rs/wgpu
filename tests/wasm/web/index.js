export async function start() {
  let url = new URL(window.location.href);
  let name = url.searchParams.get("name");
  let wasm = url.searchParams.get("wasm");
  const { default: init, run_test, gpu_report } = await import(`./${wasm}.js`);

  init().then(async () => {
    window.gpu_report = gpu_report;


    if (name != null) {
      await run_test(name)
    }
  });
}
