/// This example shows how to describe the adapter in use.
async fn run() {
    #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
    let adapter = wgpu::Instance::new(wgpu::BackendBit::PRIMARY)
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    println!("{:?}", adapter.get_info())
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }
}
