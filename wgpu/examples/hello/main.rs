/// This example shows how to describe the adapter in use.
async fn run() {
    #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
    let adapter = {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        #[cfg(not(target_arch = "wasm32"))]
        {
            println!("Available adapters:");
            for a in instance.enumerate_adapters(wgpu::Backends::all()) {
                println!("    {:?}", a.get_info())
            }
        }
        instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap()
    };

    #[cfg(not(target_arch = "wasm32"))]
    println!("Selected adapter: {:?}", adapter.get_info())
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
