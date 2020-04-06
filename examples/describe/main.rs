/// This example shows how to describe the adapter in use.
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();
    futures::executor::block_on(run());
}

#[cfg(not(target_arch = "wasm32"))]
async fn run() {
    let adapter = wgpu::Adapter::request(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
            compatible_surface: None,
        },
        wgpu::BackendBit::PRIMARY,
    )
    .await
    .unwrap();

    println!("{:?}", adapter.get_info())
}
