/// This example shows how to describe the adapter in use.
fn main() {
    env_logger::init();
    futures::executor::block_on(run());
}

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
