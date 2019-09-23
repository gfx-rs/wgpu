/// This example shows how to describe the adapter in use.
fn main() {
    env_logger::init();

    let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::Default,
        backends: wgpu::BackendBit::PRIMARY,
    }).unwrap();

    println!("{:?}", adapter.get_info())
}
