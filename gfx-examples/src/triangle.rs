extern crate wgpu;

#[path="framework.rs"]
mod fw;

struct Triangle;

impl fw::Example for Triangle {
	fn init(device: &wgpu::Device) -> Self {
		Triangle
	}

	fn update(&mut self, _event: fw::winit::WindowEvent) {
	}

	fn render(&mut self, frame: &wgpu::SwapChainOutput, device: &wgpu::Device) {
	}
}

fn main() {
	fw::run::<Triangle>();
}
