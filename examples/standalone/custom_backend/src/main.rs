use std::marker::PhantomData;

use custom::{Counter, CustomShaderModule};
use wgpu::{DeviceDescriptor, RequestAdapterOptions};

mod custom;

#[pollster::main]
async fn main() {
    let counter = Counter::new();
    {
        let custom_instance = custom::CustomInstance(counter.clone());
        // wrap custom instance into wgpu abstraction
        let instance = wgpu::Instance::from_custom(custom_instance);
        assert_eq!(counter.count(), 2);
        // do work on instance (usually by passing it to other libs)

        // here we will simulate a library and ensure that counter is incremented
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .unwrap();
        assert_eq!(counter.count(), 3);

        let (device, _queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("device"),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(counter.count(), 5);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Dummy(PhantomData),
        });

        let custom_module = module.as_custom::<CustomShaderModule>().unwrap();
        assert_eq!(custom_module.0.count(), 6);
        let _module_clone = module.clone();
        assert_eq!(counter.count(), 6);

        let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });

        assert_eq!(counter.count(), 7);
    }
    assert_eq!(counter.count(), 1);
}
