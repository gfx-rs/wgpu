use std::marker::PhantomData;

use custom::Counter;
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
            .request_device(
                &DeviceDescriptor {
                    label: Some("device"),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
        assert_eq!(counter.count(), 5);

        let _module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Dummy(PhantomData),
        });

        assert_eq!(counter.count(), 6);
    }
    assert_eq!(counter.count(), 1);
}
