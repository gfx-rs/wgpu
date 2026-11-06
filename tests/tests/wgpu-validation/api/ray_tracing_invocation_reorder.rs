use wgpu::*;
use wgpu_test::{fail, valid};

const HIT_OBJECT_SHADER: &str = "
enable wgpu_ray_tracing_pipeline;
enable wgpu_ray_tracing_invocation_reorder;

struct Payload {
    color: vec3<f32>,
}

var<ray_payload> payload: Payload;

@group(0) @binding(0)
var acc_struct: acceleration_structure;

@ray_generation
fn ray_gen_main() {
    var ho: hit_object;
    let desc = RayDesc(RAY_FLAG_NONE, 0xffu, 0.01, 100.0, vec3(0.0), vec3(0.0, 1.0, 0.0));

    hitObjectTraceRay(&ho, acc_struct, desc, &payload);
    reorderThread(&ho);
    hitObjectExecuteShader(&ho, &payload);
}
";

fn device_with(features: Features) -> Device {
    let (device, _queue) = Device::noop(&DeviceDescriptor {
        required_features: features,
        // SAFETY: the noop backend does not execute anything.
        experimental_features: unsafe { ExperimentalFeatures::enabled() },
        ..Default::default()
    });
    device
}

fn create_hit_object_shader(device: &Device) -> ShaderModule {
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some("hit object shader"),
        source: ShaderSource::Wgsl(HIT_OBJECT_SHADER.into()),
    })
}

#[test]
fn noop_adapter_does_not_reorder() {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::NOOP,
        backend_options: BackendOptions {
            noop: NoopBackendOptions::enabled(),
            ..Default::default()
        },
        ..InstanceDescriptor::new_without_display_handle()
    });
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions::default()))
        .expect("noop backend adapter absent when it should be");

    assert_eq!(
        adapter.ray_tracing_invocation_reorder_mode(),
        RayTracingInvocationReorderMode::None
    );
}

#[test]
fn hit_objects_require_invocation_reorder_feature() {
    let device = device_with(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES);

    fail(
        &device,
        || {
            let module = create_hit_object_shader(&device);
            let info = pollster::block_on(module.get_compilation_info());
            assert!(info.messages[0].message.contains(
                "the `wgpu_ray_tracing_invocation_reorder` extension is not supported in the \
                 current environment"
            ));
        },
        Some("Shader 'hit object shader' parsing error"),
    );
}

#[test]
fn hit_objects_with_invocation_reorder_feature() {
    let device = device_with(
        Features::EXPERIMENTAL_RAY_TRACING_PIPELINES
            | Features::EXPERIMENTAL_RAY_TRACING_INVOCATION_REORDER,
    );

    valid(&device, || create_hit_object_shader(&device));
}
