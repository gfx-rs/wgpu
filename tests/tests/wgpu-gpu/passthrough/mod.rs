use std::{
    env::temp_dir,
    hash::{DefaultHasher, Hash, Hasher},
    process::Command,
};

use wgpu::{
    Backends, ColorTargetState, ColorWrites, Features, FragmentState, MultisampleState,
    PipelineLayoutDescriptor, RenderPipelineDescriptor, VertexState,
};
use wgpu_test::{
    gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};
use wgpu_types::CreateShaderModuleDescriptorPassthrough;

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.push(METAL_PASSTHROUGH_SHADER);
    tests.push(METALLIB_PASSTHROUGH_SHADER);
}

fn metal_test(ctx: TestingContext) {
    let module = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                label: None,
                num_workgroups: (0, 0, 0),
                msl: Some(std::borrow::Cow::Borrowed(METAL_SOURCE)),
                ..Default::default()
            })
    };
    let layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });
    let _pipeline = ctx
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWrites::all(),
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
}

const METAL_SOURCE: &str = include_str!("shader.metal");

#[gpu_test]
static METAL_PASSTHROUGH_SHADER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::EXPERIMENTAL_PASSTHROUGH_SHADERS)
            .skip(FailureCase::backend(Backends::all() - Backends::METAL)),
    )
    .run_sync(metal_test);

fn metallib_test(ctx: TestingContext) {
    let metal_compiler = Command::new("xcrun")
        .args(["--find", "metal"])
        .status()
        .is_ok_and(|a| a.success());
    let metallib_linker = Command::new("xcrun")
        .args(["--find", "metallib"])
        .status()
        .is_ok_and(|a| a.success());
    if !metal_compiler || !metallib_linker {
        panic!("Metal compiler or metallib linker not present. Most users can safely ignore this.");
    }
    let mut hasher = DefaultHasher::new();
    ctx.hash(&mut hasher);
    let result = hasher.finish();

    let dir = format!("{}/{result}", temp_dir().display());
    let input_name = format!("{dir}/input.metal");
    let air_name = format!("{dir}/intermediate.air");
    let output_name = format!("{dir}/output.metallib");
    println!("Attempting to create dir {dir}");
    std::fs::create_dir(&dir).unwrap();

    std::fs::write(&input_name, METAL_SOURCE).unwrap();
    {
        let output = Command::new("xcrun")
            .args(["metal", "-c", &input_name, "-o", &air_name])
            .output()
            .unwrap();
        if !output.status.success() {
            panic!(
                "Failed to compile .metal into .air: {}",
                String::from_utf8(output.stderr).unwrap()
            );
        }
    }
    {
        let output = Command::new("xcrun")
            .args(["metallib", &air_name, "-o", &output_name])
            .output()
            .unwrap();
        if !output.status.success() {
            panic!(
                "Failed to compile .air into .metallib: {}",
                String::from_utf8(output.stderr).unwrap()
            );
        }
    }
    let source = std::fs::read(&output_name).unwrap();
    let vertex = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                label: None,
                num_workgroups: (0, 0, 0),
                metallib: Some(std::borrow::Cow::Borrowed(&source)),
                ..Default::default()
            })
    };
    let fragment = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                label: None,
                num_workgroups: (0, 0, 0),
                metallib: Some(std::borrow::Cow::Borrowed(&source)),
                ..Default::default()
            })
    };
    let layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });
    let _pipeline = ctx
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: VertexState {
                module: &vertex,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &fragment,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWrites::all(),
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
}

#[gpu_test]
static METALLIB_PASSTHROUGH_SHADER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::EXPERIMENTAL_PASSTHROUGH_SHADERS)
            .skip(FailureCase::backend(Backends::all() - Backends::METAL)),
    )
    .run_sync(metallib_test);
