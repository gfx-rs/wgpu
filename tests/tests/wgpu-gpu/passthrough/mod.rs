use std::{
    borrow::Cow,
    hash::{DefaultHasher, Hash, Hasher},
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
    tests.push(HLSL_PASSTHROUGH_SHADER);
    tests.push(DXIL_PASSTHROUGH_SHADER);
    tests.push(SPIRV_PASSTHROUGH_SHADER);
    tests.push(GLSL_PASSTHROUGH_SHADER);
    tests.push(WGSL_PASSTHROUGH_SHADER);
    tests.push(ALL_PASSTHROUGH_SHADERS_BINARY);
    tests.push(ALL_PASSTHROUGH_SHADERS_SOURCE);
    tests.push(PASSTHROUGH_SHADERS_EXPLICIT_LAYOUT_VALIDATION);
}

fn test_hash(ctx: &TestingContext, name: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    ctx.hash(&mut hasher);
    name.hash(&mut hasher);
    hasher.finish()
}

fn test_with_module(ctx: TestingContext, vertex: wgpu::ShaderModule, fragment: wgpu::ShaderModule) {
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

fn metal_source() -> Cow<'static, str> {
    Cow::Borrowed(include_str!("shader.metal"))
}

fn metal_test(ctx: TestingContext) {
    let module = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                label: None,
                num_workgroups: (0, 0, 0),
                msl: Some(metal_source()),
                ..Default::default()
            })
    };
    test_with_module(ctx, module.clone(), module);
}

#[gpu_test]
static METAL_PASSTHROUGH_SHADER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::PASSTHROUGH_SHADERS)
            .skip(FailureCase::backend(!Backends::METAL)),
    )
    .run_sync(metal_test);

fn metallib_source(test_hash: u64) -> Cow<'static, [u8]> {
    struct FileDropGuard<'a> {
        file_name: &'a str,
    }
    impl Drop for FileDropGuard<'_> {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(self.file_name);
        }
    }
    if cfg!(not(target_vendor = "apple")) {
        return Cow::Borrowed(&[]);
    }
    let metal_compiler = std::process::Command::new("xcrun")
        .args(["--find", "metal"])
        .status()
        .is_ok_and(|a| a.success());
    let metallib_linker = std::process::Command::new("xcrun")
        .args(["--find", "metallib"])
        .status()
        .is_ok_and(|a| a.success());
    if !metal_compiler || !metallib_linker {
        panic!("Metal compiler or metallib linker not present. Most users can safely ignore this.");
    }
    let air_name = format!(
        "{}/tests/wgpu-gpu/passthrough/shader{test_hash}.air",
        env!("CARGO_MANIFEST_DIR")
    );
    let output_name = format!(
        "{}/tests/wgpu-gpu/passthrough/shader{test_hash}.metallib",
        env!("CARGO_MANIFEST_DIR")
    );

    let _air_drop_guard = FileDropGuard {
        file_name: &air_name,
    };

    {
        let output = std::process::Command::new("xcrun")
            .args([
                "metal",
                "-c",
                &format!(
                    "{}/tests/wgpu-gpu/passthrough/shader.metal",
                    env!("CARGO_MANIFEST_DIR")
                ),
                "-o",
                &air_name,
            ])
            .output()
            .unwrap();
        if !output.status.success() {
            panic!(
                "Failed to compile .metal into .air: {}",
                String::from_utf8(output.stderr).unwrap()
            );
        }
    }

    let _metallib_drop_guard = FileDropGuard {
        file_name: &output_name,
    };

    {
        let output = std::process::Command::new("xcrun")
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
    Cow::Owned(source)
}

fn metallib_test(ctx: TestingContext) {
    let test_hash = test_hash(&ctx, "metallib_test");
    let source = metallib_source(test_hash);
    let module = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                label: None,
                num_workgroups: (0, 0, 0),
                metallib: Some(std::borrow::Cow::Borrowed(&source)),
                ..Default::default()
            })
    };
    test_with_module(ctx, module.clone(), module);
}

#[gpu_test]
static METALLIB_PASSTHROUGH_SHADER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::PASSTHROUGH_SHADERS)
            .skip(FailureCase::backend(!Backends::METAL)),
    )
    .run_sync(metallib_test);

fn hlsl_source() -> Cow<'static, str> {
    std::borrow::Cow::Borrowed(include_str!("shader.hlsl"))
}

fn hlsl_test(ctx: TestingContext) {
    let module = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                hlsl: Some(hlsl_source()),
                ..Default::default()
            })
    };
    test_with_module(ctx, module.clone(), module);
}

#[gpu_test]
static HLSL_PASSTHROUGH_SHADER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::PASSTHROUGH_SHADERS)
            .skip(FailureCase::backend(!Backends::DX12)),
    )
    .run_sync(hlsl_test);

fn compile_dxil(entry: &str, stage_str: &str, test_hash: u64) -> Cow<'static, [u8]> {
    let out_path = format!(
        "{}/tests/wgpu-gpu/passthrough/shader{test_hash}.{stage_str}.cso",
        env!("CARGO_MANIFEST_DIR")
    );
    let cmd = std::process::Command::new("dxc")
        .args([
            "-T",
            &format!("{stage_str}_6_3"),
            "-E",
            entry,
            &format!(
                "{}/tests/wgpu-gpu/passthrough/shader.hlsl",
                env!("CARGO_MANIFEST_DIR")
            ),
            "-Fo",
            &out_path,
        ])
        .output()
        .unwrap();
    let file = std::fs::read(&out_path);
    let _ = std::fs::remove_file(out_path);
    // Remove the file before checking for status
    if !cmd.status.success() {
        panic!("DXC failed:\n{}", String::from_utf8(cmd.stderr).unwrap());
    }
    let file = file.unwrap();
    Cow::Owned(file)
}

fn dxil_vertex_source(test_hash: u64) -> Cow<'static, [u8]> {
    if cfg!(target_os = "windows") {
        compile_dxil("vertex_main", "vs", test_hash)
    } else {
        Cow::Borrowed(&[])
    }
}

fn dxil_fragment_source(test_hash: u64) -> Cow<'static, [u8]> {
    if cfg!(target_os = "windows") {
        compile_dxil("fragment_main", "ps", test_hash)
    } else {
        Cow::Borrowed(&[])
    }
}

fn dxil_test(ctx: TestingContext) {
    let test_hash = test_hash(&ctx, "dxil_test");
    let vertex_source = dxil_vertex_source(test_hash);
    let vertex = unsafe {
        ctx.device
            .create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: None,
                num_workgroups: (1, 1, 1),
                dxil: Some(vertex_source),
                ..Default::default()
            })
    };
    let fragment_source = dxil_fragment_source(test_hash);
    let fragment = unsafe {
        ctx.device
            .create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: None,
                num_workgroups: (1, 1, 1),
                dxil: Some(fragment_source),
                ..Default::default()
            })
    };
    test_with_module(ctx, vertex, fragment);
}

#[gpu_test]
static DXIL_PASSTHROUGH_SHADER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::PASSTHROUGH_SHADERS)
            .skip(FailureCase::backend(!Backends::DX12)),
    )
    .run_sync(dxil_test);

fn spirv_source(test_hash: u64) -> Cow<'static, [u32]> {
    let out_path = format!(
        "{}/tests/wgpu-gpu/passthrough/shade{test_hash}.spv",
        env!("CARGO_MANIFEST_DIR")
    );
    let cmd = std::process::Command::new("dxc")
        .args([
            "-spirv",
            "-T",
            "lib_6_3",
            "-fspv-target-env=vulkan1.0",
            // We need to tell it to compile for SPIRV which requires different info
            "-D",
            "SPIRV",
            &format!(
                "{}/tests/wgpu-gpu/passthrough/shader.hlsl",
                env!("CARGO_MANIFEST_DIR")
            ),
            "-Fo",
            &out_path,
        ])
        .output()
        .unwrap();
    let file = std::fs::read(&out_path);
    let _ = std::fs::remove_file(out_path);
    // Remove the file before checking for status
    if !cmd.status.success() {
        panic!("DXC failed:\n{}", String::from_utf8(cmd.stderr).unwrap());
    }
    let file = file.unwrap();
    let spirv = bytemuck::pod_collect_to_vec::<u8, u32>(&file);
    Cow::Owned(spirv)
}

fn spirv_test(ctx: TestingContext) {
    let test_hash = test_hash(&ctx, "spirv_test");
    let module = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                spirv: Some(spirv_source(test_hash)),
                ..Default::default()
            })
    };
    test_with_module(ctx, module.clone(), module);
}

#[gpu_test]
static SPIRV_PASSTHROUGH_SHADER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::PASSTHROUGH_SHADERS)
            .skip(FailureCase::backend(!Backends::VULKAN)),
    )
    .run_sync(spirv_test);

fn glsl_vertex_source() -> Cow<'static, str> {
    std::borrow::Cow::Borrowed(include_str!("shader.vert"))
}

fn glsl_fragment_source() -> Cow<'static, str> {
    std::borrow::Cow::Borrowed(include_str!("shader.frag"))
}

fn glsl_test(ctx: TestingContext) {
    let vertex = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                glsl: Some(glsl_vertex_source()),
                ..Default::default()
            })
    };
    let fragment = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                glsl: Some(glsl_fragment_source()),
                ..Default::default()
            })
    };
    test_with_module(ctx, vertex, fragment);
}

#[gpu_test]
static GLSL_PASSTHROUGH_SHADER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::PASSTHROUGH_SHADERS)
            .skip(FailureCase::backend(!Backends::GL)),
    )
    .run_sync(glsl_test);

fn wgsl_source() -> Cow<'static, str> {
    std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))
}

fn wgsl_test(ctx: TestingContext) {
    let module = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                wgsl: Some(wgsl_source()),
                ..Default::default()
            })
    };
    test_with_module(ctx, module.clone(), module);
}

#[gpu_test]
static WGSL_PASSTHROUGH_SHADER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::PASSTHROUGH_SHADERS)
            .skip(FailureCase::backend(!Backends::BROWSER_WEBGPU)),
    )
    .run_sync(wgsl_test);

fn all_passthrough_shaders_binary(ctx: TestingContext) {
    let test_hash = test_hash(&ctx, "all_passthrough_binary");
    let vertex = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                label: None,
                num_workgroups: (0, 0, 0),
                spirv: Some(spirv_source(test_hash)),
                dxil: Some(dxil_vertex_source(test_hash)),
                hlsl: None,
                metallib: Some(metallib_source(test_hash)),
                msl: None,
                glsl: Some(glsl_vertex_source()),
                wgsl: Some(wgsl_source()),
            })
    };
    let fragment = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                label: None,
                num_workgroups: (0, 0, 0),
                spirv: Some(spirv_source(test_hash)),
                dxil: Some(dxil_fragment_source(test_hash)),
                hlsl: None,
                metallib: Some(metallib_source(test_hash)),
                msl: None,
                glsl: Some(glsl_fragment_source()),
                wgsl: Some(wgsl_source()),
            })
    };
    test_with_module(ctx, vertex, fragment);
}

#[gpu_test]
static ALL_PASSTHROUGH_SHADERS_BINARY: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(Features::PASSTHROUGH_SHADERS))
    .run_sync(all_passthrough_shaders_binary);

fn all_passthrough_shader_source(ctx: TestingContext) {
    let test_hash = test_hash(&ctx, "all_passthrough_source");
    let vertex = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                label: None,
                num_workgroups: (0, 0, 0),
                spirv: Some(spirv_source(test_hash)),
                dxil: None,
                hlsl: Some(hlsl_source()),
                metallib: None,
                msl: Some(metal_source()),
                glsl: Some(glsl_vertex_source()),
                wgsl: Some(wgsl_source()),
            })
    };
    let fragment = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                label: None,
                num_workgroups: (0, 0, 0),
                spirv: Some(spirv_source(test_hash)),
                dxil: None,
                hlsl: Some(hlsl_source()),
                metallib: None,
                msl: Some(metal_source()),
                glsl: Some(glsl_fragment_source()),
                wgsl: Some(wgsl_source()),
            })
    };
    test_with_module(ctx, vertex, fragment);
}

#[gpu_test]
static ALL_PASSTHROUGH_SHADERS_SOURCE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(Features::PASSTHROUGH_SHADERS))
    .run_sync(all_passthrough_shader_source);

fn explicit_layout_validation(ctx: TestingContext) {
    let test_hash = test_hash(&ctx, "explicit_layout_validation");
    let vertex = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                label: None,
                num_workgroups: (0, 0, 0),
                spirv: Some(spirv_source(test_hash)),
                dxil: None,
                hlsl: Some(hlsl_source()),
                metallib: None,
                msl: Some(metal_source()),
                glsl: Some(glsl_vertex_source()),
                wgsl: Some(wgsl_source()),
            })
    };
    let fragment = unsafe {
        ctx.device
            .create_shader_module_passthrough(CreateShaderModuleDescriptorPassthrough {
                label: None,
                num_workgroups: (0, 0, 0),
                spirv: Some(spirv_source(test_hash)),
                dxil: None,
                hlsl: Some(hlsl_source()),
                metallib: None,
                msl: Some(metal_source()),
                glsl: Some(glsl_fragment_source()),
                wgsl: Some(wgsl_source()),
            })
    };

    let _pipeline = ctx
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: None,
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
static PASSTHROUGH_SHADERS_EXPLICIT_LAYOUT_VALIDATION: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .features(Features::PASSTHROUGH_SHADERS)
                .expect_fail(FailureCase::always()),
        )
        .run_sync(explicit_layout_validation);
