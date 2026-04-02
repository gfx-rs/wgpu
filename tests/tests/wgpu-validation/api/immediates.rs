//! Validation tests for `var<immediate>`

use wgpu_test::fail;

const COMPUTE_SHADER: &str = "
    var<immediate> im: vec4<f32>;

    @group(0) @binding(0)
    var<storage, read_write> output: vec4<f32>;

    @compute @workgroup_size(1)
    fn main() {
        output = im;
    }
";

fn setup_compute() -> (
    wgpu::Device,
    wgpu::Queue,
    wgpu::ComputePipeline,
    wgpu::BindGroup,
) {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::IMMEDIATES,
        required_limits: wgpu::Limits {
            max_immediate_size: 64,
            ..Default::default()
        },
        ..Default::default()
    });

    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(COMPUTE_SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 16,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &sm,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    (device, queue, pipeline, bind_group)
}

#[test]
fn dispatch_without_setting_immediates_fails() {
    let (device, _queue, pipeline, bind_group) = setup_compute();

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    fail(&device, || encoder.finish(), Some("immediate data"));
}

#[test]
fn dispatch_with_partial_immediates_fails() {
    let (device, _queue, pipeline, bind_group) = setup_compute();

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &[0u8; 8]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    fail(&device, || encoder.finish(), Some("immediate data"));
}

#[test]
fn dispatch_with_all_immediates_set_succeeds() {
    let (device, _queue, pipeline, bind_group) = setup_compute();

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &[0u8; 16]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    wgpu_test::valid(&device, || encoder.finish());
}

#[test]
fn dispatch_with_incremental_immediates_succeeds() {
    let (device, _queue, pipeline, bind_group) = setup_compute();

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &[0u8; 8]);
        pass.set_immediates(8, &[0u8; 8]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    wgpu_test::valid(&device, || encoder.finish());
}

const STRUCT_SHADER: &str = "
    struct S {
        a: f32,
        // 12 bytes padding
        b: vec4<f32>,
    }
    var<immediate> im: S;

    @group(0) @binding(0)
    var<storage, read_write> output: vec4<f32>;

    @compute @workgroup_size(1)
    fn main() {
        output = im.b;
    }
";

#[test]
fn struct_padding_slots_not_required() {
    let (device, _q) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::IMMEDIATES,
        required_limits: wgpu::Limits {
            max_immediate_size: 64,
            ..Default::default()
        },
        ..Default::default()
    });

    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(STRUCT_SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 32,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &sm,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // skip padding at bytes 4..16
        pass.set_immediates(0, &[0u8; 4]);
        pass.set_immediates(16, &[0u8; 16]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    wgpu_test::valid(&device, || encoder.finish());
}

const NO_IMMEDIATES_SHADER: &str = "
    @group(0) @binding(0)
    var<storage, read_write> output: u32;

    @compute @workgroup_size(1)
    fn main() {
        output = 42u;
    }
";

#[test]
fn pipeline_without_immediates_needs_none() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(NO_IMMEDIATES_SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &sm,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    wgpu_test::valid(&device, || encoder.finish());
}

#[test]
fn auto_layout_infers_immediate_size() {
    let (device, _q) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::IMMEDIATES,
        required_limits: wgpu::Limits {
            max_immediate_size: 64,
            ..Default::default()
        },
        ..Default::default()
    });

    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(COMPUTE_SHADER.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &sm,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &[0u8; 16]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    wgpu_test::valid(&device, || encoder.finish());
}
