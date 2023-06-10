use std::io;

use bitflags::Flags;

trait FlagsExt: Flags {
    fn name(&self) -> &'static str {
        self.iter_names().next().unwrap().0
    }

    fn valid_bits() -> std::iter::Enumerate<bitflags::iter::Iter<Self>> {
        Self::all().iter().enumerate()
    }

    fn max_debug_print_width() -> usize {
        let mut width = 0;
        for bit in Self::all().iter() {
            width = width.max(bit.name().len());
        }
        width
    }

    fn println_table_header(output: &mut dyn io::Write) {
        write!(output, "┌─");
        for (i, bit) in Self::valid_bits() {
            if i != 0 {
                write!(output, "─┬─");
            }
            let length = bit.name().len();
            write!(output, "{}", "─".repeat(length));
        }
        writeln!(output, "─┐");
    }

    fn println_table_footer(output: &mut dyn io::Write) {
        write!(output, "└─");
        for (i, bit) in Self::valid_bits() {
            if i != 0 {
                write!(output, "─┴─");
            }
            let length = bit.name().len();
            write!(output, "{}", "─".repeat(length));
        }
        writeln!(output, "─┘");
    }
}

impl<T> FlagsExt for T where T: Flags {}

fn max_texture_format_name_size() -> usize {
    TEXTURE_FORMAT_LIST
        .into_iter()
        .map(|f| texture_format_name(f).len())
        .max()
        .unwrap()
}

fn texture_format_name(format: wgpu::TextureFormat) -> String {
    match format {
        wgpu::TextureFormat::Astc { block, channel } => {
            format!("Astc{block:?}{channel:?}:")
        }
        _ => {
            format!("{format:?}:")
        }
    }
}

// Lets keep these on one line
#[rustfmt::skip]
const TEXTURE_FORMAT_LIST: [wgpu::TextureFormat; 114] = [
    wgpu::TextureFormat::R8Unorm,
    wgpu::TextureFormat::R8Snorm,
    wgpu::TextureFormat::R8Uint,
    wgpu::TextureFormat::R8Sint,
    wgpu::TextureFormat::R16Uint,
    wgpu::TextureFormat::R16Sint,
    wgpu::TextureFormat::R16Unorm,
    wgpu::TextureFormat::R16Snorm,
    wgpu::TextureFormat::R16Float,
    wgpu::TextureFormat::Rg8Unorm,
    wgpu::TextureFormat::Rg8Snorm,
    wgpu::TextureFormat::Rg8Uint,
    wgpu::TextureFormat::Rg8Sint,
    wgpu::TextureFormat::R32Uint,
    wgpu::TextureFormat::R32Sint,
    wgpu::TextureFormat::R32Float,
    wgpu::TextureFormat::Rg16Uint,
    wgpu::TextureFormat::Rg16Sint,
    wgpu::TextureFormat::Rg16Unorm,
    wgpu::TextureFormat::Rg16Snorm,
    wgpu::TextureFormat::Rg16Float,
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba8UnormSrgb,
    wgpu::TextureFormat::Rgba8Snorm,
    wgpu::TextureFormat::Rgba8Uint,
    wgpu::TextureFormat::Rgba8Sint,
    wgpu::TextureFormat::Bgra8Unorm,
    wgpu::TextureFormat::Bgra8UnormSrgb,
    wgpu::TextureFormat::Rgb10a2Unorm,
    wgpu::TextureFormat::Rg11b10Float,
    wgpu::TextureFormat::Rg32Uint,
    wgpu::TextureFormat::Rg32Sint,
    wgpu::TextureFormat::Rg32Float,
    wgpu::TextureFormat::Rgba16Uint,
    wgpu::TextureFormat::Rgba16Sint,
    wgpu::TextureFormat::Rgba16Unorm,
    wgpu::TextureFormat::Rgba16Snorm,
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rgba32Uint,
    wgpu::TextureFormat::Rgba32Sint,
    wgpu::TextureFormat::Rgba32Float,
    wgpu::TextureFormat::Stencil8,
    wgpu::TextureFormat::Depth16Unorm,
    wgpu::TextureFormat::Depth32Float,
    wgpu::TextureFormat::Depth32FloatStencil8,
    wgpu::TextureFormat::Depth24Plus,
    wgpu::TextureFormat::Depth24PlusStencil8,
    wgpu::TextureFormat::Rgb9e5Ufloat,
    wgpu::TextureFormat::Bc1RgbaUnorm,
    wgpu::TextureFormat::Bc1RgbaUnormSrgb,
    wgpu::TextureFormat::Bc2RgbaUnorm,
    wgpu::TextureFormat::Bc2RgbaUnormSrgb,
    wgpu::TextureFormat::Bc3RgbaUnorm,
    wgpu::TextureFormat::Bc3RgbaUnormSrgb,
    wgpu::TextureFormat::Bc4RUnorm,
    wgpu::TextureFormat::Bc4RSnorm,
    wgpu::TextureFormat::Bc5RgUnorm,
    wgpu::TextureFormat::Bc5RgSnorm,
    wgpu::TextureFormat::Bc6hRgbUfloat,
    wgpu::TextureFormat::Bc6hRgbFloat,
    wgpu::TextureFormat::Bc7RgbaUnorm,
    wgpu::TextureFormat::Bc7RgbaUnormSrgb,
    wgpu::TextureFormat::Etc2Rgb8Unorm,
    wgpu::TextureFormat::Etc2Rgb8UnormSrgb,
    wgpu::TextureFormat::Etc2Rgb8A1Unorm,
    wgpu::TextureFormat::Etc2Rgb8A1UnormSrgb,
    wgpu::TextureFormat::Etc2Rgba8Unorm,
    wgpu::TextureFormat::Etc2Rgba8UnormSrgb,
    wgpu::TextureFormat::EacR11Unorm,
    wgpu::TextureFormat::EacR11Snorm,
    wgpu::TextureFormat::EacRg11Unorm,
    wgpu::TextureFormat::EacRg11Snorm,
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x4, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x4, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x4, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x5, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x5, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x5, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x5, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x5, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x5, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x5, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x5, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x5, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x6, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x6, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x6, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x5, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x5, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x5, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x6, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x6, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x6, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x8, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x8, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x8, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x10, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x10, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x10, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x10, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x10, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x10, channel: wgpu::AstcChannel::Hdr },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x12, channel: wgpu::AstcChannel::Unorm },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x12, channel: wgpu::AstcChannel::UnormSrgb },
    wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x12, channel: wgpu::AstcChannel::Hdr },
];

// Lets keep these on one line
#[rustfmt::skip]
fn print_adapter(output: &mut dyn io::Write, adapter: &wgpu::Adapter, idx: usize) {
    let info = adapter.get_info();
    let downlevel = adapter.get_downlevel_capabilities();
    let features = adapter.features();
    let limits = adapter.limits();

    //////////////////
    // Adapter Info //
    //////////////////

    writeln!(output, "Adapter {idx}:");
    writeln!(output, "\t   Backend: {:?}", info.backend);
    writeln!(output, "\t      Name: {:?}", info.name);
    writeln!(output, "\t  VendorID: {:?}", info.vendor);
    writeln!(output, "\t  DeviceID: {:?}", info.device);
    writeln!(output, "\t      Type: {:?}", info.device_type);
    writeln!(output, "\t    Driver: {:?}", info.driver);
    writeln!(output, "\tDriverInfo: {:?}", info.driver_info);
    writeln!(output, "\t Compliant: {:?}", downlevel.is_webgpu_compliant());

    //////////////
    // Features //
    //////////////

    writeln!(output, "\tFeatures:");
    let max_feature_flag_width = wgpu::Features::max_debug_print_width();
    for bit in wgpu::Features::all().iter() {
        writeln!(output, "\t\t{:>width$}: {}", bit.name(), features.contains(bit), width = max_feature_flag_width);
    }

    ////////////
    // Limits //
    ////////////

    writeln!(output, "\tLimits:");
    let wgpu::Limits {
        max_texture_dimension_1d,
        max_texture_dimension_2d,
        max_texture_dimension_3d,
        max_texture_array_layers,
        max_bind_groups,
        max_bindings_per_bind_group,
        max_dynamic_uniform_buffers_per_pipeline_layout,
        max_dynamic_storage_buffers_per_pipeline_layout,
        max_sampled_textures_per_shader_stage,
        max_samplers_per_shader_stage,
        max_storage_buffers_per_shader_stage,
        max_storage_textures_per_shader_stage,
        max_uniform_buffers_per_shader_stage,
        max_uniform_buffer_binding_size,
        max_storage_buffer_binding_size,
        max_buffer_size,
        max_vertex_buffers,
        max_vertex_attributes,
        max_vertex_buffer_array_stride,
        max_push_constant_size,
        min_uniform_buffer_offset_alignment,
        min_storage_buffer_offset_alignment,
        max_inter_stage_shader_components,
        max_compute_workgroup_storage_size,
        max_compute_invocations_per_workgroup,
        max_compute_workgroup_size_x,
        max_compute_workgroup_size_y,
        max_compute_workgroup_size_z,
        max_compute_workgroups_per_dimension,
    } = limits;
    writeln!(output, "\t\t                        Max Texture Dimension 1d: {max_texture_dimension_1d}");
    writeln!(output, "\t\t                        Max Texture Dimension 2d: {max_texture_dimension_2d}");
    writeln!(output, "\t\t                        Max Texture Dimension 3d: {max_texture_dimension_3d}");
    writeln!(output, "\t\t                        Max Texture Array Layers: {max_texture_array_layers}");
    writeln!(output, "\t\t                                 Max Bind Groups: {max_bind_groups}");
    writeln!(output, "\t\t                     Max Bindings Per Bind Group: {max_bindings_per_bind_group}");
    writeln!(output, "\t\t Max Dynamic Uniform Buffers Per Pipeline Layout: {max_dynamic_uniform_buffers_per_pipeline_layout}");
    writeln!(output, "\t\t Max Dynamic Storage Buffers Per Pipeline Layout: {max_dynamic_storage_buffers_per_pipeline_layout}");
    writeln!(output, "\t\t           Max Sampled Textures Per Shader Stage: {max_sampled_textures_per_shader_stage}");
    writeln!(output, "\t\t                   Max Samplers Per Shader Stage: {max_samplers_per_shader_stage}");
    writeln!(output, "\t\t            Max Storage Buffers Per Shader Stage: {max_storage_buffers_per_shader_stage}");
    writeln!(output, "\t\t           Max Storage Textures Per Shader Stage: {max_storage_textures_per_shader_stage}");
    writeln!(output, "\t\t            Max Uniform Buffers Per Shader Stage: {max_uniform_buffers_per_shader_stage}");
    writeln!(output, "\t\t                 Max Uniform Buffer Binding Size: {max_uniform_buffer_binding_size}");
    writeln!(output, "\t\t                 Max Storage Buffer Binding Size: {max_storage_buffer_binding_size}");
    writeln!(output, "\t\t                                 Max Buffer Size: {max_buffer_size}");
    writeln!(output, "\t\t                              Max Vertex Buffers: {max_vertex_buffers}");
    writeln!(output, "\t\t                           Max Vertex Attributes: {max_vertex_attributes}");
    writeln!(output, "\t\t                  Max Vertex Buffer Array Stride: {max_vertex_buffer_array_stride}");
    writeln!(output, "\t\t                          Max Push Constant Size: {max_push_constant_size}");
    writeln!(output, "\t\t             Min Uniform Buffer Offset Alignment: {min_uniform_buffer_offset_alignment}");
    writeln!(output, "\t\t             Min Storage Buffer Offset Alignment: {min_storage_buffer_offset_alignment}");
    writeln!(output, "\t\t                Max Inter-Stage Shader Component: {max_inter_stage_shader_components}");
    writeln!(output, "\t\t              Max Compute Workgroup Storage Size: {max_compute_workgroup_storage_size}");
    writeln!(output, "\t\t           Max Compute Invocations Per Workgroup: {max_compute_invocations_per_workgroup}");
    writeln!(output, "\t\t                    Max Compute Workgroup Size X: {max_compute_workgroup_size_x}");
    writeln!(output, "\t\t                    Max Compute Workgroup Size Y: {max_compute_workgroup_size_y}");
    writeln!(output, "\t\t                    Max Compute Workgroup Size Z: {max_compute_workgroup_size_z}");
    writeln!(output, "\t\t            Max Compute Workgroups Per Dimension: {max_compute_workgroups_per_dimension}");

    //////////////////////////
    // Downlevel Properties //
    //////////////////////////

    writeln!(output, "\tDownlevel Properties:");
    let wgpu::DownlevelCapabilities {
        shader_model,
        limits: _,
        flags,
    } = downlevel;
    writeln!(output, "\t\t                       Shader Model: {shader_model:?}");
    let max_downlevel_flag_width = wgpu::DownlevelFlags::max_debug_print_width();
    for bit in wgpu::DownlevelFlags::all().iter() {
        writeln!(output, "\t\t{:>width$}: {}", bit.name(), flags.contains(bit), width = max_downlevel_flag_width);
    };

    ////////////////////
    // Texture Usages //
    ////////////////////

    let max_format_name_size = max_texture_format_name_size();
    let texture_format_whitespace = " ".repeat(max_format_name_size);

    writeln!(output, "\n\t Texture Format Allowed Usages:");

    write!(output, "\t\t {texture_format_whitespace}");
    wgpu::TextureUsages::println_table_header(output);
    for format in TEXTURE_FORMAT_LIST {
        let features = adapter.get_texture_format_features(format);
        let format_name = texture_format_name(format);
        write!(output, "\t\t{format_name:>0$}", max_format_name_size);
        for bit in wgpu::TextureUsages::all().iter() {
            write!(output, " │ ");
            if features.allowed_usages.contains(bit) {
                write!(output, "{}", bit.name());
            }
            else {
                let length = bit.name().len();
                write!(output, "{}", " ".repeat(length));
            }
        };
        writeln!(output, " │");
    }
    write!(output, "\t\t {texture_format_whitespace}");
    wgpu::TextureUsages::println_table_footer(output);

    //////////////////////////
    // Texture Format Flags //
    //////////////////////////

    writeln!(output, "\n\t Texture Format Flags:");

    write!(output, "\t\t {texture_format_whitespace}");
    wgpu::TextureFormatFeatureFlags::println_table_header(output);

    for format in TEXTURE_FORMAT_LIST {
        let features = adapter.get_texture_format_features(format);
        let format_name = texture_format_name(format);

        write!(output, "\t\t{format_name:>0$}", max_format_name_size);
        for bit in wgpu::TextureFormatFeatureFlags::all().iter() {
            write!(output, " │ ");
            if features.flags.contains(bit) {
                write!(output, "{}", bit.name());
            }
            else {
                let length = bit.name().len();
                write!(output, "{}", " ".repeat(length));
            }
        };
        writeln!(output, " │");
    }
    write!(output, "\t\t {texture_format_whitespace}");
    wgpu::TextureFormatFeatureFlags::println_table_footer(output);
}

pub fn print_adapters(output: &mut dyn io::Write) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());

    for (idx, adapter) in adapters.into_iter().enumerate() {
        print_adapter(output, &adapter, idx);
    }
}
