use std::io;

use bitflags::Flags;

use crate::{
    report::{AdapterReport, GpuReport},
    texture::{self, TEXTURE_FORMAT_LIST},
};

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

    fn println_table_header(output: &mut impl io::Write) -> io::Result<()> {
        write!(output, "┌─")?;
        for (i, bit) in Self::valid_bits() {
            if i != 0 {
                write!(output, "─┬─")?;
            }
            let length = bit.name().len();
            write!(output, "{}", "─".repeat(length))?;
        }
        writeln!(output, "─┐")?;
        Ok(())
    }

    fn println_table_footer(output: &mut impl io::Write) -> io::Result<()> {
        write!(output, "└─")?;
        for (i, bit) in Self::valid_bits() {
            if i != 0 {
                write!(output, "─┴─")?;
            }
            let length = bit.name().len();
            write!(output, "{}", "─".repeat(length))?;
        }
        writeln!(output, "─┘")?;
        Ok(())
    }
}

impl<T> FlagsExt for T where T: Flags {}

fn print_empty_string(input: &str) -> &str {
    if input.is_empty() {
        "<empty>"
    } else {
        input
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PrintingVerbosity {
    /// Corresponds to the `-q` flag
    NameOnly,
    /// Corresponds to no flag.
    Information,
    /// Corresponds to the `-v` flag
    InformationFeaturesLimits,
    /// Corresponds to the `-vv` flag
    InformationFeaturesLimitsTexture,
}

// Lets keep these print statements on one line
#[rustfmt::skip]
fn print_adapter(output: &mut impl io::Write, report: &AdapterReport, idx: usize, verbosity: PrintingVerbosity) -> io::Result<()> {
    let AdapterReport { 
        info,
        features,
        limits,
        downlevel_caps:
        downlevel,
        texture_format_features
    } = &report;

    //////////////////
    // Adapter Info //
    //////////////////

    if matches!(verbosity, PrintingVerbosity::NameOnly) {
        writeln!(output, "Adapter {idx}: {} ({:?})", info.name, info.backend)?;
        return Ok(());
    }

    writeln!(output, "Adapter {idx}:")?;
    writeln!(output, "\t         Backend: {:?}", info.backend)?;
    writeln!(output, "\t            Name: {}", info.name)?;
    writeln!(output, "\t        VendorID: {:#X?}", info.vendor)?;
    writeln!(output, "\t        DeviceID: {:#X?}", info.device)?;
    writeln!(output, "\t            Type: {:?}", info.device_type)?;
    writeln!(output, "\t          Driver: {}", print_empty_string(&info.driver))?;
    writeln!(output, "\t      DriverInfo: {}", print_empty_string(&info.driver_info))?;
    writeln!(output, "\tWebGPU Compliant: {:?}", downlevel.is_webgpu_compliant())?;

    if matches!(verbosity, PrintingVerbosity::Information) {
        return Ok(());
    }

    //////////////
    // Features //
    //////////////

    writeln!(output, "\tFeatures:")?;
    let max_feature_flag_width = wgpu::Features::max_debug_print_width();
    for bit in wgpu::Features::all().iter() {
        writeln!(output, "\t\t{:>width$}: {}", bit.name(), features.contains(bit), width = max_feature_flag_width)?;
    }

    ////////////
    // Limits //
    ////////////

    writeln!(output, "\tLimits:")?;
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
        min_subgroup_size,
        max_subgroup_size,
        max_push_constant_size,
        min_uniform_buffer_offset_alignment,
        min_storage_buffer_offset_alignment,
        max_inter_stage_shader_components,
        max_color_attachments,
        max_color_attachment_bytes_per_sample,
        max_compute_workgroup_storage_size,
        max_compute_invocations_per_workgroup,
        max_compute_workgroup_size_x,
        max_compute_workgroup_size_y,
        max_compute_workgroup_size_z,
        max_compute_workgroups_per_dimension,
        max_non_sampler_bindings,
    } = limits;
    writeln!(output, "\t\t                        Max Texture Dimension 1d: {max_texture_dimension_1d}")?;
    writeln!(output, "\t\t                        Max Texture Dimension 2d: {max_texture_dimension_2d}")?;
    writeln!(output, "\t\t                        Max Texture Dimension 3d: {max_texture_dimension_3d}")?;
    writeln!(output, "\t\t                        Max Texture Array Layers: {max_texture_array_layers}")?;
    writeln!(output, "\t\t                                 Max Bind Groups: {max_bind_groups}")?;
    writeln!(output, "\t\t                     Max Bindings Per Bind Group: {max_bindings_per_bind_group}")?;
    writeln!(output, "\t\t Max Dynamic Uniform Buffers Per Pipeline Layout: {max_dynamic_uniform_buffers_per_pipeline_layout}")?;
    writeln!(output, "\t\t Max Dynamic Storage Buffers Per Pipeline Layout: {max_dynamic_storage_buffers_per_pipeline_layout}")?;
    writeln!(output, "\t\t           Max Sampled Textures Per Shader Stage: {max_sampled_textures_per_shader_stage}")?;
    writeln!(output, "\t\t                   Max Samplers Per Shader Stage: {max_samplers_per_shader_stage}")?;
    writeln!(output, "\t\t            Max Storage Buffers Per Shader Stage: {max_storage_buffers_per_shader_stage}")?;
    writeln!(output, "\t\t           Max Storage Textures Per Shader Stage: {max_storage_textures_per_shader_stage}")?;
    writeln!(output, "\t\t            Max Uniform Buffers Per Shader Stage: {max_uniform_buffers_per_shader_stage}")?;
    writeln!(output, "\t\t                 Max Uniform Buffer Binding Size: {max_uniform_buffer_binding_size}")?;
    writeln!(output, "\t\t                 Max Storage Buffer Binding Size: {max_storage_buffer_binding_size}")?;
    writeln!(output, "\t\t                                 Max Buffer Size: {max_buffer_size}")?;
    writeln!(output, "\t\t                              Max Vertex Buffers: {max_vertex_buffers}")?;
    writeln!(output, "\t\t                           Max Vertex Attributes: {max_vertex_attributes}")?;
    writeln!(output, "\t\t                  Max Vertex Buffer Array Stride: {max_vertex_buffer_array_stride}")?;
    writeln!(output, "\t\t                               Min Subgroup Size: {min_subgroup_size}")?;
    writeln!(output, "\t\t                               Max Subgroup Size: {max_subgroup_size}")?;
    writeln!(output, "\t\t                          Max Push Constant Size: {max_push_constant_size}")?;
    writeln!(output, "\t\t             Min Uniform Buffer Offset Alignment: {min_uniform_buffer_offset_alignment}")?;
    writeln!(output, "\t\t             Min Storage Buffer Offset Alignment: {min_storage_buffer_offset_alignment}")?;
    writeln!(output, "\t\t                Max Inter-Stage Shader Component: {max_inter_stage_shader_components}")?;
    writeln!(output, "\t\t                           Max Color Attachments: {max_color_attachments}")?;
    writeln!(output, "\t\t           Max Color Attachment Bytes per sample: {max_color_attachment_bytes_per_sample}")?;
    writeln!(output, "\t\t              Max Compute Workgroup Storage Size: {max_compute_workgroup_storage_size}")?;
    writeln!(output, "\t\t           Max Compute Invocations Per Workgroup: {max_compute_invocations_per_workgroup}")?;
    writeln!(output, "\t\t                    Max Compute Workgroup Size X: {max_compute_workgroup_size_x}")?;
    writeln!(output, "\t\t                    Max Compute Workgroup Size Y: {max_compute_workgroup_size_y}")?;
    writeln!(output, "\t\t                    Max Compute Workgroup Size Z: {max_compute_workgroup_size_z}")?;
    writeln!(output, "\t\t            Max Compute Workgroups Per Dimension: {max_compute_workgroups_per_dimension}")?;

    // This one reflects more of a wgpu implementation limitations than a hardware limit
    // so don't show it here.
    let _ = max_non_sampler_bindings;

    //////////////////////////
    // Downlevel Properties //
    //////////////////////////

    writeln!(output, "\tDownlevel Properties:")?;
    let wgpu::DownlevelCapabilities {
        shader_model: _,
        limits: _,
        flags,
    } = downlevel;
    let max_downlevel_flag_width = wgpu::DownlevelFlags::max_debug_print_width();
    for bit in wgpu::DownlevelFlags::all().iter() {
        writeln!(output, "\t\t{:>width$}: {}", bit.name(), flags.contains(bit), width = max_downlevel_flag_width)?;
    };

    if matches!(verbosity, PrintingVerbosity::InformationFeaturesLimits) {
        return Ok(());
    }

    ////////////////////
    // Texture Usages //
    ////////////////////

    let max_format_name_size = texture::max_texture_format_string_size();
    let texture_format_whitespace = " ".repeat(max_format_name_size);

    writeln!(output, "\n\t Texture Format Allowed Usages:")?;

    write!(output, "\t\t {texture_format_whitespace}")?;
    wgpu::TextureUsages::println_table_header(output)?;
    for format in TEXTURE_FORMAT_LIST {
        let features = texture_format_features[&format];
        let format_name = texture::texture_format_name(format);
        write!(output, "\t\t{format_name:>max_format_name_size$}")?;
        for bit in wgpu::TextureUsages::all().iter() {
            write!(output, " │ ")?;
            if features.allowed_usages.contains(bit) {
                write!(output, "{}", bit.name())?;
            }
            else {
                let length = bit.name().len();
                write!(output, "{}", " ".repeat(length))?;
            }
        };
        writeln!(output, " │")?;
    }
    write!(output, "\t\t {texture_format_whitespace}")?;
    wgpu::TextureUsages::println_table_footer(output)?;

    //////////////////////////
    // Texture Format Flags //
    //////////////////////////

    writeln!(output, "\n\t Texture Format Flags:")?;

    write!(output, "\t\t {texture_format_whitespace}")?;
    wgpu::TextureFormatFeatureFlags::println_table_header(output)?;

    for format in TEXTURE_FORMAT_LIST {
        let features = texture_format_features[&format];
        let format_name = texture::texture_format_name(format);

        write!(output, "\t\t{format_name:>max_format_name_size$}")?;
        for bit in wgpu::TextureFormatFeatureFlags::all().iter() {
            write!(output, " │ ")?;
            if features.flags.contains(bit) {
                write!(output, "{}", bit.name())?;
            }
            else {
                let length = bit.name().len();
                write!(output, "{}", " ".repeat(length))?;
            }
        };
        writeln!(output, " │")?;
    }
    write!(output, "\t\t {texture_format_whitespace}")?;
    wgpu::TextureFormatFeatureFlags::println_table_footer(output)?;
    Ok(())
}

pub fn print_adapters(
    output: &mut impl io::Write,
    report: &GpuReport,
    verbosity: PrintingVerbosity,
) -> io::Result<()> {
    for (idx, adapter) in report.devices.iter().enumerate() {
        print_adapter(output, adapter, idx, verbosity)?;
    }
    Ok(())
}
