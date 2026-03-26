use std::io;

use bitflags::Flags;
use exhaust::Exhaust;
use wgpu::AdapterInfo;

use crate::{
    report::{AdapterReport, GpuReport},
    texture,
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

    let AdapterInfo {
        name,
        vendor,
        device,
        device_type,
        device_pci_bus_id,
        driver,
        driver_info,
        backend,
        subgroup_min_size,
        subgroup_max_size,
        transient_saves_memory,
        limit_bucket,
    } = info;

    if matches!(verbosity, PrintingVerbosity::NameOnly) {
        writeln!(output, "Adapter {idx}: {} ({:?})", info.name, info.backend)?;
        return Ok(());
    }

    writeln!(output, "Adapter {idx}:")?;
    writeln!(output, "\t               Backend: {backend:?}")?;
    writeln!(output, "\t                  Name: {name}")?;
    writeln!(output, "\t             Vendor ID: {vendor:#X?}")?;
    writeln!(output, "\t             Device ID: {device:#X?}")?;
    writeln!(output, "\t     Device PCI Bus ID: {}", print_empty_string(device_pci_bus_id))?;
    writeln!(output, "\t                  Type: {device_type:?}")?;
    writeln!(output, "\t                Driver: {}", print_empty_string(driver))?;
    writeln!(output, "\t           Driver Info: {}", print_empty_string(driver_info))?;
    writeln!(output, "\t     Subgroup Min Size: {subgroup_min_size}")?;
    writeln!(output, "\t     Subgroup Max Size: {subgroup_max_size}")?;
    writeln!(output, "\tTransient Saves Memory: {transient_saves_memory}")?;
    writeln!(output, "\t          Limit Bucket: {}", limit_bucket.as_ref().map_or("<disabled>", |b| &b.name))?;
    writeln!(output, "\t      WebGPU Compliant: {:?}", downlevel.is_webgpu_compliant())?;

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
        max_binding_array_elements_per_shader_stage,
        max_binding_array_sampler_elements_per_shader_stage,
        max_binding_array_acceleration_structure_elements_per_shader_stage,
        max_uniform_buffer_binding_size,
        max_storage_buffer_binding_size,
        max_vertex_buffers,
        max_buffer_size,
        max_vertex_attributes,
        max_vertex_buffer_array_stride,
        max_inter_stage_shader_variables,
        min_uniform_buffer_offset_alignment,
        min_storage_buffer_offset_alignment,
        max_color_attachments,
        max_color_attachment_bytes_per_sample,
        max_compute_workgroup_storage_size,
        max_compute_invocations_per_workgroup,
        max_compute_workgroup_size_x,
        max_compute_workgroup_size_y,
        max_compute_workgroup_size_z,
        max_compute_workgroups_per_dimension,
        max_immediate_size,
        max_non_sampler_bindings,

        max_task_mesh_workgroup_total_count,
        max_task_mesh_workgroups_per_dimension,
        max_task_invocations_per_workgroup,
        max_task_invocations_per_dimension,
        max_mesh_invocations_per_workgroup,
        max_mesh_invocations_per_dimension,
        max_task_payload_size,
        max_mesh_output_vertices,
        max_mesh_output_primitives,
        max_mesh_output_layers,
        max_mesh_multiview_view_count,

        max_blas_primitive_count,
        max_blas_geometry_count,
        max_tlas_instance_count,
        max_acceleration_structures_per_shader_stage,

        max_multiview_view_count,
    } = limits;
    writeln!(output, "\t\t                           Max Texture Dimension 1d: {max_texture_dimension_1d}")?;
    writeln!(output, "\t\t                           Max Texture Dimension 2d: {max_texture_dimension_2d}")?;
    writeln!(output, "\t\t                           Max Texture Dimension 3d: {max_texture_dimension_3d}")?;
    writeln!(output, "\t\t                           Max Texture Array Layers: {max_texture_array_layers}")?;
    writeln!(output, "\t\t                                    Max Bind Groups: {max_bind_groups}")?;
    writeln!(output, "\t\t                        Max Bindings Per Bind Group: {max_bindings_per_bind_group}")?;
    writeln!(output, "\t\t    Max Dynamic Uniform Buffers Per Pipeline Layout: {max_dynamic_uniform_buffers_per_pipeline_layout}")?;
    writeln!(output, "\t\t    Max Dynamic Storage Buffers Per Pipeline Layout: {max_dynamic_storage_buffers_per_pipeline_layout}")?;
    writeln!(output, "\t\t              Max Sampled Textures Per Shader Stage: {max_sampled_textures_per_shader_stage}")?;
    writeln!(output, "\t\t                      Max Samplers Per Shader Stage: {max_samplers_per_shader_stage}")?;
    writeln!(output, "\t\t               Max Storage Buffers Per Shader Stage: {max_storage_buffers_per_shader_stage}")?;
    writeln!(output, "\t\t              Max Storage Textures Per Shader Stage: {max_storage_textures_per_shader_stage}")?;
    writeln!(output, "\t\t               Max Uniform Buffers Per Shader Stage: {max_uniform_buffers_per_shader_stage}")?;
    writeln!(output, "\t\t        Max Binding Array Elements Per Shader Stage: {max_binding_array_elements_per_shader_stage}")?;
    writeln!(output, "\t\tMax Binding Array Sampler Elements Per Shader Stage: {max_binding_array_sampler_elements_per_shader_stage}")?;
    writeln!(output, "\t\t   Max Binding Array AS Elements Per Shader Stage: {max_binding_array_acceleration_structure_elements_per_shader_stage}")?;
    writeln!(output, "\t\t                    Max Uniform Buffer Binding Size: {max_uniform_buffer_binding_size}")?;
    writeln!(output, "\t\t                    Max Storage Buffer Binding Size: {max_storage_buffer_binding_size}")?;
    writeln!(output, "\t\t                                    Max Buffer Size: {max_buffer_size}")?;
    writeln!(output, "\t\t                                 Max Vertex Buffers: {max_vertex_buffers}")?;
    writeln!(output, "\t\t                              Max Vertex Attributes: {max_vertex_attributes}")?;
    writeln!(output, "\t\t                     Max Vertex Buffer Array Stride: {max_vertex_buffer_array_stride}")?;
    writeln!(output, "\t\t                            Max Immediate data Size: {max_immediate_size}")?;
    writeln!(output, "\t\t                   Max Inter-stage Shader Variables: {max_inter_stage_shader_variables}")?;
    writeln!(output, "\t\t                Min Uniform Buffer Offset Alignment: {min_uniform_buffer_offset_alignment}")?;
    writeln!(output, "\t\t                Min Storage Buffer Offset Alignment: {min_storage_buffer_offset_alignment}")?;
    writeln!(output, "\t\t                              Max Color Attachments: {max_color_attachments}")?;
    writeln!(output, "\t\t              Max Color Attachment Bytes per sample: {max_color_attachment_bytes_per_sample}")?;
    writeln!(output, "\t\t                 Max Compute Workgroup Storage Size: {max_compute_workgroup_storage_size}")?;
    writeln!(output, "\t\t              Max Compute Invocations Per Workgroup: {max_compute_invocations_per_workgroup}")?;
    writeln!(output, "\t\t                       Max Compute Workgroup Size X: {max_compute_workgroup_size_x}")?;
    writeln!(output, "\t\t                       Max Compute Workgroup Size Y: {max_compute_workgroup_size_y}")?;
    writeln!(output, "\t\t                       Max Compute Workgroup Size Z: {max_compute_workgroup_size_z}")?;
    writeln!(output, "\t\t               Max Compute Workgroups Per Dimension: {max_compute_workgroups_per_dimension}")?;

    writeln!(output, "\t\t                Max Task/Mesh Workgroup Total Count: {max_task_mesh_workgroup_total_count}")?;
    writeln!(output, "\t\t             Max Task/Mesh Workgroups Per Dimension: {max_task_mesh_workgroups_per_dimension}")?;
    writeln!(output, "\t\t                 Max Task Invocations Per Workgroup: {max_task_invocations_per_workgroup}")?;
    writeln!(output, "\t\t                 Max Task Invocations Per Dimension: {max_task_invocations_per_dimension}")?;
    writeln!(output, "\t\t                 Max Mesh Invocations Per Workgroup: {max_mesh_invocations_per_workgroup}")?;
    writeln!(output, "\t\t                 Max Mesh Invocations Per Dimension: {max_mesh_invocations_per_dimension}")?;

    writeln!(output, "\t\t                              Max Task Payload Size: {max_task_payload_size}")?;
    writeln!(output, "\t\t                           Max Mesh Output Vertices: {max_mesh_output_vertices}")?;
    writeln!(output, "\t\t                         Max Mesh Output Primitives: {max_mesh_output_primitives}")?;
    writeln!(output, "\t\t                             Max Mesh Output Layers: {max_mesh_output_layers}")?;
    writeln!(output, "\t\t                      Max Mesh Multiview View Count: {max_mesh_multiview_view_count}")?;

    writeln!(output, "\t\t                           Max BLAS Primitive count: {max_blas_primitive_count}")?;
    writeln!(output, "\t\t                            Max BLAS Geometry count: {max_blas_geometry_count}")?;
    writeln!(output, "\t\t                            Max TLAS Instance count: {max_tlas_instance_count}")?;
    writeln!(output, "\t\t       Max Acceleration Structures Per Shader Stage: {max_acceleration_structures_per_shader_stage}")?;

    writeln!(output, "\t\t                           Max Multiview View Count: {max_multiview_view_count}")?;
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
    for format in wgpu::TextureFormat::exhaust() {
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

    for format in wgpu::TextureFormat::exhaust() {
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
