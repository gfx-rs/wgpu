#![allow(unused_variables)]

#[cfg(not(target_arch = "wasm32"))]
mod egl;

mod adapter;
mod conv;
mod device;

#[cfg(not(target_arch = "wasm32"))]
use self::egl::{Instance, Surface};

use glow::HasContext;

use std::{ops::Range, sync::Arc};

#[derive(Clone)]
pub struct Api;
pub struct Context;
pub struct Encoder;
#[derive(Debug)]
pub struct Resource;

type DeviceResult<T> = Result<T, crate::DeviceError>;

impl crate::Api for Api {
    type Instance = Instance;
    type Surface = Surface;
    type Adapter = Adapter;
    type Device = Device;

    type Queue = Queue;
    type CommandEncoder = Encoder;
    type CommandBuffer = Resource;

    type Buffer = Buffer;
    type Texture = Texture;
    type SurfaceTexture = Texture;
    type TextureView = TextureView;
    type Sampler = Sampler;
    type QuerySet = Resource;
    type Fence = Resource;

    type BindGroupLayout = Resource;
    type BindGroup = Resource;
    type PipelineLayout = Resource;
    type ShaderModule = Resource;
    type RenderPipeline = Resource;
    type ComputePipeline = Resource;
}

bitflags::bitflags! {
    /// Flags that affect internal code paths but do not
    /// change the exposed feature set.
    struct PrivateCapability: u32 {
        /// Support explicit layouts in shader.
        const EXPLICIT_LAYOUTS_IN_SHADER = 0x00002000;
    }
}

type BindTarget = u32;
type TextureFormat = u32;

trait Sampled {
    unsafe fn set_param_float(&self, gl: &glow::Context, key: u32, value: f32);
    // see https://github.com/grovesNL/glow/issues/170
    unsafe fn set_param_float_vec(&self, gl: &glow::Context, key: u32, values: &mut [f32]);
    unsafe fn set_param_int(&self, gl: &glow::Context, key: u32, value: i32);

    unsafe fn configure_sampling(&self, gl: &glow::Context, desc: &crate::SamplerDescriptor) {
        let (min, mag) =
            conv::map_filter_modes(desc.min_filter, desc.mag_filter, desc.mipmap_filter);

        self.set_param_int(gl, glow::TEXTURE_MIN_FILTER, min as i32);
        self.set_param_int(gl, glow::TEXTURE_MAG_FILTER, mag as i32);

        self.set_param_int(
            gl,
            glow::TEXTURE_WRAP_S,
            conv::map_address_mode(desc.address_modes[0]) as i32,
        );
        self.set_param_int(
            gl,
            glow::TEXTURE_WRAP_T,
            conv::map_address_mode(desc.address_modes[1]) as i32,
        );
        self.set_param_int(
            gl,
            glow::TEXTURE_WRAP_R,
            conv::map_address_mode(desc.address_modes[2]) as i32,
        );

        if let Some(border_color) = desc.border_color {
            let mut border = match border_color {
                wgt::SamplerBorderColor::TransparentBlack => [0.0; 4],
                wgt::SamplerBorderColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
                wgt::SamplerBorderColor::OpaqueWhite => [1.0; 4],
            };
            self.set_param_float_vec(gl, glow::TEXTURE_BORDER_COLOR, &mut border);
        }

        if let Some(ref range) = desc.lod_clamp {
            self.set_param_float(gl, glow::TEXTURE_MIN_LOD, range.start);
            self.set_param_float(gl, glow::TEXTURE_MAX_LOD, range.end);
        }

        //TODO: `desc.anisotropy_clamp` depends on the downlevel flag
        // self.set_param_float(glow::TEXTURE_MAX_ANISOTROPY, aniso as f32);

        //set_param_float(glow::TEXTURE_LOD_BIAS, info.lod_bias.0);

        if let Some(compare) = desc.compare {
            self.set_param_int(
                gl,
                glow::TEXTURE_COMPARE_MODE,
                glow::COMPARE_REF_TO_TEXTURE as i32,
            );
            self.set_param_int(
                gl,
                glow::TEXTURE_COMPARE_FUNC,
                conv::map_compare_func(compare) as i32,
            );
        }
    }
}

struct SamplerBinding(glow::Sampler);
impl Sampled for SamplerBinding {
    unsafe fn set_param_float(&self, gl: &glow::Context, key: u32, value: f32) {
        gl.sampler_parameter_f32(self.0, key, value);
    }
    unsafe fn set_param_float_vec(&self, gl: &glow::Context, key: u32, values: &mut [f32]) {
        gl.sampler_parameter_f32_slice(self.0, key, values);
    }
    unsafe fn set_param_int(&self, gl: &glow::Context, key: u32, value: i32) {
        gl.sampler_parameter_i32(self.0, key, value);
    }
}

struct SampledTextureBinding(BindTarget);
impl Sampled for SampledTextureBinding {
    unsafe fn set_param_float(&self, gl: &glow::Context, key: u32, value: f32) {
        gl.tex_parameter_f32(self.0, key, value);
    }
    unsafe fn set_param_float_vec(&self, gl: &glow::Context, key: u32, values: &mut [f32]) {
        gl.tex_parameter_f32_slice(self.0, key, values);
    }
    unsafe fn set_param_int(&self, gl: &glow::Context, key: u32, value: i32) {
        gl.tex_parameter_i32(self.0, key, value);
    }
}

#[derive(Debug, Clone, Copy)]
enum VertexAttribKind {
    Float,   // glVertexAttribPointer
    Integer, // glVertexAttribIPointer
    Double,  // glVertexAttribLPointer
}

struct FormatDescription {
    tex_internal: u32,
    tex_external: u32,
    data_type: u32,
    num_components: u8,
    va_kind: VertexAttribKind,
}

struct AdapterShared {
    context: glow::Context,
    private_caps: PrivateCapability,
}

pub struct Adapter {
    shared: Arc<AdapterShared>,
}

pub struct Device {
    shared: Arc<AdapterShared>,
    main_vao: glow::VertexArray,
}

pub struct Queue {
    shared: Arc<AdapterShared>,
    features: wgt::Features,
}

#[derive(Debug)]
pub struct Buffer {
    raw: glow::Buffer,
    target: BindTarget,
    map_flags: u32,
}

#[derive(Debug)]
pub enum Texture {
    Renderbuffer {
        raw: glow::Renderbuffer,
        aspects: crate::FormatAspect,
    },
    Texture {
        raw: glow::Texture,
        target: BindTarget,
    },
}

#[derive(Debug)]
pub enum TextureView {
    Renderbuffer {
        raw: glow::Renderbuffer,
        aspects: crate::FormatAspect,
    },
    Texture {
        raw: glow::Texture,
        target: BindTarget,
        range: wgt::ImageSubresourceRange,
    },
}

#[derive(Debug)]
pub struct Sampler {
    raw: glow::Sampler,
}

impl crate::Queue<Api> for Queue {
    unsafe fn submit(
        &mut self,
        command_buffers: &[&Resource],
        signal_fence: Option<(&mut Resource, crate::FenceValue)>,
    ) -> DeviceResult<()> {
        Ok(())
    }
    unsafe fn present(
        &mut self,
        surface: &mut Surface,
        texture: Texture,
    ) -> Result<(), crate::SurfaceError> {
        Ok(())
    }
}

impl crate::CommandEncoder<Api> for Encoder {
    unsafe fn begin_encoding(&mut self, label: crate::Label) -> DeviceResult<()> {
        Ok(())
    }
    unsafe fn discard_encoding(&mut self) {}
    unsafe fn end_encoding(&mut self) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn reset_all<I>(&mut self, command_buffers: I) {}

    unsafe fn transition_buffers<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::BufferBarrier<'a, Api>>,
    {
    }

    unsafe fn transition_textures<'a, T>(&mut self, barriers: T)
    where
        T: Iterator<Item = crate::TextureBarrier<'a, Api>>,
    {
    }

    unsafe fn fill_buffer(&mut self, buffer: &Buffer, range: crate::MemoryRange, value: u8) {}

    unsafe fn copy_buffer_to_buffer<T>(&mut self, src: &Buffer, dst: &Buffer, regions: T) {}

    unsafe fn copy_texture_to_texture<T>(
        &mut self,
        src: &Texture,
        src_usage: crate::TextureUse,
        dst: &Texture,
        regions: T,
    ) {
    }

    unsafe fn copy_buffer_to_texture<T>(&mut self, src: &Buffer, dst: &Texture, regions: T) {}

    unsafe fn copy_texture_to_buffer<T>(
        &mut self,
        src: &Texture,
        src_usage: crate::TextureUse,
        dst: &Buffer,
        regions: T,
    ) {
    }

    unsafe fn begin_query(&mut self, set: &Resource, index: u32) {}
    unsafe fn end_query(&mut self, set: &Resource, index: u32) {}
    unsafe fn write_timestamp(&mut self, set: &Resource, index: u32) {}
    unsafe fn reset_queries(&mut self, set: &Resource, range: Range<u32>) {}
    unsafe fn copy_query_results(
        &mut self,
        set: &Resource,
        range: Range<u32>,
        buffer: &Buffer,
        offset: wgt::BufferAddress,
    ) {
    }

    // render

    unsafe fn begin_render_pass(&mut self, desc: &crate::RenderPassDescriptor<Api>) {}
    unsafe fn end_render_pass(&mut self) {}

    unsafe fn set_bind_group(
        &mut self,
        layout: &Resource,
        index: u32,
        group: &Resource,
        dynamic_offsets: &[wgt::DynamicOffset],
    ) {
    }
    unsafe fn set_push_constants(
        &mut self,
        layout: &Resource,
        stages: wgt::ShaderStage,
        offset: u32,
        data: &[u32],
    ) {
    }

    unsafe fn insert_debug_marker(&mut self, label: &str) {}
    unsafe fn begin_debug_marker(&mut self, group_label: &str) {}
    unsafe fn end_debug_marker(&mut self) {}

    unsafe fn set_render_pipeline(&mut self, pipeline: &Resource) {}

    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: crate::BufferBinding<'a, Api>,
        format: wgt::IndexFormat,
    ) {
    }
    unsafe fn set_vertex_buffer<'a>(&mut self, index: u32, binding: crate::BufferBinding<'a, Api>) {
    }
    unsafe fn set_viewport(&mut self, rect: &crate::Rect<f32>, depth_range: Range<f32>) {}
    unsafe fn set_scissor_rect(&mut self, rect: &crate::Rect<u32>) {}
    unsafe fn set_stencil_reference(&mut self, value: u32) {}
    unsafe fn set_blend_constants(&mut self, color: &wgt::Color) {}

    unsafe fn draw(
        &mut self,
        start_vertex: u32,
        vertex_count: u32,
        start_instance: u32,
        instance_count: u32,
    ) {
    }
    unsafe fn draw_indexed(
        &mut self,
        start_index: u32,
        index_count: u32,
        base_vertex: i32,
        start_instance: u32,
        instance_count: u32,
    ) {
    }
    unsafe fn draw_indirect(
        &mut self,
        buffer: &Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
    }
    unsafe fn draw_indexed_indirect(
        &mut self,
        buffer: &Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) {
    }
    unsafe fn draw_indirect_count(
        &mut self,
        buffer: &Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
    }
    unsafe fn draw_indexed_indirect_count(
        &mut self,
        buffer: &Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
    }

    // compute

    unsafe fn begin_compute_pass(&mut self, desc: &crate::ComputePassDescriptor) {}
    unsafe fn end_compute_pass(&mut self) {}

    unsafe fn set_compute_pipeline(&mut self, pipeline: &Resource) {}

    unsafe fn dispatch(&mut self, count: [u32; 3]) {}
    unsafe fn dispatch_indirect(&mut self, buffer: &Buffer, offset: wgt::BufferAddress) {}
}
