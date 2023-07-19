use wgt::Limits;

#[cfg(all(any(feature = "dx11", feature = "dx12"), windows))]
pub(super) mod dxgi;

#[cfg(all(not(target_arch = "wasm32"), feature = "renderdoc"))]
pub(super) mod renderdoc;

pub mod db {
    pub mod amd {
        pub const VENDOR: u32 = 0x1002;
    }
    pub mod apple {
        pub const VENDOR: u32 = 0x106B;
    }
    pub mod arm {
        pub const VENDOR: u32 = 0x13B5;
    }
    pub mod broadcom {
        pub const VENDOR: u32 = 0x14E4;
    }
    pub mod imgtec {
        pub const VENDOR: u32 = 0x1010;
    }
    pub mod intel {
        pub const VENDOR: u32 = 0x8086;
        pub const DEVICE_KABY_LAKE_MASK: u32 = 0x5900;
        pub const DEVICE_SKY_LAKE_MASK: u32 = 0x1900;
    }
    pub mod mesa {
        // Mesa does not actually have a PCI vendor id.
        //
        // To match Vulkan, we use the VkVendorId for Mesa in the gles backend so that lavapipe (Vulkan) and
        // llvmpipe (OpenGL) have the same vendor id.
        pub const VENDOR: u32 = 0x10005;
    }
    pub mod nvidia {
        pub const VENDOR: u32 = 0x10DE;
    }
    pub mod qualcomm {
        pub const VENDOR: u32 = 0x5143;
    }
}

/// Maximum binding size for the shaders that only support `i32` indexing.
/// Interestingly, the index itself can't reach that high, because the minimum
/// element size is 4 bytes, but the compiler toolchain still computes the
/// offset at some intermediate point, internally, as i32.
pub const MAX_I32_BINDING_SIZE: u32 = 1 << 31;

/// Per the [WebGPU spec.]:
///
/// > **_max shader stages per pipeline_** is `2`, because a `GPURenderPipeline` supports both
/// > a vertex and fragment shader.
///
/// [WebGPU spec.]: https://gpuweb.github.io/gpuweb/#max-shader-stages-per-pipeline
//#[cfg(not(target_arch = "wasm32"))]
const MAX_SHADER_STAGES_PER_PIPELINE: u32 = 2;

/// Input for [`max_bindings_per_bind_group`].
pub(crate) struct MaxBindingsPerBindGroupInput {
    pub max_sampled_textures_per_shader_stage: u32,
    pub max_samplers_per_shader_stage: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_storage_textures_per_shader_stage: u32,
    pub max_uniform_buffers_per_shader_stage: u32,
}

/// Calculates the maximum bindings per bind group, according to [this formula from the adapter
/// capabilities guarantees list in the WebGPU spec.]:
///
/// > `maxBindingsPerBindGroup` must be must be ≥ (max bindings per shader stage × max shader
/// > stages per pipeline), where:
/// >
/// > - max bindings per shader stage is (`maxSampledTexturesPerShaderStage` +
/// >   `maxSamplersPerShaderStage` + `maxStorageBuffersPerShaderStage` +
/// >   `maxStorageTexturesPerShaderStage` + `maxUniformBuffersPerShaderStage`).
/// > - max shader stages per pipeline is `2`, because
/// >   a `[GPURenderPipeline](https://gpuweb.github.io/gpuweb/#gpurenderpipeline)` supports both
/// >   a vertex and fragment shader.
///
/// We choose to interpret the above additions as saturating operations. If, for some reason, the
/// output of this formula is <= default, it is clamped to the default.
///
/// See also from the spec.:
///
/// * Documentation for
///   [`maxBindingsPerBindGroup`](https://gpuweb.github.io/gpuweb/#dom-supported-limits-maxbindingsperbindgroup)
/// * [4.2.1 Adapter Capability Guarantees](adapter-cap-guarantees)
///
/// [adapter-cap-guarantees]: https://gpuweb.github.io/gpuweb/#adapter-capability-guarantees
pub(crate) fn max_bindings_per_bind_group(input: MaxBindingsPerBindGroupInput) -> u32 {
    let minimum = Limits::default().max_bindings_per_bind_group;

    let MaxBindingsPerBindGroupInput {
        max_sampled_textures_per_shader_stage,
        max_samplers_per_shader_stage,
        max_storage_buffers_per_shader_stage,
        max_storage_textures_per_shader_stage,
        max_uniform_buffers_per_shader_stage,
    } = input;

    let mut max_bindings_per_bind_group = (max_sampled_textures_per_shader_stage
        .saturating_add(max_samplers_per_shader_stage)
        .saturating_add(max_storage_buffers_per_shader_stage)
        .saturating_add(max_storage_textures_per_shader_stage)
        .saturating_add(max_uniform_buffers_per_shader_stage))
    .saturating_mul(MAX_SHADER_STAGES_PER_PIPELINE);

    if max_bindings_per_bind_group < minimum {
        log::warn!(
            "`max_bindings_per_bind_group` was < 1000, clamping to 1000 to adhere to WebGPU spec."
        );
        max_bindings_per_bind_group = minimum;
    }

    if max_bindings_per_bind_group > minimum {
        // Yes, we're throwing away the calculated value! We're clamping to this value right now
        // because we want to limit exposure to driver bugs, like Vulkan is known to have.
        max_bindings_per_bind_group = minimum;
    }

    max_bindings_per_bind_group
}

pub fn map_naga_stage(stage: naga::ShaderStage) -> wgt::ShaderStages {
    match stage {
        naga::ShaderStage::Vertex => wgt::ShaderStages::VERTEX,
        naga::ShaderStage::Fragment => wgt::ShaderStages::FRAGMENT,
        naga::ShaderStage::Compute => wgt::ShaderStages::COMPUTE,
    }
}

impl crate::CopyExtent {
    pub fn map_extent_to_copy_size(extent: &wgt::Extent3d, dim: wgt::TextureDimension) -> Self {
        Self {
            width: extent.width,
            height: extent.height,
            depth: match dim {
                wgt::TextureDimension::D1 | wgt::TextureDimension::D2 => 1,
                wgt::TextureDimension::D3 => extent.depth_or_array_layers,
            },
        }
    }

    pub fn min(&self, other: &Self) -> Self {
        Self {
            width: self.width.min(other.width),
            height: self.height.min(other.height),
            depth: self.depth.min(other.depth),
        }
    }

    // Get the copy size at a specific mipmap level. This doesn't make most sense,
    // since the copy extents are provided *for* a mipmap level to start with.
    // But backends use `CopyExtent` more sparingly, and this piece is shared.
    pub fn at_mip_level(&self, level: u32) -> Self {
        Self {
            width: (self.width >> level).max(1),
            height: (self.height >> level).max(1),
            depth: (self.depth >> level).max(1),
        }
    }
}

impl crate::TextureCopyBase {
    pub fn max_copy_size(&self, full_size: &crate::CopyExtent) -> crate::CopyExtent {
        let mip = full_size.at_mip_level(self.mip_level);
        crate::CopyExtent {
            width: mip.width - self.origin.x,
            height: mip.height - self.origin.y,
            depth: mip.depth - self.origin.z,
        }
    }
}

impl crate::BufferTextureCopy {
    pub fn clamp_size_to_virtual(&mut self, full_size: &crate::CopyExtent) {
        let max_size = self.texture_base.max_copy_size(full_size);
        self.size = self.size.min(&max_size);
    }
}

impl crate::TextureCopy {
    pub fn clamp_size_to_virtual(
        &mut self,
        full_src_size: &crate::CopyExtent,
        full_dst_size: &crate::CopyExtent,
    ) {
        let max_src_size = self.src_base.max_copy_size(full_src_size);
        let max_dst_size = self.dst_base.max_copy_size(full_dst_size);
        self.size = self.size.min(&max_src_size).min(&max_dst_size);
    }
}

/// Construct a `CStr` from a byte slice, up to the first zero byte.
///
/// Return a `CStr` extending from the start of `bytes` up to and
/// including the first zero byte. If there is no zero byte in
/// `bytes`, return `None`.
///
/// This can be removed when `CStr::from_bytes_until_nul` is stabilized.
/// ([#95027](https://github.com/rust-lang/rust/issues/95027))
#[allow(dead_code)]
pub(crate) fn cstr_from_bytes_until_nul(bytes: &[std::os::raw::c_char]) -> Option<&std::ffi::CStr> {
    if bytes.contains(&0) {
        // Safety for `CStr::from_ptr`:
        // - We've ensured that the slice does contain a null terminator.
        // - The range is valid to read, because the slice covers it.
        // - The memory won't be changed, because the slice borrows it.
        unsafe { Some(std::ffi::CStr::from_ptr(bytes.as_ptr())) }
    } else {
        None
    }
}
