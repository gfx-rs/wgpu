#[cfg(dx12)]
pub(super) mod dxgi;

#[cfg(all(native, feature = "renderdoc"))]
pub(super) mod renderdoc;

pub mod db {
    pub mod amd {
        /// cbindgen:ignore
        pub const VENDOR: u32 = 0x1002;
    }
    pub mod apple {
        /// cbindgen:ignore
        pub const VENDOR: u32 = 0x106B;
    }
    pub mod arm {
        /// cbindgen:ignore
        pub const VENDOR: u32 = 0x13B5;
    }
    pub mod broadcom {
        /// cbindgen:ignore
        pub const VENDOR: u32 = 0x14E4;
    }
    pub mod imgtec {
        /// cbindgen:ignore
        pub const VENDOR: u32 = 0x1010;
    }
    pub mod intel {
        /// cbindgen:ignore
        pub const VENDOR: u32 = 0x8086;
        pub const DEVICE_KABY_LAKE_MASK: u32 = 0x5900;
        pub const DEVICE_SKY_LAKE_MASK: u32 = 0x1900;
    }
    pub mod mesa {
        // Mesa does not actually have a PCI vendor id.
        //
        // To match Vulkan, we use the VkVendorId for Mesa in the gles backend so that lavapipe (Vulkan) and
        // llvmpipe (OpenGL) have the same vendor id.
        /// cbindgen:ignore
        pub const VENDOR: u32 = 0x10005;
    }
    pub mod nvidia {
        /// cbindgen:ignore
        pub const VENDOR: u32 = 0x10DE;
    }
    pub mod qualcomm {
        /// cbindgen:ignore
        pub const VENDOR: u32 = 0x5143;
    }
}

/// Maximum binding size for the shaders that only support `i32` indexing.
/// Interestingly, the index itself can't reach that high, because the minimum
/// element size is 4 bytes, but the compiler toolchain still computes the
/// offset at some intermediate point, internally, as i32.
pub const MAX_I32_BINDING_SIZE: u32 = (1 << 31) - 1;

pub fn map_naga_stage(stage: naga::ShaderStage) -> wgt::ShaderStages {
    match stage {
        naga::ShaderStage::Vertex => wgt::ShaderStages::VERTEX,
        naga::ShaderStage::Fragment => wgt::ShaderStages::FRAGMENT,
        naga::ShaderStage::Compute => wgt::ShaderStages::COMPUTE,
        naga::ShaderStage::Task => wgt::ShaderStages::TASK,
        naga::ShaderStage::Mesh => wgt::ShaderStages::MESH,
        naga::ShaderStage::RayGeneration => wgt::ShaderStages::RAY_GENERATION,
        naga::ShaderStage::AnyHit => wgt::ShaderStages::ANY_HIT,
        naga::ShaderStage::ClosestHit => wgt::ShaderStages::CLOSEST_HIT,
        naga::ShaderStage::Miss => wgt::ShaderStages::MISS,
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

/// Clamp the limits in `limits` to honor HAL-imposed maximums and WebGPU
/// alignment requirements.
///
/// Other limits are left unchanged.
#[cfg_attr(not(any_backend), allow(dead_code))]
pub(crate) fn apply_hal_limits(mut limits: wgt::Limits) -> wgt::Limits {
    // The Metal backend maintains two copies of many limit values (one as
    // `wgt::Limits` and one as `metal::PrivateCapabilities`). In order to avoid
    // confusing discrepancies between the two, some of the logic here is
    // duplicated in the initialization of `metal::PrivateCapabilities`.
    // See <https://github.com/gfx-rs/wgpu/issues/8715>.

    limits.max_bind_groups = limits.max_bind_groups.min(crate::MAX_BIND_GROUPS as u32);
    limits.max_vertex_buffers = limits
        .max_vertex_buffers
        .min(crate::MAX_VERTEX_BUFFERS as u32);
    limits.max_color_attachments = limits
        .max_color_attachments
        .min(crate::MAX_COLOR_ATTACHMENTS as u32);

    // Round some limits down to the WebGPU alignment requirement, to avoid
    // suggesting values that won't work. (In particular, the CTS queries limits
    // and then tests the exact limit value.)
    limits.max_storage_buffer_binding_size &= !(wgt::STORAGE_BINDING_SIZE_ALIGNMENT - 1);
    limits.max_vertex_buffer_array_stride &= !(wgt::VERTEX_ALIGNMENT as u32 - 1);

    limits
}
