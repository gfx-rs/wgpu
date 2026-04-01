//! Functionality related to device and adapter limits.
//!
//! # Limit Bucketing
//!
//! Web browsers make various information about their operating environment
//! available to content to provide a better experience. For example, content is
//! able to detect whether the device has a touch-screen, in order to provide an
//! appropriate user interface.
//!
//! [Browser fingerprinting][bfp] employs this information for the purpose of
//! constructing a unique "fingerprint" value that is unique to a single browser
//! or shared among a relatively small number of browsers. Fingerprinting can be
//! used for various purposes, including to identify and track users across
//! different websites.
//!
//! Limit bucketing can reduce the ability to fingerprint users based on GPU
//! hardware characteristics when using `wgpu` in applications like a web
//! browser.
//!
//! When limit bucketing is enabled, the adapter limits offered by `wgpu` do not
//! necessarily reflect the exact capabilities of the hardware. Instead, the
//! hardware capabilities are rounded down to one of several pre-defined buckets.
//! The goal of doing this is for there to be enough devices assigned to each
//! bucket that knowledge of which bucket applies is minimally useful for
//! fingerprinting.
//!
//! Limit bucketing may be requested by setting `apply_limit_buckets` in
//! [`wgt::RequestAdapterOptions`] or by setting `apply_limit_buckets` to
//! true when calling [`enumerate_adapters`].
//!
//! If your application does not expose `wgpu` to untrusted content, limit
//! bucketing is not necessary.
//!
//! [bfp]: https://support.mozilla.org/en-US/kb/firefox-protection-against-fingerprinting
//! [`enumerate_adapters`]: `crate::instance::Instance::enumerate_adapters`

use core::mem;

use alloc::{borrow::Cow, vec::Vec};
use thiserror::Error;
use wgt::error::{ErrorType, WebGpuError};
use wgt::{AdapterInfo, AdapterLimitBucketInfo, DeviceType, Features, Limits};

use crate::api_log;

#[derive(Clone, Debug, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[error("Limit '{name}' value {requested} is better than allowed {allowed}")]
pub struct FailedLimit {
    name: Cow<'static, str>,
    requested: u64,
    allowed: u64,
}

impl WebGpuError for FailedLimit {
    fn webgpu_error_type(&self) -> ErrorType {
        ErrorType::Validation
    }
}

pub(crate) fn check_limits(requested: &Limits, allowed: &Limits) -> Vec<FailedLimit> {
    let mut failed = Vec::new();

    requested.check_limits_with_fail_fn(allowed, false, |name, requested, allowed| {
        failed.push(FailedLimit {
            name: Cow::Borrowed(name),
            requested,
            allowed,
        })
    });

    failed
}

/// Fields in [`wgt::AdapterInfo`] relevant to limit bucketing.
pub(crate) struct BucketedAdapterInfo {
    // Equivalent to `adapter.info.device_type == wgt::DeviceType::Cpu`
    is_fallback_adapter: bool,

    subgroup_min_size: u32,
    subgroup_max_size: u32,
}

impl BucketedAdapterInfo {
    const fn defaults() -> Self {
        Self {
            is_fallback_adapter: false,
            subgroup_min_size: 4,
            subgroup_max_size: 128,
        }
    }
}

impl Default for BucketedAdapterInfo {
    fn default() -> Self {
        Self::defaults()
    }
}

pub(crate) struct Bucket {
    name: &'static str,
    limits: Limits,
    info: BucketedAdapterInfo,
    features: Features,
}

impl Bucket {
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Returns `true` if the device having `limits`, `info`, and `features` satisfies
    /// the bucket definition in `self`.
    pub fn is_compatible(&self, limits: &Limits, info: &AdapterInfo, features: Features) -> bool {
        // In the context of limit checks, "allowed" or "available" means
        // what the device supports. If an application requests an
        // unsupported value, the error message might say "limit of {} exceeds
        // allowed value {}". For purposes of bucket compatibility, the bucket values
        // take the place of application-requested values. If the bucket value
        // is beyond what the device supports, then the device does not qualify
        // for that bucket.
        let candidate_is_fallback_adapter = info.device_type == DeviceType::Cpu;

        let failing_limits = check_limits(&self.limits, limits);
        let limits_ok = failing_limits.is_empty();

        if !limits_ok {
            log::debug!("Failing limits: {:#?}", failing_limits);
        }

        let bucket_has_subgroups = self.features.contains(Features::SUBGROUP);
        let subgroups_ok = !bucket_has_subgroups
            || info.subgroup_min_size >= self.info.subgroup_min_size
                && info.subgroup_max_size <= self.info.subgroup_max_size;
        if !subgroups_ok {
            log::debug!(
                "Subgroup min/max {}/{} is not compatible with allowed {}/{}",
                self.info.subgroup_min_size,
                self.info.subgroup_max_size,
                info.subgroup_min_size,
                info.subgroup_max_size,
            );
        }

        let features_ok = features.contains(self.features);
        if !features_ok {
            log::debug!("{:?} are not available", self.features - features);
        }

        limits_ok
            && candidate_is_fallback_adapter == self.info.is_fallback_adapter
            && subgroups_ok
            && features_ok
    }

    pub fn try_apply_to(&self, adapter: &mut hal::DynExposedAdapter) -> bool {
        if !self.is_compatible(
            &adapter.capabilities.limits,
            &adapter.info,
            adapter.features,
        ) {
            log::debug!("bucket `{}` is not compatible", self.name);
            return false;
        }

        let raw_limits = mem::replace(&mut adapter.capabilities.limits, self.limits.clone());

        // Features in EXEMPT_FEATURES are not affected by limit bucketing.
        let exposed_features = adapter
            .features
            .intersection(EXEMPT_FEATURES)
            .union(self.features);
        let raw_features = mem::replace(&mut adapter.features, exposed_features);

        let (bucket_subgroup_min_size, bucket_subgroup_max_size) =
            if self.features.contains(Features::SUBGROUP) {
                (self.info.subgroup_min_size, self.info.subgroup_max_size)
            } else {
                // WebGPU requires that we report these values when subgroups are
                // not supported
                (
                    wgt::MINIMUM_SUBGROUP_MIN_SIZE,
                    wgt::MAXIMUM_SUBGROUP_MAX_SIZE,
                )
            };
        let raw_subgroup_min_size = mem::replace(
            &mut adapter.info.subgroup_min_size,
            bucket_subgroup_min_size,
        );
        let raw_subgroup_max_size = mem::replace(
            &mut adapter.info.subgroup_max_size,
            bucket_subgroup_max_size,
        );

        adapter.info.limit_bucket = Some(AdapterLimitBucketInfo {
            name: Cow::Borrowed(self.name),
            raw_limits,
            raw_features,
            raw_subgroup_min_size,
            raw_subgroup_max_size,
        });

        true
    }
}

/// Apply [limit bucketing][lt] to the adapter limits and features in `raw`.
///
/// Finds a supported bucket and replaces the capabilities with the set defined by
/// the bucket. If no suitable bucket is found, returns `None`, but this should only
/// happen with downlevel devices, and attempting to use limit bucketing with
/// downlevel devices is not recommended.
///
/// [lt]: self#Limit-bucketing
pub fn apply_limit_buckets(mut raw: hal::DynExposedAdapter) -> Option<hal::DynExposedAdapter> {
    for bucket in buckets() {
        if bucket.try_apply_to(&mut raw) {
            let name = bucket.name();
            api_log!("Applied limit bucket `{name}`");
            return Some(raw);
        }
    }
    log::warn!(
        "No suitable limit bucket found for device with {:?}, {:?}, {:?}",
        raw.capabilities.limits,
        raw.info,
        raw.features,
    );
    None
}

/// These features are left alone by limit bucketing. They will be exposed to higher layers
/// whenever the device supports them, and they are not considered when determining bucket
/// compatibility.
///
/// All four features in the list are related to external textures. (The texture format
/// features are used internally by Firefox to support external textures.)
///
/// Handling them this way is a bit of a kludge, but is expected to be a short-term
/// situation only until external texture support is universally available.
///
/// Note that while NV12 and P010 can be hidden from content by excluding them from WebIDL,
/// TEXTURE_FORMATS_16BIT_NORM will eventually be replaced with TEXTURE_FORMATS_TIER1, and
/// at that point neither excluding the tier1 formats from WebIDL entirely nor allowing
/// content to use them on a device that doesn't have the feature enabled will be
/// acceptable. See <https://github.com/gfx-rs/wgpu/issues/8122>.
const EXEMPT_FEATURES: Features = Features::EXTERNAL_TEXTURE
    .union(Features::TEXTURE_FORMAT_NV12)
    .union(Features::TEXTURE_FORMAT_P010)
    .union(Features::TEXTURE_FORMAT_16BIT_NORM);

/// Return the defined adapter feature/limit buckets
///
/// Buckets are not always subsets of preceding buckets, but [`enumerate_adapters`]
/// considers them in the order they are listed here and uses the first bucket satisfied
/// by the device.
///
/// [`enumerate_adapters`]: `crate::instance::Instance::enumerate_adapters`
pub(crate) fn buckets() -> impl Iterator<Item = &'static Bucket> {
    [
        &BUCKET_M1,
        &BUCKET_A2,
        &BUCKET_I1,
        &BUCKET_N1,
        &BUCKET_A1,
        &BUCKET_NO_F16,
        &BUCKET_LLVMPIPE,
        &BUCKET_WARP,
        &BUCKET_DEFAULT,
        &BUCKET_FALLBACK,
    ]
    .iter()
    .copied()
}

// The following limits could be higher for some hardware, but are capped where they
// are to avoid introducing platform or backend dependencies.
//
// **`max_vertex_attributes`:** While there is broad support for 32, Intel hardware with
// Vulkan only supports 29.
// See <https://gitlab.freedesktop.org/mesa/mesa/-/blob/465c186fc5f72c51bda943ac0e19f6512f8e6262/src/intel/vulkan/anv_private.h#L188>.
//
// **`max_dynamic_{storage,uniform}_buffers_per_pipeline_layout`:** These are limited to
// 4 and 8 by DX12.

// UPLEVEL is not a bucket that is actually applied to devices. It serves as a baseline from
// which most of the rest of the buckets are derived. (It could be a real bucket if desired,
// but since UPLEVEL is an intersection across many devices, there is usually a better match
// for any particular device.)
const UPLEVEL: Bucket = Bucket {
    name: "uplevel-defaults",
    limits: Limits {
        max_bind_groups: 8,
        // wgpu does not implement max_bind_groups_plus_vertex_buffers
        // use default max_bindings_per_bind_group
        max_buffer_size: 1 << 30, // 1 GB
        max_color_attachment_bytes_per_sample: 64,
        // use default max_color_attachments
        max_compute_invocations_per_workgroup: 1024,
        max_compute_workgroup_size_x: 1024,
        max_compute_workgroup_size_y: 1024,
        // use default max_compute_workgroup_size_z
        max_compute_workgroup_storage_size: 32 << 10, // 32 kB
        // use default max_compute_workgroups_per_dimension
        // use default max_dynamic_storage_buffers_per_pipeline_layout
        // use default max_dynamic_uniform_buffers_per_pipeline_layout
        max_inter_stage_shader_variables: 28,
        // use default max_sampled_textures_per_shader_stage
        // use default max_samplers_per_shader_stage
        // use default max_storage_buffer_binding_size
        // wgpu does not implement max_storage_buffers_in_fragment_stage: 8,
        // wgpu does not implement max_storage_buffers_in_vertex_stage: 8,
        // use default max_storage_buffers_per_shader_stage
        // wgpu does not implement max_storage_textures_in_fragment_stage: 8,
        // wgpu does not implement max_storage_textures_in_vertex_stage: 8,
        max_storage_textures_per_shader_stage: 8,
        max_texture_array_layers: 2048,
        max_texture_dimension_1d: 16384,
        max_texture_dimension_2d: 16384,
        // use default max_texture_dimension_3d
        // use default max_uniform_buffer_binding_size
        // use default max_uniform_buffers_per_shader_stage
        max_vertex_attributes: 29,
        // use default max_vertex_buffer_array_stride
        // use default max_vertex_buffers
        // use default min_storage_buffer_offset_alignment
        // use default min_uniform_buffer_offset_alignment
        ..Limits::defaults()
    },
    info: BucketedAdapterInfo {
        is_fallback_adapter: false,
        subgroup_min_size: 4,
        subgroup_max_size: 128,
    },
    features: Features::DEPTH_CLIP_CONTROL
        .union(Features::DEPTH32FLOAT_STENCIL8)
        // omit TEXTURE_COMPRESSION_ASTC
        // omit TEXTURE_COMPRESSION_ASTC_SLICED_3D
        .union(Features::TEXTURE_COMPRESSION_BC)
        .union(Features::TEXTURE_COMPRESSION_BC_SLICED_3D)
        // omit TEXTURE_COMPRESSION_ETC2
        .union(Features::TIMESTAMP_QUERY)
        .union(Features::INDIRECT_FIRST_INSTANCE)
        // omit SHADER_F16
        .union(Features::RG11B10UFLOAT_RENDERABLE)
        .union(Features::BGRA8UNORM_STORAGE)
        .union(Features::FLOAT32_FILTERABLE)
        // FLOAT32_BLENDABLE not implemented in wgpu dx12 backend; https://github.com/gfx-rs/wgpu/issues/6555
        // CLIP_DISTANCES not implemented in wgpu dx12 backend; https://github.com/gfx-rs/wgpu/issues/6236
        .union(Features::DUAL_SOURCE_BLENDING)
        // TIER1/TIER2 not implemented in wgpu; https://github.com/gfx-rs/wgpu/issues/8122
        .union(Features::PRIMITIVE_INDEX)
        // TEXTURE_COMPONENT_SWIZZLE not implemented in wgpu; https://github.com/gfx-rs/wgpu/issues/1028
        .union(Features::SUBGROUP)
        .union(Features::IMMEDIATES),
};

// e.g. Apple M Series
const BUCKET_M1: Bucket = Bucket {
    name: "m1",
    limits: Limits {
        max_dynamic_uniform_buffers_per_pipeline_layout: 12,
        max_sampled_textures_per_shader_stage: 48,
        max_storage_buffer_binding_size: 1 << 30, // 1 GB,
        max_vertex_attributes: 31,
        ..UPLEVEL.limits
    },
    info: BucketedAdapterInfo {
        subgroup_min_size: 4,
        subgroup_max_size: 64,
        ..UPLEVEL.info
    },
    features: UPLEVEL
        .features
        .union(Features::TEXTURE_COMPRESSION_ASTC)
        .union(Features::TEXTURE_COMPRESSION_ASTC_SLICED_3D)
        .union(Features::TEXTURE_COMPRESSION_ETC2)
        .union(Features::SHADER_F16)
        .union(Features::FLOAT32_BLENDABLE)
        .union(Features::CLIP_DISTANCES),
};

// e.g. Radeon Vega
const BUCKET_A2: Bucket = Bucket {
    name: "a2",
    limits: Limits {
        max_color_attachment_bytes_per_sample: 128,
        max_compute_workgroup_storage_size: 64 << 10, // 64 kB,
        max_sampled_textures_per_shader_stage: 48,
        max_storage_buffer_binding_size: 1 << 30, // 1 GB,
        max_storage_buffers_per_shader_stage: 16,
        max_vertex_attributes: 32,
        ..UPLEVEL.limits
    },
    info: BucketedAdapterInfo {
        subgroup_min_size: 64,
        subgroup_max_size: 64,
        ..UPLEVEL.info
    },
    features: UPLEVEL.features.union(Features::SHADER_F16),
};

// e.g. Intel Arc, UHD 600 Series, Iris Xe
const BUCKET_I1: Bucket = Bucket {
    name: "i1",
    limits: Limits {
        max_color_attachment_bytes_per_sample: 128,
        max_sampled_textures_per_shader_stage: 48,
        max_storage_buffer_binding_size: 1 << 29, // 512 MB,
        max_storage_buffers_per_shader_stage: 16,
        ..UPLEVEL.limits
    },
    info: BucketedAdapterInfo {
        subgroup_min_size: 8,
        subgroup_max_size: 32,
        ..UPLEVEL.info
    },
    features: UPLEVEL.features.union(Features::SHADER_F16),
};

// e.g. GeForce GTX 1650, GeForce RTX 20, 30, 40, 50 Series
const BUCKET_N1: Bucket = Bucket {
    name: "n1",
    limits: Limits {
        max_color_attachment_bytes_per_sample: 128,
        max_compute_workgroup_storage_size: 48 << 10, // 48 kB,
        max_sampled_textures_per_shader_stage: 48,
        max_storage_buffer_binding_size: 1 << 30, // 1 GB,
        max_storage_buffers_per_shader_stage: 16,
        max_vertex_attributes: 32,
        ..UPLEVEL.limits
    },
    info: BucketedAdapterInfo {
        subgroup_min_size: 32,
        subgroup_max_size: 32,
        ..UPLEVEL.info
    },
    features: UPLEVEL.features.union(Features::SHADER_F16),
};

// e.g. Radeon RX 6000, 7000, 9000 Series
const BUCKET_A1: Bucket = Bucket {
    name: "a1",
    limits: Limits {
        max_color_attachment_bytes_per_sample: 128,
        max_sampled_textures_per_shader_stage: 48,
        max_storage_buffer_binding_size: 1 << 30, // 1 GB,
        max_storage_buffers_per_shader_stage: 16,
        max_vertex_attributes: 32,
        ..UPLEVEL.limits
    },
    info: BucketedAdapterInfo {
        subgroup_min_size: 32,
        subgroup_max_size: 64,
        ..UPLEVEL.info
    },
    features: UPLEVEL.features.union(Features::SHADER_F16),
};

// e.g. GeForce GTX 1050, Radeon WX 5100
const BUCKET_NO_F16: Bucket = Bucket {
    name: "no-f16",
    limits: Limits {
        max_color_attachment_bytes_per_sample: 128,
        max_compute_workgroup_storage_size: 48 << 10, // 48 kB
        max_sampled_textures_per_shader_stage: 48,
        max_storage_buffer_binding_size: 1 << 30, // 1 GB
        max_storage_buffers_per_shader_stage: 16,
        max_vertex_attributes: 32,
        ..UPLEVEL.limits
    },
    info: BucketedAdapterInfo {
        subgroup_min_size: 32,
        subgroup_max_size: 64,
        ..UPLEVEL.info
    },
    features: UPLEVEL.features,
};

const BUCKET_LLVMPIPE: Bucket = Bucket {
    name: "llvmpipe",
    limits: Limits {
        max_color_attachment_bytes_per_sample: 128,
        max_sampled_textures_per_shader_stage: 48,
        max_storage_buffers_per_shader_stage: 16,
        max_vertex_attributes: 32,
        ..UPLEVEL.limits
    },
    info: BucketedAdapterInfo {
        is_fallback_adapter: true,
        subgroup_min_size: 8,
        subgroup_max_size: 8,
    },
    features: UPLEVEL
        .features
        .union(Features::SHADER_F16)
        .union(Features::CLIP_DISTANCES),
};

// a.k.a. Microsoft Basic Render Driver
const BUCKET_WARP: Bucket = Bucket {
    name: "warp",
    limits: Limits {
        max_color_attachment_bytes_per_sample: 128,
        max_sampled_textures_per_shader_stage: 48,
        max_storage_buffers_per_shader_stage: 16,
        max_vertex_attributes: 32,
        ..UPLEVEL.limits
    },
    info: BucketedAdapterInfo {
        is_fallback_adapter: true,
        subgroup_min_size: 4,
        subgroup_max_size: 128,
    },
    features: UPLEVEL.features.union(Features::SHADER_F16),
};

// WebGPU default limits, not a fallback adapter
const BUCKET_DEFAULT: Bucket = Bucket {
    name: "default",
    limits: Limits::defaults(),
    info: BucketedAdapterInfo::defaults(),
    features: Features::empty(),
};

// WebGPU default limits, is a fallback adapter
const BUCKET_FALLBACK: Bucket = Bucket {
    name: "fallback",
    limits: Limits::defaults(),
    info: BucketedAdapterInfo {
        is_fallback_adapter: true,
        ..BucketedAdapterInfo::defaults()
    },
    features: Features::empty(),
};

#[cfg(test)]
mod tests {
    use super::*;
    use wgt::Features;

    #[test]
    fn enumerate_webgpu_features() {
        let difference = Features::all_webgpu_mask().difference(
            Features::DEPTH_CLIP_CONTROL
                .union(Features::DEPTH32FLOAT_STENCIL8)
                .union(Features::TEXTURE_COMPRESSION_ASTC)
                .union(Features::TEXTURE_COMPRESSION_ASTC_SLICED_3D)
                .union(Features::TEXTURE_COMPRESSION_BC)
                .union(Features::TEXTURE_COMPRESSION_BC_SLICED_3D)
                .union(Features::TEXTURE_COMPRESSION_ETC2)
                .union(Features::TIMESTAMP_QUERY)
                .union(Features::INDIRECT_FIRST_INSTANCE)
                .union(Features::SHADER_F16)
                .union(Features::RG11B10UFLOAT_RENDERABLE)
                .union(Features::BGRA8UNORM_STORAGE)
                .union(Features::FLOAT32_FILTERABLE)
                .union(Features::FLOAT32_BLENDABLE)
                .union(Features::CLIP_DISTANCES)
                .union(Features::DUAL_SOURCE_BLENDING)
                .union(Features::SUBGROUP)
                //.union(Features::TEXTURE_FORMATS_TIER1) not implemented
                //.union(Features::TEXTURE_FORMATS_TIER2) not implemented
                .union(Features::PRIMITIVE_INDEX)
                //.union(Features::TEXTURE_COMPONENT_SWIZZLE) not implemented
                // Standard-track features not in official spec
                .union(Features::IMMEDIATES),
        );
        assert!(
            difference.is_empty(),
            "New WebGPU features should be assigned to appropriate limit buckets; missing {difference:?}"
        );
    }

    #[test]
    fn relationships() {
        // Check that each bucket is a superset of UPLEVEL, ignoring the `is_fallback_adapter` flag.
        for bucket in [
            &BUCKET_M1,
            &BUCKET_A2,
            &BUCKET_I1,
            &BUCKET_N1,
            &BUCKET_A1,
            &BUCKET_NO_F16,
            &BUCKET_WARP,
            &BUCKET_LLVMPIPE,
        ] {
            let info = AdapterInfo {
                subgroup_min_size: bucket.info.subgroup_min_size,
                subgroup_max_size: bucket.info.subgroup_max_size,
                ..AdapterInfo::new(
                    DeviceType::DiscreteGpu, // not a fallback adapter
                    wgt::Backend::Noop,
                )
            };
            assert!(
                UPLEVEL.is_compatible(&bucket.limits, &info, bucket.features),
                "Bucket `{}` should be a superset of UPLEVEL",
                bucket.name(),
            );
        }
    }
}
