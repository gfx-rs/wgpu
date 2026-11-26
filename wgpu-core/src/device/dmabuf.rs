//! Importing a Linux DMA-buf as a read-only texture, without `unsafe`.
//!
//! Every precondition of hal's `texture_from_dmabuf_fd` is checked here or in
//! hal before Vulkan sees the descriptor: the device's features, that the fd
//! is a DMA-buf, the format, and that the plane lies inside the buffer. The
//! texture is read only, so the GPU never writes memory that its producer may
//! map; and its tracker state is foreign-owned, so each command buffer that
//! uses it acquires it from the producer and releases it back.

use alloc::{borrow::Cow, string::String, sync::Arc, vec::Vec};
use std::os::fd::{AsRawFd, OwnedFd};

use thiserror::Error;

use crate::{
    conv,
    device::{Device, DeviceError, MissingFeatures},
    resource::{self, Texture, TextureOwnership},
};

/// The formats a DMA-buf plane can be imported as: one plane each, colour,
/// uncompressed. NV12 is an R8 plane and an Rg8 plane; P010 is R16 and Rg16;
/// YUYV is Rgba8 at half width.
pub const DMABUF_FORMATS: &[wgt::TextureFormat] = &[
    wgt::TextureFormat::R8Unorm,
    wgt::TextureFormat::Rg8Unorm,
    wgt::TextureFormat::Rgba8Unorm,
    wgt::TextureFormat::Bgra8Unorm,
    wgt::TextureFormat::R16Unorm,
    wgt::TextureFormat::Rg16Unorm,
];

/// The DRM format modifier for a linear layout, rows one after another.
const DRM_FORMAT_MOD_LINEAR: u64 = 0;

/// One linear plane of a DMA-buf, to import as a 2D texture.
#[derive(Clone, Debug)]
pub struct DmabufTextureDescriptor<'a> {
    /// Debug label of the texture.
    pub label: Option<Cow<'a, str>>,
    /// The plane's width in texels.
    pub width: u32,
    /// The plane's height in texels.
    pub height: u32,
    /// The plane's format, one of [`DMABUF_FORMATS`].
    pub format: wgt::TextureFormat,
    /// Where the plane starts in the DMA-buf, in bytes.
    pub offset: u64,
    /// From one row's start to the next, in bytes.
    pub pitch: u64,
}

/// Why a DMA-buf could not be imported.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum ImportDmabufError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error("DMA-buf import needs the Vulkan backend")]
    NotVulkan,
    #[error("The Vulkan device lacks {}", .0.to_string_lossy())]
    MissingExtension(&'static core::ffi::CStr),
    #[error("The file descriptor is not a DMA-buf")]
    NotDmabuf,
    #[error("{0:?} is not a format a DMA-buf plane can be imported as")]
    UnsupportedFormat(wgt::TextureFormat),
    #[error("A {width}x{height} plane is empty or larger than the device allows")]
    InvalidSize { width: u32, height: u32 },
    #[error("A pitch of {pitch} bytes is shorter than a row of {row} bytes")]
    PitchTooSmall { pitch: u64, row: u64 },
    #[error("The plane needs {needed} bytes but the DMA-buf holds {size}")]
    TooSmall { needed: u64, size: u64 },
    #[error("The driver refused the plane layout (offset {offset}, pitch {pitch})")]
    PlaneLayoutRefused { offset: u64, pitch: u64 },
    #[error("The driver can't import {0:?} from a linear DMA-buf")]
    DriverUnsupported(wgt::TextureFormat),
    #[error("No memory type fits both the image and the DMA-buf")]
    NoCompatibleMemoryType,
}

impl From<hal::vulkan::ImportTextureError> for ImportDmabufError {
    fn from(error: hal::vulkan::ImportTextureError) -> Self {
        use hal::vulkan::ImportTextureError as E;
        match error {
            E::DeviceError(e) => Self::Device(DeviceError::from_hal(e)),
            E::MissingExtension(extension) => Self::MissingExtension(extension),
            E::UnsupportedFormat { format, .. } => Self::DriverUnsupported(format),
            E::PlaneLayoutRefused { offset, row_pitch } => Self::PlaneLayoutRefused {
                offset,
                pitch: row_pitch,
            },
            E::TooSmall { needed, size } => Self::TooSmall { needed, size },
            E::NoCompatibleMemoryType { .. } => Self::NoCompatibleMemoryType,
        }
    }
}

/// The size of the DMA-buf `fd`, from the kernel's `fdinfo` for it; `None`
/// when `fd` is not a DMA-buf, or `/proc` can't say.
///
/// A DMA-buf's `fdinfo` carries `size:` and `exp_name:` lines
/// (`dma_buf_show_fdinfo`), and its `/proc/self/fd` link names `/dmabuf:`.
/// `fd` stays open while both are read, so its number names this file
/// throughout.
fn dmabuf_size(fd: &OwnedFd) -> Option<u64> {
    let number = fd.as_raw_fd();
    let link = std::fs::read_link(alloc::format!("/proc/self/fd/{number}")).ok()?;
    if !link.to_string_lossy().starts_with("/dmabuf:") {
        return None;
    }
    let info: String =
        std::fs::read_to_string(alloc::format!("/proc/self/fdinfo/{number}")).ok()?;
    let field = |name: &str| {
        info.lines()
            .find_map(|line| line.strip_prefix(name))
            .map(str::trim)
    };
    field("exp_name:")?;
    field("size:")?.parse().ok()
}

impl Device {
    /// Import one linear plane of the DMA-buf `fd` as a texture that can be
    /// sampled and copied from, never written.
    ///
    /// The texture's contents belong to the DMA-buf's producer. Each command
    /// buffer that uses the texture acquires it from the producer first and
    /// releases it back after, so the producer may write between submissions;
    /// one that writes while the GPU reads gives a torn image, never undefined
    /// behaviour. Wait for the submission to finish before giving the buffer
    /// back to its producer.
    ///
    /// On success the device owns `fd`; on failure it is closed.
    pub fn import_dmabuf_texture(
        self: &Arc<Self>,
        fd: OwnedFd,
        desc: &DmabufTextureDescriptor,
    ) -> Result<Arc<Texture>, ImportDmabufError> {
        self.check_is_valid()?;
        let vulkan = self
            .raw()
            .as_any()
            .downcast_ref::<hal::vulkan::Device>()
            .ok_or(ImportDmabufError::NotVulkan)?;
        if let Some(extension) = vulkan.missing_dmabuf_extension() {
            return Err(ImportDmabufError::MissingExtension(extension));
        }

        let format = desc.format;
        if !DMABUF_FORMATS.contains(&format) {
            return Err(ImportDmabufError::UnsupportedFormat(format));
        }
        let (width, height) = (desc.width, desc.height);
        let max = self.limits.max_texture_dimension_2d;
        if width == 0 || height == 0 || width > max || height > max {
            return Err(ImportDmabufError::InvalidSize { width, height });
        }
        let texel = u64::from(
            format
                .block_copy_size(None)
                .ok_or(ImportDmabufError::UnsupportedFormat(format))?,
        );
        let row = u64::from(width) * texel;
        if desc.pitch < row {
            return Err(ImportDmabufError::PitchTooSmall {
                pitch: desc.pitch,
                row,
            });
        }

        let size = dmabuf_size(&fd).ok_or(ImportDmabufError::NotDmabuf)?;
        let needed = desc
            .pitch
            .checked_mul(u64::from(height) - 1)
            .and_then(|rows| rows.checked_add(row))
            .and_then(|plane| plane.checked_add(desc.offset))
            .unwrap_or(u64::MAX);
        if needed > size {
            return Err(ImportDmabufError::TooSmall { needed, size });
        }

        let usage = wgt::TextureUsages::TEXTURE_BINDING | wgt::TextureUsages::COPY_SRC;
        let texture_desc = resource::TextureDescriptor {
            label: desc.label.clone(),
            size: wgt::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgt::TextureDimension::D2,
            format,
            usage,
            view_formats: Vec::new(),
            texture_binding_view_dimension: None,
        };
        let format_features = self.describe_format_features(format)?;
        let hal_usage = conv::map_texture_usage(usage, format.into(), format_features.flags);
        let hal_desc = hal::TextureDescriptor {
            label: desc.label.as_deref(),
            size: texture_desc.size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgt::TextureDimension::D2,
            format,
            usage: hal_usage,
            memory_flags: hal::MemoryFlags::empty(),
            view_formats: Vec::new(),
            texture_binding_view_dimension: None,
        };

        // SAFETY: `fd` is a DMA-buf (`dmabuf_size` above); `hal_desc` is 2D
        // with one mip level, one layer and one sample; and the texture is
        // marked foreign below, so each use acquires it from the producer and
        // releases it after.
        let raw = unsafe {
            vulkan.texture_from_dmabuf_fd(
                fd,
                &hal_desc,
                DRM_FORMAT_MOD_LINEAR,
                desc.pitch,
                desc.offset,
            )
        }?;

        let ownership = TextureOwnership::Foreign {
            queue_family: vulkan.queue_family_index(),
        };
        // Cleared: the contents are the producer's, never zeroed by wgpu.
        self.create_texture_from_hal_inner(
            alloc::boxed::Box::new(raw),
            &texture_desc,
            TextureOwnership::FOREIGN_RESTING,
            true,
            ownership,
        )
        .map_err(|error| match error {
            resource::CreateTextureError::Device(e) => ImportDmabufError::Device(e),
            resource::CreateTextureError::MissingFeatures(_, e) => {
                ImportDmabufError::MissingFeatures(e)
            }
            _ => ImportDmabufError::UnsupportedFormat(format),
        })
    }
}
