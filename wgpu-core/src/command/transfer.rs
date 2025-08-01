use alloc::{format, string::String, sync::Arc, vec::Vec};

use arrayvec::ArrayVec;
use thiserror::Error;
use wgt::{
    error::{ErrorType, WebGpuError},
    BufferAddress, BufferTextureCopyInfoError, BufferUsages, Extent3d, TextureSelector,
    TextureUsages,
};

#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;
use crate::{
    api_log,
    command::{clear_texture, CommandEncoderError, EncoderStateError},
    conv,
    device::{Device, MissingDownlevelFlags},
    global::Global,
    id::{BufferId, CommandEncoderId, TextureId},
    init_tracker::{
        has_copy_partial_init_tracker_coverage, MemoryInitKind, TextureInitRange,
        TextureInitTrackerAction,
    },
    resource::{
        MissingBufferUsageError, MissingTextureUsageError, ParentDevice, RawResourceAccess,
        Texture, TextureErrorDimension,
    },
    snatch::SnatchGuard,
};

use super::{ClearError, CommandBufferMutable};

pub type TexelCopyBufferInfo = wgt::TexelCopyBufferInfo<BufferId>;
pub type TexelCopyTextureInfo = wgt::TexelCopyTextureInfo<TextureId>;
pub type CopyExternalImageDestInfo = wgt::CopyExternalImageDestInfo<TextureId>;

#[derive(Clone, Copy, Debug)]
pub enum CopySide {
    Source,
    Destination,
}

/// Error encountered while attempting a data transfer.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TransferError {
    #[error("Source and destination cannot be the same buffer")]
    SameSourceDestinationBuffer,
    #[error(transparent)]
    MissingBufferUsage(#[from] MissingBufferUsageError),
    #[error(transparent)]
    MissingTextureUsage(#[from] MissingTextureUsageError),
    #[error("Copy of {start_offset}..{end_offset} would end up overrunning the bounds of the {side:?} buffer of size {buffer_size}")]
    BufferOverrun {
        start_offset: BufferAddress,
        end_offset: BufferAddress,
        buffer_size: BufferAddress,
        side: CopySide,
    },
    #[error("Copy of {dimension:?} {start_offset}..{end_offset} would end up overrunning the bounds of the {side:?} texture of {dimension:?} size {texture_size}")]
    TextureOverrun {
        start_offset: u32,
        end_offset: u32,
        texture_size: u32,
        dimension: TextureErrorDimension,
        side: CopySide,
    },
    #[error("Partial copy of {start_offset}..{end_offset} on {dimension:?} dimension with size {texture_size} \
             is not supported for the {side:?} texture format {format:?} with {sample_count} samples")]
    UnsupportedPartialTransfer {
        format: wgt::TextureFormat,
        sample_count: u32,
        start_offset: u32,
        end_offset: u32,
        texture_size: u32,
        dimension: TextureErrorDimension,
        side: CopySide,
    },
    #[error(
        "Copying{} layers {}..{} to{} layers {}..{} of the same texture is not allowed",
        if *src_aspects == wgt::TextureAspect::All { String::new() } else { format!(" {src_aspects:?}") },
        src_origin_z,
        src_origin_z + array_layer_count,
        if *dst_aspects == wgt::TextureAspect::All { String::new() } else { format!(" {dst_aspects:?}") },
        dst_origin_z,
        dst_origin_z + array_layer_count,
    )]
    InvalidCopyWithinSameTexture {
        src_aspects: wgt::TextureAspect,
        dst_aspects: wgt::TextureAspect,
        src_origin_z: u32,
        dst_origin_z: u32,
        array_layer_count: u32,
    },
    #[error("Unable to select texture aspect {aspect:?} from format {format:?}")]
    InvalidTextureAspect {
        format: wgt::TextureFormat,
        aspect: wgt::TextureAspect,
    },
    #[error("Unable to select texture mip level {level} out of {total}")]
    InvalidTextureMipLevel { level: u32, total: u32 },
    #[error("Texture dimension must be 2D when copying from an external texture")]
    InvalidDimensionExternal,
    #[error("Buffer offset {0} is not aligned to block size or `COPY_BUFFER_ALIGNMENT`")]
    UnalignedBufferOffset(BufferAddress),
    #[error("Copy size {0} does not respect `COPY_BUFFER_ALIGNMENT`")]
    UnalignedCopySize(BufferAddress),
    #[error("Copy width is not a multiple of block width")]
    UnalignedCopyWidth,
    #[error("Copy height is not a multiple of block height")]
    UnalignedCopyHeight,
    #[error("Copy origin's x component is not a multiple of block width")]
    UnalignedCopyOriginX,
    #[error("Copy origin's y component is not a multiple of block height")]
    UnalignedCopyOriginY,
    #[error("Bytes per row does not respect `COPY_BYTES_PER_ROW_ALIGNMENT`")]
    UnalignedBytesPerRow,
    #[error("Number of bytes per row needs to be specified since more than one row is copied")]
    UnspecifiedBytesPerRow,
    #[error("Number of rows per image needs to be specified since more than one image is copied")]
    UnspecifiedRowsPerImage,
    #[error("Number of bytes per row is less than the number of bytes in a complete row")]
    InvalidBytesPerRow,
    #[error("Number of rows per image is invalid")]
    InvalidRowsPerImage,
    #[error("Overflow while computing the size of the copy")]
    SizeOverflow,
    #[error("Copy source aspects must refer to all aspects of the source texture format")]
    CopySrcMissingAspects,
    #[error(
        "Copy destination aspects must refer to all aspects of the destination texture format"
    )]
    CopyDstMissingAspects,
    #[error("Copy aspect must refer to a single aspect of texture format")]
    CopyAspectNotOne,
    #[error("Copying from textures with format {format:?} and aspect {aspect:?} is forbidden")]
    CopyFromForbiddenTextureFormat {
        format: wgt::TextureFormat,
        aspect: wgt::TextureAspect,
    },
    #[error("Copying to textures with format {format:?} and aspect {aspect:?} is forbidden")]
    CopyToForbiddenTextureFormat {
        format: wgt::TextureFormat,
        aspect: wgt::TextureAspect,
    },
    #[error(
        "Copying to textures with format {0:?} is forbidden when copying from external texture"
    )]
    ExternalCopyToForbiddenTextureFormat(wgt::TextureFormat),
    #[error(
        "Source format ({src_format:?}) and destination format ({dst_format:?}) are not copy-compatible (they may only differ in srgb-ness)"
    )]
    TextureFormatsNotCopyCompatible {
        src_format: wgt::TextureFormat,
        dst_format: wgt::TextureFormat,
    },
    #[error(transparent)]
    MemoryInitFailure(#[from] ClearError),
    #[error("Cannot encode this copy because of a missing downelevel flag")]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
    #[error("Source texture sample count must be 1, got {sample_count}")]
    InvalidSampleCount { sample_count: u32 },
    #[error(
        "Source sample count ({src_sample_count:?}) and destination sample count ({dst_sample_count:?}) are not equal"
    )]
    SampleCountNotEqual {
        src_sample_count: u32,
        dst_sample_count: u32,
    },
    #[error("Requested mip level {requested} does no exist (count: {count})")]
    InvalidMipLevel { requested: u32, count: u32 },
}

impl WebGpuError for TransferError {
    fn webgpu_error_type(&self) -> ErrorType {
        let e: &dyn WebGpuError = match self {
            Self::MissingBufferUsage(e) => e,
            Self::MissingTextureUsage(e) => e,
            Self::MemoryInitFailure(e) => e,

            Self::BufferOverrun { .. }
            | Self::TextureOverrun { .. }
            | Self::UnsupportedPartialTransfer { .. }
            | Self::InvalidCopyWithinSameTexture { .. }
            | Self::InvalidTextureAspect { .. }
            | Self::InvalidTextureMipLevel { .. }
            | Self::InvalidDimensionExternal
            | Self::UnalignedBufferOffset(..)
            | Self::UnalignedCopySize(..)
            | Self::UnalignedCopyWidth
            | Self::UnalignedCopyHeight
            | Self::UnalignedCopyOriginX
            | Self::UnalignedCopyOriginY
            | Self::UnalignedBytesPerRow
            | Self::UnspecifiedBytesPerRow
            | Self::UnspecifiedRowsPerImage
            | Self::InvalidBytesPerRow
            | Self::InvalidRowsPerImage
            | Self::SizeOverflow
            | Self::CopySrcMissingAspects
            | Self::CopyDstMissingAspects
            | Self::CopyAspectNotOne
            | Self::CopyFromForbiddenTextureFormat { .. }
            | Self::CopyToForbiddenTextureFormat { .. }
            | Self::ExternalCopyToForbiddenTextureFormat(..)
            | Self::TextureFormatsNotCopyCompatible { .. }
            | Self::MissingDownlevelFlags(..)
            | Self::InvalidSampleCount { .. }
            | Self::SampleCountNotEqual { .. }
            | Self::InvalidMipLevel { .. }
            | Self::SameSourceDestinationBuffer => return ErrorType::Validation,
        };
        e.webgpu_error_type()
    }
}

impl From<BufferTextureCopyInfoError> for TransferError {
    fn from(value: BufferTextureCopyInfoError) -> Self {
        match value {
            BufferTextureCopyInfoError::InvalidBytesPerRow => Self::InvalidBytesPerRow,
            BufferTextureCopyInfoError::InvalidRowsPerImage => Self::InvalidRowsPerImage,
            BufferTextureCopyInfoError::ImageStrideOverflow
            | BufferTextureCopyInfoError::ImageBytesOverflow(_)
            | BufferTextureCopyInfoError::ArraySizeOverflow(_) => Self::SizeOverflow,
        }
    }
}

pub(crate) fn extract_texture_selector<T>(
    copy_texture: &wgt::TexelCopyTextureInfo<T>,
    copy_size: &Extent3d,
    texture: &Texture,
) -> Result<(TextureSelector, hal::TextureCopyBase), TransferError> {
    let format = texture.desc.format;
    let copy_aspect = hal::FormatAspects::new(format, copy_texture.aspect);
    if copy_aspect.is_empty() {
        return Err(TransferError::InvalidTextureAspect {
            format,
            aspect: copy_texture.aspect,
        });
    }

    let (layers, origin_z) = match texture.desc.dimension {
        wgt::TextureDimension::D1 => (0..1, 0),
        wgt::TextureDimension::D2 => (
            copy_texture.origin.z..copy_texture.origin.z + copy_size.depth_or_array_layers,
            0,
        ),
        wgt::TextureDimension::D3 => (0..1, copy_texture.origin.z),
    };
    let base = hal::TextureCopyBase {
        origin: wgt::Origin3d {
            x: copy_texture.origin.x,
            y: copy_texture.origin.y,
            z: origin_z,
        },
        // this value will be incremented per copied layer
        array_layer: layers.start,
        mip_level: copy_texture.mip_level,
        aspect: copy_aspect,
    };
    let selector = TextureSelector {
        mips: copy_texture.mip_level..copy_texture.mip_level + 1,
        layers,
    };

    Ok((selector, base))
}

/// WebGPU's [validating linear texture data][vltd] algorithm.
///
/// Copied with some modifications from WebGPU standard.
///
/// If successful, returns a pair `(bytes, stride)`, where:
/// - `bytes` is the number of buffer bytes required for this copy, and
/// - `stride` number of bytes between array layers.
///
/// [vltd]: https://gpuweb.github.io/gpuweb/#abstract-opdef-validating-linear-texture-data
pub(crate) fn validate_linear_texture_data(
    layout: &wgt::TexelCopyBufferLayout,
    format: wgt::TextureFormat,
    aspect: wgt::TextureAspect,
    buffer_size: BufferAddress,
    buffer_side: CopySide,
    copy_size: &Extent3d,
    need_copy_aligned_rows: bool,
) -> Result<(BufferAddress, BufferAddress), TransferError> {
    let wgt::BufferTextureCopyInfo {
        copy_width,
        copy_height,
        depth_or_array_layers,

        offset,

        block_size_bytes,
        block_width_texels,
        block_height_texels,

        width_blocks: _,
        height_blocks,

        row_bytes_dense: _,
        row_stride_bytes,

        image_stride_rows: _,
        image_stride_bytes,

        image_rows_dense: _,
        image_bytes_dense: _,

        bytes_in_copy,
    } = layout.get_buffer_texture_copy_info(format, aspect, copy_size)?;

    if copy_width % block_width_texels != 0 {
        return Err(TransferError::UnalignedCopyWidth);
    }
    if copy_height % block_height_texels != 0 {
        return Err(TransferError::UnalignedCopyHeight);
    }

    let requires_multiple_rows = depth_or_array_layers > 1 || height_blocks > 1;
    let requires_multiple_images = depth_or_array_layers > 1;

    // `get_buffer_texture_copy_info()` already proceeded with defaults if these
    // were not specified, and ensured that the values satisfy the minima if
    // they were, but now we enforce the WebGPU requirement that they be
    // specified any time they apply.
    if layout.bytes_per_row.is_none() && requires_multiple_rows {
        return Err(TransferError::UnspecifiedBytesPerRow);
    }

    if layout.rows_per_image.is_none() && requires_multiple_images {
        return Err(TransferError::UnspecifiedRowsPerImage);
    };

    if need_copy_aligned_rows {
        let bytes_per_row_alignment = wgt::COPY_BYTES_PER_ROW_ALIGNMENT as BufferAddress;

        let mut offset_alignment = block_size_bytes;
        if format.is_depth_stencil_format() {
            offset_alignment = 4
        }
        if offset % offset_alignment != 0 {
            return Err(TransferError::UnalignedBufferOffset(offset));
        }

        // The alignment of row_stride_bytes is only required if there are
        // multiple rows
        if requires_multiple_rows && row_stride_bytes % bytes_per_row_alignment != 0 {
            return Err(TransferError::UnalignedBytesPerRow);
        }
    }

    // Avoid underflow in the subtraction by checking bytes_in_copy against buffer_size first.
    if bytes_in_copy > buffer_size || offset > buffer_size - bytes_in_copy {
        return Err(TransferError::BufferOverrun {
            start_offset: offset,
            end_offset: offset.wrapping_add(bytes_in_copy),
            buffer_size,
            side: buffer_side,
        });
    }

    Ok((bytes_in_copy, image_stride_bytes))
}

/// Validation for texture/buffer copies.
///
/// This implements the following checks from WebGPU's [validating texture buffer copy][vtbc]
/// algorithm:
///  * The texture must not be multisampled.
///  * The copy must be from/to a single aspect of the texture.
///  * If `aligned` is true, the buffer offset must be aligned appropriately.
///
/// The following steps in the algorithm are implemented elsewhere:
///  * Invocation of other validation algorithms.
///  * The texture usage (COPY_DST / COPY_SRC) check.
///  * The check for non-copyable depth/stencil formats. The caller must perform
///    this check using `is_valid_copy_src_format` / `is_valid_copy_dst_format`
///    before calling this function. This function will panic if
///    [`wgt::TextureFormat::block_copy_size`] returns `None` due to a
///    non-copyable format.
///
/// [vtbc]: https://gpuweb.github.io/gpuweb/#abstract-opdef-validating-texture-buffer-copy
pub(crate) fn validate_texture_buffer_copy<T>(
    texture_copy_view: &wgt::TexelCopyTextureInfo<T>,
    aspect: hal::FormatAspects,
    desc: &wgt::TextureDescriptor<(), Vec<wgt::TextureFormat>>,
    offset: BufferAddress,
    aligned: bool,
) -> Result<(), TransferError> {
    if desc.sample_count != 1 {
        return Err(TransferError::InvalidSampleCount {
            sample_count: desc.sample_count,
        });
    }

    if !aspect.is_one() {
        return Err(TransferError::CopyAspectNotOne);
    }

    let offset_alignment = if desc.format.is_depth_stencil_format() {
        4
    } else {
        // The case where `block_copy_size` returns `None` is currently
        // unreachable both for the reason in the expect message, and also
        // because the currently-defined non-copyable formats are depth/stencil
        // formats so would take the `if` branch.
        desc.format
            .block_copy_size(Some(texture_copy_view.aspect))
            .expect("non-copyable formats should have been rejected previously")
    };

    if aligned && offset % u64::from(offset_alignment) != 0 {
        return Err(TransferError::UnalignedBufferOffset(offset));
    }

    Ok(())
}

/// Validate the extent and alignment of a texture copy.
///
/// Copied with minor modifications from WebGPU standard. This mostly follows
/// the [validating GPUTexelCopyTextureInfo][vtcti] and [validating texture copy
/// range][vtcr] algorithms.
///
/// Returns the HAL copy extent and the layer count.
///
/// [vtcti]: https://gpuweb.github.io/gpuweb/#abstract-opdef-validating-gputexelcopytextureinfo
/// [vtcr]: https://gpuweb.github.io/gpuweb/#abstract-opdef-validating-texture-copy-range
pub(crate) fn validate_texture_copy_range<T>(
    texture_copy_view: &wgt::TexelCopyTextureInfo<T>,
    desc: &wgt::TextureDescriptor<(), Vec<wgt::TextureFormat>>,
    texture_side: CopySide,
    copy_size: &Extent3d,
) -> Result<(hal::CopyExtent, u32), TransferError> {
    let (block_width, block_height) = desc.format.block_dimensions();

    let extent_virtual = desc.mip_level_size(texture_copy_view.mip_level).ok_or(
        TransferError::InvalidTextureMipLevel {
            level: texture_copy_view.mip_level,
            total: desc.mip_level_count,
        },
    )?;
    // physical size can be larger than the virtual
    let extent = extent_virtual.physical_size(desc.format);

    // Multisampled and depth-stencil formats do not support partial copies
    // on x and y dimensions, but do support copying a subset of layers.
    let requires_exact_size = desc.format.is_depth_stencil_format() || desc.sample_count > 1;

    // Return `Ok` if a run `size` texels long starting at `start_offset` is
    // valid for `texture_size`. Otherwise, return an appropriate a`Err`.
    let check_dimension = |dimension: TextureErrorDimension,
                           start_offset: u32,
                           size: u32,
                           texture_size: u32,
                           requires_exact_size: bool|
     -> Result<(), TransferError> {
        if requires_exact_size && (start_offset != 0 || size != texture_size) {
            Err(TransferError::UnsupportedPartialTransfer {
                format: desc.format,
                sample_count: desc.sample_count,
                start_offset,
                end_offset: start_offset.wrapping_add(size),
                texture_size,
                dimension,
                side: texture_side,
            })
        // Avoid underflow in the subtraction by checking start_offset against
        // texture_size first.
        } else if start_offset > texture_size || texture_size - start_offset < size {
            Err(TransferError::TextureOverrun {
                start_offset,
                end_offset: start_offset.wrapping_add(size),
                texture_size,
                dimension,
                side: texture_side,
            })
        } else {
            Ok(())
        }
    };

    check_dimension(
        TextureErrorDimension::X,
        texture_copy_view.origin.x,
        copy_size.width,
        extent.width,
        requires_exact_size,
    )?;
    check_dimension(
        TextureErrorDimension::Y,
        texture_copy_view.origin.y,
        copy_size.height,
        extent.height,
        requires_exact_size,
    )?;
    check_dimension(
        TextureErrorDimension::Z,
        texture_copy_view.origin.z,
        copy_size.depth_or_array_layers,
        extent.depth_or_array_layers,
        false, // partial copy always allowed on Z/layer dimension
    )?;

    if texture_copy_view.origin.x % block_width != 0 {
        return Err(TransferError::UnalignedCopyOriginX);
    }
    if texture_copy_view.origin.y % block_height != 0 {
        return Err(TransferError::UnalignedCopyOriginY);
    }
    if copy_size.width % block_width != 0 {
        return Err(TransferError::UnalignedCopyWidth);
    }
    if copy_size.height % block_height != 0 {
        return Err(TransferError::UnalignedCopyHeight);
    }

    let (depth, array_layer_count) = match desc.dimension {
        wgt::TextureDimension::D1 => (1, 1),
        wgt::TextureDimension::D2 => (1, copy_size.depth_or_array_layers),
        wgt::TextureDimension::D3 => (copy_size.depth_or_array_layers, 1),
    };

    let copy_extent = hal::CopyExtent {
        width: copy_size.width,
        height: copy_size.height,
        depth,
    };
    Ok((copy_extent, array_layer_count))
}

/// Validate a copy within the same texture.
///
/// This implements the WebGPU requirement that the [sets of subresources for
/// texture copy][srtc] of the source and destination be disjoint, i.e. that the
/// source and destination do not overlap.
///
/// This function assumes that the copy ranges have already been validated with
/// `validate_texture_copy_range`.
///
/// [srtc]: https://gpuweb.github.io/gpuweb/#abstract-opdef-set-of-subresources-for-texture-copy
pub(crate) fn validate_copy_within_same_texture<T>(
    src: &wgt::TexelCopyTextureInfo<T>,
    dst: &wgt::TexelCopyTextureInfo<T>,
    format: wgt::TextureFormat,
    array_layer_count: u32,
) -> Result<(), TransferError> {
    let src_aspects = hal::FormatAspects::new(format, src.aspect);
    let dst_aspects = hal::FormatAspects::new(format, dst.aspect);
    if (src_aspects & dst_aspects).is_empty() {
        // Copying between different aspects (if it even makes sense), is okay.
        return Ok(());
    }

    if src.origin.z >= dst.origin.z + array_layer_count
        || dst.origin.z >= src.origin.z + array_layer_count
    {
        // Copying between non-overlapping layer ranges is okay.
        return Ok(());
    }

    if src.mip_level != dst.mip_level {
        // Copying between different mip levels is okay.
        return Ok(());
    }

    Err(TransferError::InvalidCopyWithinSameTexture {
        src_aspects: src.aspect,
        dst_aspects: dst.aspect,
        src_origin_z: src.origin.z,
        dst_origin_z: dst.origin.z,
        array_layer_count,
    })
}

fn handle_texture_init(
    init_kind: MemoryInitKind,
    cmd_buf_data: &mut CommandBufferMutable,
    device: &Device,
    copy_texture: &TexelCopyTextureInfo,
    copy_size: &Extent3d,
    texture: &Arc<Texture>,
    snatch_guard: &SnatchGuard<'_>,
) -> Result<(), ClearError> {
    let init_action = TextureInitTrackerAction {
        texture: texture.clone(),
        range: TextureInitRange {
            mip_range: copy_texture.mip_level..copy_texture.mip_level + 1,
            layer_range: copy_texture.origin.z
                ..(copy_texture.origin.z + copy_size.depth_or_array_layers),
        },
        kind: init_kind,
    };

    // Register the init action.
    let immediate_inits = cmd_buf_data
        .texture_memory_actions
        .register_init_action(&{ init_action });

    // In rare cases we may need to insert an init operation immediately onto the command buffer.
    if !immediate_inits.is_empty() {
        let cmd_buf_raw = cmd_buf_data.encoder.open()?;
        for init in immediate_inits {
            clear_texture(
                &init.texture,
                TextureInitRange {
                    mip_range: init.mip_level..(init.mip_level + 1),
                    layer_range: init.layer..(init.layer + 1),
                },
                cmd_buf_raw,
                &mut cmd_buf_data.trackers.textures,
                &device.alignments,
                device.zero_buffer.as_ref(),
                snatch_guard,
                device.instance_flags,
            )?;
        }
    }

    Ok(())
}

/// Prepare a transfer's source texture.
///
/// Ensure the source texture of a transfer is in the right initialization
/// state, and record the state for after the transfer operation.
fn handle_src_texture_init(
    cmd_buf_data: &mut CommandBufferMutable,
    device: &Device,
    source: &TexelCopyTextureInfo,
    copy_size: &Extent3d,
    texture: &Arc<Texture>,
    snatch_guard: &SnatchGuard<'_>,
) -> Result<(), TransferError> {
    handle_texture_init(
        MemoryInitKind::NeedsInitializedMemory,
        cmd_buf_data,
        device,
        source,
        copy_size,
        texture,
        snatch_guard,
    )?;
    Ok(())
}

/// Prepare a transfer's destination texture.
///
/// Ensure the destination texture of a transfer is in the right initialization
/// state, and record the state for after the transfer operation.
fn handle_dst_texture_init(
    cmd_buf_data: &mut CommandBufferMutable,
    device: &Device,
    destination: &TexelCopyTextureInfo,
    copy_size: &Extent3d,
    texture: &Arc<Texture>,
    snatch_guard: &SnatchGuard<'_>,
) -> Result<(), TransferError> {
    // Attention: If we don't write full texture subresources, we need to a full
    // clear first since we don't track subrects. This means that in rare cases
    // even a *destination* texture of a transfer may need an immediate texture
    // init.
    let dst_init_kind = if has_copy_partial_init_tracker_coverage(
        copy_size,
        destination.mip_level,
        &texture.desc,
    ) {
        MemoryInitKind::NeedsInitializedMemory
    } else {
        MemoryInitKind::ImplicitlyInitialized
    };

    handle_texture_init(
        dst_init_kind,
        cmd_buf_data,
        device,
        destination,
        copy_size,
        texture,
        snatch_guard,
    )?;
    Ok(())
}

impl Global {
    pub fn command_encoder_copy_buffer_to_buffer(
        &self,
        command_encoder_id: CommandEncoderId,
        source: BufferId,
        source_offset: BufferAddress,
        destination: BufferId,
        destination_offset: BufferAddress,
        size: Option<BufferAddress>,
    ) -> Result<(), EncoderStateError> {
        profiling::scope!("CommandEncoder::copy_buffer_to_buffer");
        api_log!(
            "CommandEncoder::copy_buffer_to_buffer {source:?} -> {destination:?} {size:?}bytes"
        );

        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(command_encoder_id);
        let mut cmd_buf_data = cmd_enc.data.lock();
        cmd_buf_data.record_with(|cmd_buf_data| -> Result<(), CommandEncoderError> {
            let device = &cmd_enc.device;
            device.check_is_valid()?;

            if source == destination {
                return Err(TransferError::SameSourceDestinationBuffer.into());
            }

            #[cfg(feature = "trace")]
            if let Some(ref mut list) = cmd_buf_data.commands {
                list.push(TraceCommand::CopyBufferToBuffer {
                    src: source,
                    src_offset: source_offset,
                    dst: destination,
                    dst_offset: destination_offset,
                    size,
                });
            }

            let snatch_guard = device.snatchable_lock.read();

            let src_buffer = hub.buffers.get(source).get()?;

            src_buffer.same_device_as(cmd_enc.as_ref())?;

            let src_pending = cmd_buf_data
                .trackers
                .buffers
                .set_single(&src_buffer, wgt::BufferUses::COPY_SRC);

            let src_raw = src_buffer.try_raw(&snatch_guard)?;
            src_buffer
                .check_usage(BufferUsages::COPY_SRC)
                .map_err(TransferError::MissingBufferUsage)?;
            // expecting only a single barrier
            let src_barrier =
                src_pending.map(|pending| pending.into_hal(&src_buffer, &snatch_guard));

            let dst_buffer = hub.buffers.get(destination).get()?;

            dst_buffer.same_device_as(cmd_enc.as_ref())?;

            let dst_pending = cmd_buf_data
                .trackers
                .buffers
                .set_single(&dst_buffer, wgt::BufferUses::COPY_DST);

            let dst_raw = dst_buffer.try_raw(&snatch_guard)?;
            dst_buffer
                .check_usage(BufferUsages::COPY_DST)
                .map_err(TransferError::MissingBufferUsage)?;
            let dst_barrier =
                dst_pending.map(|pending| pending.into_hal(&dst_buffer, &snatch_guard));

            let (size, source_end_offset) = match size {
                Some(size) => (size, source_offset + size),
                None => (src_buffer.size - source_offset, src_buffer.size),
            };

            if size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
                return Err(TransferError::UnalignedCopySize(size).into());
            }
            if source_offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
                return Err(TransferError::UnalignedBufferOffset(source_offset).into());
            }
            if destination_offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
                return Err(TransferError::UnalignedBufferOffset(destination_offset).into());
            }
            if !device
                .downlevel
                .flags
                .contains(wgt::DownlevelFlags::UNRESTRICTED_INDEX_BUFFER)
                && (src_buffer.usage.contains(BufferUsages::INDEX)
                    || dst_buffer.usage.contains(BufferUsages::INDEX))
            {
                let forbidden_usages = BufferUsages::VERTEX
                    | BufferUsages::UNIFORM
                    | BufferUsages::INDIRECT
                    | BufferUsages::STORAGE;
                if src_buffer.usage.intersects(forbidden_usages)
                    || dst_buffer.usage.intersects(forbidden_usages)
                {
                    return Err(TransferError::MissingDownlevelFlags(MissingDownlevelFlags(
                        wgt::DownlevelFlags::UNRESTRICTED_INDEX_BUFFER,
                    ))
                    .into());
                }
            }

            let destination_end_offset = destination_offset + size;
            if source_end_offset > src_buffer.size {
                return Err(TransferError::BufferOverrun {
                    start_offset: source_offset,
                    end_offset: source_end_offset,
                    buffer_size: src_buffer.size,
                    side: CopySide::Source,
                }
                .into());
            }
            if destination_end_offset > dst_buffer.size {
                return Err(TransferError::BufferOverrun {
                    start_offset: destination_offset,
                    end_offset: destination_end_offset,
                    buffer_size: dst_buffer.size,
                    side: CopySide::Destination,
                }
                .into());
            }

            if size == 0 {
                log::trace!("Ignoring copy_buffer_to_buffer of size 0");
                return Ok(());
            }

            // Make sure source is initialized memory and mark dest as initialized.
            cmd_buf_data.buffer_memory_init_actions.extend(
                dst_buffer.initialization_status.read().create_action(
                    &dst_buffer,
                    destination_offset..(destination_offset + size),
                    MemoryInitKind::ImplicitlyInitialized,
                ),
            );
            cmd_buf_data.buffer_memory_init_actions.extend(
                src_buffer.initialization_status.read().create_action(
                    &src_buffer,
                    source_offset..(source_offset + size),
                    MemoryInitKind::NeedsInitializedMemory,
                ),
            );

            let region = hal::BufferCopy {
                src_offset: source_offset,
                dst_offset: destination_offset,
                size: wgt::BufferSize::new(size).unwrap(),
            };
            let cmd_buf_raw = cmd_buf_data.encoder.open()?;
            let barriers = src_barrier
                .into_iter()
                .chain(dst_barrier)
                .collect::<Vec<_>>();
            unsafe {
                cmd_buf_raw.transition_buffers(&barriers);
                cmd_buf_raw.copy_buffer_to_buffer(src_raw, dst_raw, &[region]);
            }

            Ok(())
        })
    }

    pub fn command_encoder_copy_buffer_to_texture(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &TexelCopyBufferInfo,
        destination: &TexelCopyTextureInfo,
        copy_size: &Extent3d,
    ) -> Result<(), EncoderStateError> {
        profiling::scope!("CommandEncoder::copy_buffer_to_texture");
        api_log!(
            "CommandEncoder::copy_buffer_to_texture {:?} -> {:?} {copy_size:?}",
            source.buffer,
            destination.texture
        );

        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(command_encoder_id);
        let mut cmd_buf_data = cmd_enc.data.lock();
        cmd_buf_data.record_with(|cmd_buf_data| -> Result<(), CommandEncoderError> {
            let device = &cmd_enc.device;
            device.check_is_valid()?;

            #[cfg(feature = "trace")]
            if let Some(ref mut list) = cmd_buf_data.commands {
                list.push(TraceCommand::CopyBufferToTexture {
                    src: *source,
                    dst: *destination,
                    size: *copy_size,
                });
            }

            let dst_texture = hub.textures.get(destination.texture).get()?;
            let src_buffer = hub.buffers.get(source.buffer).get()?;

            dst_texture.same_device_as(cmd_enc.as_ref())?;
            src_buffer.same_device_as(cmd_enc.as_ref())?;

            let (hal_copy_size, array_layer_count) = validate_texture_copy_range(
                destination,
                &dst_texture.desc,
                CopySide::Destination,
                copy_size,
            )?;

            let (dst_range, dst_base) =
                extract_texture_selector(destination, copy_size, &dst_texture)?;

            let snatch_guard = device.snatchable_lock.read();

            let src_raw = src_buffer.try_raw(&snatch_guard)?;
            let dst_raw = dst_texture.try_raw(&snatch_guard)?;

            if copy_size.width == 0 || copy_size.height == 0 || copy_size.depth_or_array_layers == 0
            {
                log::trace!("Ignoring copy_buffer_to_texture of size 0");
                return Ok(());
            }

            // Handle texture init *before* dealing with barrier transitions so we
            // have an easier time inserting "immediate-inits" that may be required
            // by prior discards in rare cases.
            handle_dst_texture_init(
                cmd_buf_data,
                device,
                destination,
                copy_size,
                &dst_texture,
                &snatch_guard,
            )?;

            let src_pending = cmd_buf_data
                .trackers
                .buffers
                .set_single(&src_buffer, wgt::BufferUses::COPY_SRC);

            src_buffer
                .check_usage(BufferUsages::COPY_SRC)
                .map_err(TransferError::MissingBufferUsage)?;
            let src_barrier =
                src_pending.map(|pending| pending.into_hal(&src_buffer, &snatch_guard));

            let dst_pending = cmd_buf_data.trackers.textures.set_single(
                &dst_texture,
                dst_range,
                wgt::TextureUses::COPY_DST,
            );
            dst_texture
                .check_usage(TextureUsages::COPY_DST)
                .map_err(TransferError::MissingTextureUsage)?;
            let dst_barrier = dst_pending
                .map(|pending| pending.into_hal(dst_raw))
                .collect::<Vec<_>>();

            if !conv::is_valid_copy_dst_texture_format(dst_texture.desc.format, destination.aspect)
            {
                return Err(TransferError::CopyToForbiddenTextureFormat {
                    format: dst_texture.desc.format,
                    aspect: destination.aspect,
                }
                .into());
            }

            validate_texture_buffer_copy(
                destination,
                dst_base.aspect,
                &dst_texture.desc,
                source.layout.offset,
                true, // alignment required for buffer offset
            )?;

            let (required_buffer_bytes_in_copy, bytes_per_array_layer) =
                validate_linear_texture_data(
                    &source.layout,
                    dst_texture.desc.format,
                    destination.aspect,
                    src_buffer.size,
                    CopySide::Source,
                    copy_size,
                    true,
                )?;

            if dst_texture.desc.format.is_depth_stencil_format() {
                device
                    .require_downlevel_flags(wgt::DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES)
                    .map_err(TransferError::from)?;
            }

            cmd_buf_data.buffer_memory_init_actions.extend(
                src_buffer.initialization_status.read().create_action(
                    &src_buffer,
                    source.layout.offset..(source.layout.offset + required_buffer_bytes_in_copy),
                    MemoryInitKind::NeedsInitializedMemory,
                ),
            );

            let regions = (0..array_layer_count)
                .map(|rel_array_layer| {
                    let mut texture_base = dst_base.clone();
                    texture_base.array_layer += rel_array_layer;
                    let mut buffer_layout = source.layout;
                    buffer_layout.offset += rel_array_layer as u64 * bytes_per_array_layer;
                    hal::BufferTextureCopy {
                        buffer_layout,
                        texture_base,
                        size: hal_copy_size,
                    }
                })
                .collect::<Vec<_>>();

            let cmd_buf_raw = cmd_buf_data.encoder.open()?;
            unsafe {
                cmd_buf_raw.transition_textures(&dst_barrier);
                cmd_buf_raw.transition_buffers(src_barrier.as_slice());
                cmd_buf_raw.copy_buffer_to_texture(src_raw, dst_raw, &regions);
            }

            Ok(())
        })
    }

    pub fn command_encoder_copy_texture_to_buffer(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &TexelCopyTextureInfo,
        destination: &TexelCopyBufferInfo,
        copy_size: &Extent3d,
    ) -> Result<(), EncoderStateError> {
        profiling::scope!("CommandEncoder::copy_texture_to_buffer");
        api_log!(
            "CommandEncoder::copy_texture_to_buffer {:?} -> {:?} {copy_size:?}",
            source.texture,
            destination.buffer
        );

        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(command_encoder_id);
        let mut cmd_buf_data = cmd_enc.data.lock();
        cmd_buf_data.record_with(|cmd_buf_data| -> Result<(), CommandEncoderError> {
            let device = &cmd_enc.device;
            device.check_is_valid()?;

            #[cfg(feature = "trace")]
            if let Some(list) = cmd_buf_data.commands.as_mut() {
                list.push(TraceCommand::CopyTextureToBuffer {
                    src: *source,
                    dst: *destination,
                    size: *copy_size,
                });
            }

            let src_texture = hub.textures.get(source.texture).get()?;
            let dst_buffer = hub.buffers.get(destination.buffer).get()?;

            src_texture.same_device_as(cmd_enc.as_ref())?;
            dst_buffer.same_device_as(cmd_enc.as_ref())?;

            let (hal_copy_size, array_layer_count) = validate_texture_copy_range(
                source,
                &src_texture.desc,
                CopySide::Source,
                copy_size,
            )?;

            let (src_range, src_base) = extract_texture_selector(source, copy_size, &src_texture)?;

            let snatch_guard = device.snatchable_lock.read();

            let src_raw = src_texture.try_raw(&snatch_guard)?;
            src_texture
                .check_usage(TextureUsages::COPY_SRC)
                .map_err(TransferError::MissingTextureUsage)?;

            if source.mip_level >= src_texture.desc.mip_level_count {
                return Err(TransferError::InvalidMipLevel {
                    requested: source.mip_level,
                    count: src_texture.desc.mip_level_count,
                }
                .into());
            }

            if !conv::is_valid_copy_src_texture_format(src_texture.desc.format, source.aspect) {
                return Err(TransferError::CopyFromForbiddenTextureFormat {
                    format: src_texture.desc.format,
                    aspect: source.aspect,
                }
                .into());
            }

            validate_texture_buffer_copy(
                source,
                src_base.aspect,
                &src_texture.desc,
                destination.layout.offset,
                true, // alignment required for buffer offset
            )?;

            let (required_buffer_bytes_in_copy, bytes_per_array_layer) =
                validate_linear_texture_data(
                    &destination.layout,
                    src_texture.desc.format,
                    source.aspect,
                    dst_buffer.size,
                    CopySide::Destination,
                    copy_size,
                    true,
                )?;

            if src_texture.desc.format.is_depth_stencil_format() {
                device
                    .require_downlevel_flags(wgt::DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES)
                    .map_err(TransferError::from)?;
            }

            let dst_raw = dst_buffer.try_raw(&snatch_guard)?;
            dst_buffer
                .check_usage(BufferUsages::COPY_DST)
                .map_err(TransferError::MissingBufferUsage)?;

            if copy_size.width == 0 || copy_size.height == 0 || copy_size.depth_or_array_layers == 0
            {
                log::trace!("Ignoring copy_texture_to_buffer of size 0");
                return Ok(());
            }

            // Handle texture init *before* dealing with barrier transitions so we
            // have an easier time inserting "immediate-inits" that may be required
            // by prior discards in rare cases.
            handle_src_texture_init(
                cmd_buf_data,
                device,
                source,
                copy_size,
                &src_texture,
                &snatch_guard,
            )?;

            let src_pending = cmd_buf_data.trackers.textures.set_single(
                &src_texture,
                src_range,
                wgt::TextureUses::COPY_SRC,
            );
            let src_barrier = src_pending
                .map(|pending| pending.into_hal(src_raw))
                .collect::<Vec<_>>();

            let dst_pending = cmd_buf_data
                .trackers
                .buffers
                .set_single(&dst_buffer, wgt::BufferUses::COPY_DST);

            let dst_barrier =
                dst_pending.map(|pending| pending.into_hal(&dst_buffer, &snatch_guard));

            cmd_buf_data.buffer_memory_init_actions.extend(
                dst_buffer.initialization_status.read().create_action(
                    &dst_buffer,
                    destination.layout.offset
                        ..(destination.layout.offset + required_buffer_bytes_in_copy),
                    MemoryInitKind::ImplicitlyInitialized,
                ),
            );

            let regions = (0..array_layer_count)
                .map(|rel_array_layer| {
                    let mut texture_base = src_base.clone();
                    texture_base.array_layer += rel_array_layer;
                    let mut buffer_layout = destination.layout;
                    buffer_layout.offset += rel_array_layer as u64 * bytes_per_array_layer;
                    hal::BufferTextureCopy {
                        buffer_layout,
                        texture_base,
                        size: hal_copy_size,
                    }
                })
                .collect::<Vec<_>>();
            let cmd_buf_raw = cmd_buf_data.encoder.open()?;
            unsafe {
                cmd_buf_raw.transition_buffers(dst_barrier.as_slice());
                cmd_buf_raw.transition_textures(&src_barrier);
                cmd_buf_raw.copy_texture_to_buffer(
                    src_raw,
                    wgt::TextureUses::COPY_SRC,
                    dst_raw,
                    &regions,
                );
            }

            Ok(())
        })
    }

    pub fn command_encoder_copy_texture_to_texture(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &TexelCopyTextureInfo,
        destination: &TexelCopyTextureInfo,
        copy_size: &Extent3d,
    ) -> Result<(), EncoderStateError> {
        profiling::scope!("CommandEncoder::copy_texture_to_texture");
        api_log!(
            "CommandEncoder::copy_texture_to_texture {:?} -> {:?} {copy_size:?}",
            source.texture,
            destination.texture
        );

        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(command_encoder_id);
        let mut cmd_buf_data = cmd_enc.data.lock();
        cmd_buf_data.record_with(|cmd_buf_data| -> Result<(), CommandEncoderError> {
            let device = &cmd_enc.device;
            device.check_is_valid()?;

            let snatch_guard = device.snatchable_lock.read();

            #[cfg(feature = "trace")]
            if let Some(ref mut list) = cmd_buf_data.commands {
                list.push(TraceCommand::CopyTextureToTexture {
                    src: *source,
                    dst: *destination,
                    size: *copy_size,
                });
            }

            let src_texture = hub.textures.get(source.texture).get()?;
            let dst_texture = hub.textures.get(destination.texture).get()?;

            src_texture.same_device_as(cmd_enc.as_ref())?;
            dst_texture.same_device_as(cmd_enc.as_ref())?;

            // src and dst texture format must be copy-compatible
            // https://gpuweb.github.io/gpuweb/#copy-compatible
            if src_texture.desc.format.remove_srgb_suffix()
                != dst_texture.desc.format.remove_srgb_suffix()
            {
                return Err(TransferError::TextureFormatsNotCopyCompatible {
                    src_format: src_texture.desc.format,
                    dst_format: dst_texture.desc.format,
                }
                .into());
            }

            let (src_copy_size, array_layer_count) = validate_texture_copy_range(
                source,
                &src_texture.desc,
                CopySide::Source,
                copy_size,
            )?;
            let (dst_copy_size, _) = validate_texture_copy_range(
                destination,
                &dst_texture.desc,
                CopySide::Destination,
                copy_size,
            )?;

            if Arc::as_ptr(&src_texture) == Arc::as_ptr(&dst_texture) {
                validate_copy_within_same_texture(
                    source,
                    destination,
                    src_texture.desc.format,
                    array_layer_count,
                )?;
            }

            let (src_range, src_tex_base) =
                extract_texture_selector(source, copy_size, &src_texture)?;
            let (dst_range, dst_tex_base) =
                extract_texture_selector(destination, copy_size, &dst_texture)?;
            let src_texture_aspects = hal::FormatAspects::from(src_texture.desc.format);
            let dst_texture_aspects = hal::FormatAspects::from(dst_texture.desc.format);
            if src_tex_base.aspect != src_texture_aspects {
                return Err(TransferError::CopySrcMissingAspects.into());
            }
            if dst_tex_base.aspect != dst_texture_aspects {
                return Err(TransferError::CopyDstMissingAspects.into());
            }

            if src_texture.desc.sample_count != dst_texture.desc.sample_count {
                return Err(TransferError::SampleCountNotEqual {
                    src_sample_count: src_texture.desc.sample_count,
                    dst_sample_count: dst_texture.desc.sample_count,
                }
                .into());
            }

            // Handle texture init *before* dealing with barrier transitions so we
            // have an easier time inserting "immediate-inits" that may be required
            // by prior discards in rare cases.
            handle_src_texture_init(
                cmd_buf_data,
                device,
                source,
                copy_size,
                &src_texture,
                &snatch_guard,
            )?;
            handle_dst_texture_init(
                cmd_buf_data,
                device,
                destination,
                copy_size,
                &dst_texture,
                &snatch_guard,
            )?;

            let src_raw = src_texture.try_raw(&snatch_guard)?;
            src_texture
                .check_usage(TextureUsages::COPY_SRC)
                .map_err(TransferError::MissingTextureUsage)?;
            let dst_raw = dst_texture.try_raw(&snatch_guard)?;
            dst_texture
                .check_usage(TextureUsages::COPY_DST)
                .map_err(TransferError::MissingTextureUsage)?;

            if copy_size.width == 0 || copy_size.height == 0 || copy_size.depth_or_array_layers == 0
            {
                log::trace!("Ignoring copy_texture_to_texture of size 0");
                return Ok(());
            }

            let src_pending = cmd_buf_data.trackers.textures.set_single(
                &src_texture,
                src_range,
                wgt::TextureUses::COPY_SRC,
            );

            //TODO: try to avoid this the collection. It's needed because both
            // `src_pending` and `dst_pending` try to hold `trackers.textures` mutably.
            let mut barriers: ArrayVec<_, 2> = src_pending
                .map(|pending| pending.into_hal(src_raw))
                .collect();

            let dst_pending = cmd_buf_data.trackers.textures.set_single(
                &dst_texture,
                dst_range,
                wgt::TextureUses::COPY_DST,
            );
            barriers.extend(dst_pending.map(|pending| pending.into_hal(dst_raw)));

            let hal_copy_size = hal::CopyExtent {
                width: src_copy_size.width.min(dst_copy_size.width),
                height: src_copy_size.height.min(dst_copy_size.height),
                depth: src_copy_size.depth.min(dst_copy_size.depth),
            };

            let regions = (0..array_layer_count).map(|rel_array_layer| {
                let mut src_base = src_tex_base.clone();
                let mut dst_base = dst_tex_base.clone();
                src_base.array_layer += rel_array_layer;
                dst_base.array_layer += rel_array_layer;
                hal::TextureCopy {
                    src_base,
                    dst_base,
                    size: hal_copy_size,
                }
            });

            let regions = if dst_tex_base.aspect == hal::FormatAspects::DEPTH_STENCIL {
                regions
                    .flat_map(|region| {
                        let (mut depth, mut stencil) = (region.clone(), region);
                        depth.src_base.aspect = hal::FormatAspects::DEPTH;
                        depth.dst_base.aspect = hal::FormatAspects::DEPTH;
                        stencil.src_base.aspect = hal::FormatAspects::STENCIL;
                        stencil.dst_base.aspect = hal::FormatAspects::STENCIL;
                        [depth, stencil]
                    })
                    .collect::<Vec<_>>()
            } else {
                regions.collect::<Vec<_>>()
            };
            let cmd_buf_raw = cmd_buf_data.encoder.open()?;
            unsafe {
                cmd_buf_raw.transition_textures(&barriers);
                cmd_buf_raw.copy_texture_to_texture(
                    src_raw,
                    wgt::TextureUses::COPY_SRC,
                    dst_raw,
                    &regions,
                );
            }

            Ok(())
        })
    }
}
