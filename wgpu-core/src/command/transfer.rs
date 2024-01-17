#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;
use crate::{
    api_log,
    command::{clear_texture, CommandBuffer, CommandEncoderError},
    conv,
    device::{Device, DeviceError, MissingDownlevelFlags},
    error::{ErrorFormatter, PrettyError},
    global::Global,
    hal_api::HalApi,
    id::{BufferId, CommandEncoderId, DeviceId, TextureId},
    identity::GlobalIdentityHandlerFactory,
    init_tracker::{
        has_copy_partial_init_tracker_coverage, MemoryInitKind, TextureInitRange,
        TextureInitTrackerAction,
    },
    resource::{Resource, Texture, TextureErrorDimension},
    track::{TextureSelector, Tracker},
};

use arrayvec::ArrayVec;
use hal::CommandEncoder as _;
use thiserror::Error;
use wgt::{BufferAddress, BufferUsages, Extent3d, TextureUsages};

use std::{iter, sync::Arc};

use super::{memory_init::CommandBufferTextureMemoryActions, ClearError, CommandEncoder};

pub type ImageCopyBuffer = wgt::ImageCopyBuffer<BufferId>;
pub type ImageCopyTexture = wgt::ImageCopyTexture<TextureId>;
pub type ImageCopyTextureTagged = wgt::ImageCopyTextureTagged<TextureId>;

#[derive(Clone, Copy, Debug)]
pub enum CopySide {
    Source,
    Destination,
}

/// Error encountered while attempting a data transfer.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TransferError {
    #[error("Device {0:?} is invalid")]
    InvalidDevice(DeviceId),
    #[error("Buffer {0:?} is invalid or destroyed")]
    InvalidBuffer(BufferId),
    #[error("Texture {0:?} is invalid or destroyed")]
    InvalidTexture(TextureId),
    #[error("Source and destination cannot be the same buffer")]
    SameSourceDestinationBuffer,
    #[error("Source buffer/texture is missing the `COPY_SRC` usage flag")]
    MissingCopySrcUsageFlag,
    #[error("Destination buffer/texture is missing the `COPY_DST` usage flag")]
    MissingCopyDstUsageFlag(Option<BufferId>, Option<TextureId>),
    #[error("Destination texture is missing the `RENDER_ATTACHMENT` usage flag")]
    MissingRenderAttachmentUsageFlag(TextureId),
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
    #[error("Unable to select texture aspect {aspect:?} from fromat {format:?}")]
    InvalidTextureAspect {
        format: wgt::TextureFormat,
        aspect: wgt::TextureAspect,
    },
    #[error("Unable to select texture mip level {level} out of {total}")]
    InvalidTextureMipLevel { level: u32, total: u32 },
    #[error("Texture dimension must be 2D when copying from an external texture")]
    InvalidDimensionExternal(TextureId),
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
    #[error("Image is 1D and the copy height and depth are not both set to 1")]
    InvalidCopySize,
    #[error("Number of rows per image is invalid")]
    InvalidRowsPerImage,
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
    #[error("The entire texture must be copied when copying from depth texture")]
    InvalidDepthTextureExtent,
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
    #[error("Requested mip level {requested} does no exist (count: {count})")]
    InvalidMipLevel { requested: u32, count: u32 },
}

impl PrettyError for TransferError {
    fn fmt_pretty(&self, fmt: &mut ErrorFormatter) {
        fmt.error(self);
        match *self {
            Self::InvalidBuffer(id) => {
                fmt.buffer_label(&id);
            }
            Self::InvalidTexture(id) => {
                fmt.texture_label(&id);
            }
            // Self::MissingCopySrcUsageFlag(buf_opt, tex_opt) => {
            //     if let Some(buf) = buf_opt {
            //         let name = crate::gfx_select!(buf => global.buffer_label(buf));
            //         ret.push_str(&format_label_line("source", &name));
            //     }
            //     if let Some(tex) = tex_opt {
            //         let name = crate::gfx_select!(tex => global.texture_label(tex));
            //         ret.push_str(&format_label_line("source", &name));
            //     }
            // }
            Self::MissingCopyDstUsageFlag(buf_opt, tex_opt) => {
                if let Some(buf) = buf_opt {
                    fmt.buffer_label_with_key(&buf, "destination");
                }
                if let Some(tex) = tex_opt {
                    fmt.texture_label_with_key(&tex, "destination");
                }
            }
            _ => {}
        };
    }
}
/// Error encountered while attempting to do a copy on a command encoder.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CopyError {
    #[error(transparent)]
    Encoder(#[from] CommandEncoderError),
    #[error("Copy error")]
    Transfer(#[from] TransferError),
}

impl From<DeviceError> for CopyError {
    fn from(err: DeviceError) -> Self {
        CopyError::Encoder(CommandEncoderError::Device(err))
    }
}

pub(crate) fn extract_texture_selector<A: HalApi>(
    copy_texture: &ImageCopyTexture,
    copy_size: &Extent3d,
    texture: &Texture<A>,
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
    layout: &wgt::ImageDataLayout,
    format: wgt::TextureFormat,
    aspect: wgt::TextureAspect,
    buffer_size: BufferAddress,
    buffer_side: CopySide,
    copy_size: &Extent3d,
    need_copy_aligned_rows: bool,
) -> Result<(BufferAddress, BufferAddress), TransferError> {
    // Convert all inputs to BufferAddress (u64) to avoid some of the overflow issues
    // Note: u64 is not always enough to prevent overflow, especially when multiplying
    // something with a potentially large depth value, so it is preferrable to validate
    // the copy size before calling this function (for example via `validate_texture_copy_range`).
    let copy_width = copy_size.width as BufferAddress;
    let copy_height = copy_size.height as BufferAddress;
    let copy_depth = copy_size.depth_or_array_layers as BufferAddress;

    let offset = layout.offset;

    let block_size = format.block_copy_size(Some(aspect)).unwrap() as BufferAddress;
    let (block_width, block_height) = format.block_dimensions();
    let block_width = block_width as BufferAddress;
    let block_height = block_height as BufferAddress;

    if copy_width % block_width != 0 {
        return Err(TransferError::UnalignedCopyWidth);
    }
    if copy_height % block_height != 0 {
        return Err(TransferError::UnalignedCopyHeight);
    }

    let width_in_blocks = copy_width / block_width;
    let height_in_blocks = copy_height / block_height;

    let bytes_in_last_row = width_in_blocks * block_size;

    let bytes_per_row = if let Some(bytes_per_row) = layout.bytes_per_row {
        let bytes_per_row = bytes_per_row as BufferAddress;
        if bytes_per_row < bytes_in_last_row {
            return Err(TransferError::InvalidBytesPerRow);
        }
        bytes_per_row
    } else {
        if copy_depth > 1 || height_in_blocks > 1 {
            return Err(TransferError::UnspecifiedBytesPerRow);
        }
        0
    };
    let block_rows_per_image = if let Some(rows_per_image) = layout.rows_per_image {
        let rows_per_image = rows_per_image as BufferAddress;
        if rows_per_image < height_in_blocks {
            return Err(TransferError::InvalidRowsPerImage);
        }
        rows_per_image
    } else {
        if copy_depth > 1 {
            return Err(TransferError::UnspecifiedRowsPerImage);
        }
        0
    };

    if need_copy_aligned_rows {
        let bytes_per_row_alignment = wgt::COPY_BYTES_PER_ROW_ALIGNMENT as BufferAddress;

        let mut offset_alignment = block_size;
        if format.is_depth_stencil_format() {
            offset_alignment = 4
        }
        if offset % offset_alignment != 0 {
            return Err(TransferError::UnalignedBufferOffset(offset));
        }

        if bytes_per_row % bytes_per_row_alignment != 0 {
            return Err(TransferError::UnalignedBytesPerRow);
        }
    }

    let bytes_per_image = bytes_per_row * block_rows_per_image;

    let required_bytes_in_copy = if copy_depth == 0 {
        0
    } else {
        let mut required_bytes_in_copy = bytes_per_image * (copy_depth - 1);
        if height_in_blocks > 0 {
            required_bytes_in_copy += bytes_per_row * (height_in_blocks - 1) + bytes_in_last_row;
        }
        required_bytes_in_copy
    };

    if offset + required_bytes_in_copy > buffer_size {
        return Err(TransferError::BufferOverrun {
            start_offset: offset,
            end_offset: offset + required_bytes_in_copy,
            buffer_size,
            side: buffer_side,
        });
    }

    Ok((required_bytes_in_copy, bytes_per_image))
}

/// WebGPU's [validating texture copy range][vtcr] algorithm.
///
/// Copied with minor modifications from WebGPU standard.
///
/// Returns the HAL copy extent and the layer count.
///
/// [vtcr]: https://gpuweb.github.io/gpuweb/#validating-texture-copy-range
pub(crate) fn validate_texture_copy_range(
    texture_copy_view: &ImageCopyTexture,
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

    if desc.format.is_depth_stencil_format() && *copy_size != extent {
        return Err(TransferError::InvalidDepthTextureExtent);
    }

    /// Return `Ok` if a run `size` texels long starting at `start_offset` falls
    /// entirely within `texture_size`. Otherwise, return an appropriate a`Err`.
    fn check_dimension(
        dimension: TextureErrorDimension,
        side: CopySide,
        start_offset: u32,
        size: u32,
        texture_size: u32,
    ) -> Result<(), TransferError> {
        // Avoid underflow in the subtraction by checking start_offset against
        // texture_size first.
        if start_offset <= texture_size && size <= texture_size - start_offset {
            Ok(())
        } else {
            Err(TransferError::TextureOverrun {
                start_offset,
                end_offset: start_offset.wrapping_add(size),
                texture_size,
                dimension,
                side,
            })
        }
    }

    check_dimension(
        TextureErrorDimension::X,
        texture_side,
        texture_copy_view.origin.x,
        copy_size.width,
        extent.width,
    )?;
    check_dimension(
        TextureErrorDimension::Y,
        texture_side,
        texture_copy_view.origin.y,
        copy_size.height,
        extent.height,
    )?;
    check_dimension(
        TextureErrorDimension::Z,
        texture_side,
        texture_copy_view.origin.z,
        copy_size.depth_or_array_layers,
        extent.depth_or_array_layers,
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

fn handle_texture_init<A: HalApi>(
    init_kind: MemoryInitKind,
    encoder: &mut CommandEncoder<A>,
    trackers: &mut Tracker<A>,
    texture_memory_actions: &mut CommandBufferTextureMemoryActions<A>,
    device: &Device<A>,
    copy_texture: &ImageCopyTexture,
    copy_size: &Extent3d,
    texture: &Arc<Texture<A>>,
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
    let immediate_inits = texture_memory_actions.register_init_action(&{ init_action });

    // In rare cases we may need to insert an init operation immediately onto the command buffer.
    if !immediate_inits.is_empty() {
        let cmd_buf_raw = encoder.open()?;
        for init in immediate_inits {
            clear_texture(
                &init.texture,
                TextureInitRange {
                    mip_range: init.mip_level..(init.mip_level + 1),
                    layer_range: init.layer..(init.layer + 1),
                },
                cmd_buf_raw,
                &mut trackers.textures,
                &device.alignments,
                device.zero_buffer.as_ref().unwrap(),
            )?;
        }
    }

    Ok(())
}

/// Prepare a transfer's source texture.
///
/// Ensure the source texture of a transfer is in the right initialization
/// state, and record the state for after the transfer operation.
fn handle_src_texture_init<A: HalApi>(
    encoder: &mut CommandEncoder<A>,
    trackers: &mut Tracker<A>,
    texture_memory_actions: &mut CommandBufferTextureMemoryActions<A>,
    device: &Device<A>,
    source: &ImageCopyTexture,
    copy_size: &Extent3d,
    texture: &Arc<Texture<A>>,
) -> Result<(), TransferError> {
    handle_texture_init(
        MemoryInitKind::NeedsInitializedMemory,
        encoder,
        trackers,
        texture_memory_actions,
        device,
        source,
        copy_size,
        texture,
    )?;
    Ok(())
}

/// Prepare a transfer's destination texture.
///
/// Ensure the destination texture of a transfer is in the right initialization
/// state, and record the state for after the transfer operation.
fn handle_dst_texture_init<A: HalApi>(
    encoder: &mut CommandEncoder<A>,
    trackers: &mut Tracker<A>,
    texture_memory_actions: &mut CommandBufferTextureMemoryActions<A>,
    device: &Device<A>,
    destination: &ImageCopyTexture,
    copy_size: &Extent3d,
    texture: &Arc<Texture<A>>,
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
        encoder,
        trackers,
        texture_memory_actions,
        device,
        destination,
        copy_size,
        texture,
    )?;
    Ok(())
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_copy_buffer_to_buffer<A: HalApi>(
        &self,
        command_encoder_id: CommandEncoderId,
        source: BufferId,
        source_offset: BufferAddress,
        destination: BufferId,
        destination_offset: BufferAddress,
        size: BufferAddress,
    ) -> Result<(), CopyError> {
        profiling::scope!("CommandEncoder::copy_buffer_to_buffer");
        api_log!(
            "CommandEncoder::copy_buffer_to_buffer {source:?} -> {destination:?} {size:?}bytes"
        );

        if source == destination {
            return Err(TransferError::SameSourceDestinationBuffer.into());
        }
        let hub = A::hub(self);

        let cmd_buf = CommandBuffer::get_encoder(hub, command_encoder_id)?;
        let mut cmd_buf_data = cmd_buf.data.lock();
        let cmd_buf_data = cmd_buf_data.as_mut().unwrap();

        let device = &cmd_buf.device;
        if !device.is_valid() {
            return Err(TransferError::InvalidDevice(cmd_buf.device.as_info().id()).into());
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

        let (src_buffer, src_pending) = {
            let buffer_guard = hub.buffers.read();
            let src_buffer = buffer_guard
                .get(source)
                .map_err(|_| TransferError::InvalidBuffer(source))?;
            cmd_buf_data
                .trackers
                .buffers
                .set_single(src_buffer, hal::BufferUses::COPY_SRC)
                .ok_or(TransferError::InvalidBuffer(source))?
        };
        let src_raw = src_buffer
            .raw
            .get(&snatch_guard)
            .ok_or(TransferError::InvalidBuffer(source))?;
        if !src_buffer.usage.contains(BufferUsages::COPY_SRC) {
            return Err(TransferError::MissingCopySrcUsageFlag.into());
        }
        // expecting only a single barrier
        let src_barrier = src_pending.map(|pending| pending.into_hal(&src_buffer, &snatch_guard));

        let (dst_buffer, dst_pending) = {
            let buffer_guard = hub.buffers.read();
            let dst_buffer = buffer_guard
                .get(destination)
                .map_err(|_| TransferError::InvalidBuffer(destination))?;
            cmd_buf_data
                .trackers
                .buffers
                .set_single(dst_buffer, hal::BufferUses::COPY_DST)
                .ok_or(TransferError::InvalidBuffer(destination))?
        };
        let dst_raw = dst_buffer
            .raw
            .get(&snatch_guard)
            .ok_or(TransferError::InvalidBuffer(destination))?;
        if !dst_buffer.usage.contains(BufferUsages::COPY_DST) {
            return Err(TransferError::MissingCopyDstUsageFlag(Some(destination), None).into());
        }
        let dst_barrier = dst_pending.map(|pending| pending.into_hal(&dst_buffer, &snatch_guard));

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
            && (src_buffer.usage.contains(wgt::BufferUsages::INDEX)
                || dst_buffer.usage.contains(wgt::BufferUsages::INDEX))
        {
            let forbidden_usages = wgt::BufferUsages::VERTEX
                | wgt::BufferUsages::UNIFORM
                | wgt::BufferUsages::INDIRECT
                | wgt::BufferUsages::STORAGE;
            if src_buffer.usage.intersects(forbidden_usages)
                || dst_buffer.usage.intersects(forbidden_usages)
            {
                return Err(TransferError::MissingDownlevelFlags(MissingDownlevelFlags(
                    wgt::DownlevelFlags::UNRESTRICTED_INDEX_BUFFER,
                ))
                .into());
            }
        }

        let source_end_offset = source_offset + size;
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
        unsafe {
            cmd_buf_raw.transition_buffers(src_barrier.into_iter().chain(dst_barrier));
            cmd_buf_raw.copy_buffer_to_buffer(src_raw, dst_raw, iter::once(region));
        }
        Ok(())
    }

    pub fn command_encoder_copy_buffer_to_texture<A: HalApi>(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &ImageCopyBuffer,
        destination: &ImageCopyTexture,
        copy_size: &Extent3d,
    ) -> Result<(), CopyError> {
        profiling::scope!("CommandEncoder::copy_buffer_to_texture");
        api_log!(
            "CommandEncoder::copy_buffer_to_texture {:?} -> {:?} {copy_size:?}",
            source.buffer,
            destination.texture
        );

        let hub = A::hub(self);

        let cmd_buf = CommandBuffer::get_encoder(hub, command_encoder_id)?;
        let device = &cmd_buf.device;
        if !device.is_valid() {
            return Err(TransferError::InvalidDevice(cmd_buf.device.as_info().id()).into());
        }

        let mut cmd_buf_data = cmd_buf.data.lock();
        let cmd_buf_data = cmd_buf_data.as_mut().unwrap();

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf_data.commands {
            list.push(TraceCommand::CopyBufferToTexture {
                src: *source,
                dst: *destination,
                size: *copy_size,
            });
        }

        let encoder = &mut cmd_buf_data.encoder;
        let tracker = &mut cmd_buf_data.trackers;
        let buffer_memory_init_actions = &mut cmd_buf_data.buffer_memory_init_actions;
        let texture_memory_actions = &mut cmd_buf_data.texture_memory_actions;

        if copy_size.width == 0 || copy_size.height == 0 || copy_size.depth_or_array_layers == 0 {
            log::trace!("Ignoring copy_buffer_to_texture of size 0");
            return Ok(());
        }

        let dst_texture = hub
            .textures
            .get(destination.texture)
            .map_err(|_| TransferError::InvalidTexture(destination.texture))?;

        let (hal_copy_size, array_layer_count) = validate_texture_copy_range(
            destination,
            &dst_texture.desc,
            CopySide::Destination,
            copy_size,
        )?;

        let (dst_range, dst_base) = extract_texture_selector(destination, copy_size, &dst_texture)?;

        // Handle texture init *before* dealing with barrier transitions so we
        // have an easier time inserting "immediate-inits" that may be required
        // by prior discards in rare cases.
        handle_dst_texture_init(
            encoder,
            tracker,
            texture_memory_actions,
            device,
            destination,
            copy_size,
            &dst_texture,
        )?;

        let snatch_guard = device.snatchable_lock.read();

        let (src_buffer, src_pending) = {
            let buffer_guard = hub.buffers.read();
            let src_buffer = buffer_guard
                .get(source.buffer)
                .map_err(|_| TransferError::InvalidBuffer(source.buffer))?;
            tracker
                .buffers
                .set_single(src_buffer, hal::BufferUses::COPY_SRC)
                .ok_or(TransferError::InvalidBuffer(source.buffer))?
        };
        let src_raw = src_buffer
            .raw
            .get(&snatch_guard)
            .ok_or(TransferError::InvalidBuffer(source.buffer))?;
        if !src_buffer.usage.contains(BufferUsages::COPY_SRC) {
            return Err(TransferError::MissingCopySrcUsageFlag.into());
        }
        let src_barrier = src_pending.map(|pending| pending.into_hal(&src_buffer, &snatch_guard));

        let dst_pending = tracker
            .textures
            .set_single(&dst_texture, dst_range, hal::TextureUses::COPY_DST)
            .ok_or(TransferError::InvalidTexture(destination.texture))?;
        let dst_raw = dst_texture
            .raw(&snatch_guard)
            .ok_or(TransferError::InvalidTexture(destination.texture))?;
        if !dst_texture.desc.usage.contains(TextureUsages::COPY_DST) {
            return Err(
                TransferError::MissingCopyDstUsageFlag(None, Some(destination.texture)).into(),
            );
        }
        let dst_barrier = dst_pending.map(|pending| pending.into_hal(dst_raw));

        if !dst_base.aspect.is_one() {
            return Err(TransferError::CopyAspectNotOne.into());
        }

        if !conv::is_valid_copy_dst_texture_format(dst_texture.desc.format, destination.aspect) {
            return Err(TransferError::CopyToForbiddenTextureFormat {
                format: dst_texture.desc.format,
                aspect: destination.aspect,
            }
            .into());
        }

        let (required_buffer_bytes_in_copy, bytes_per_array_layer) = validate_linear_texture_data(
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

        buffer_memory_init_actions.extend(src_buffer.initialization_status.read().create_action(
            &src_buffer,
            source.layout.offset..(source.layout.offset + required_buffer_bytes_in_copy),
            MemoryInitKind::NeedsInitializedMemory,
        ));

        let regions = (0..array_layer_count).map(|rel_array_layer| {
            let mut texture_base = dst_base.clone();
            texture_base.array_layer += rel_array_layer;
            let mut buffer_layout = source.layout;
            buffer_layout.offset += rel_array_layer as u64 * bytes_per_array_layer;
            hal::BufferTextureCopy {
                buffer_layout,
                texture_base,
                size: hal_copy_size,
            }
        });

        let cmd_buf_raw = encoder.open()?;
        unsafe {
            cmd_buf_raw.transition_textures(dst_barrier.into_iter());
            cmd_buf_raw.transition_buffers(src_barrier.into_iter());
            cmd_buf_raw.copy_buffer_to_texture(src_raw, dst_raw, regions);
        }
        Ok(())
    }

    pub fn command_encoder_copy_texture_to_buffer<A: HalApi>(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &ImageCopyTexture,
        destination: &ImageCopyBuffer,
        copy_size: &Extent3d,
    ) -> Result<(), CopyError> {
        profiling::scope!("CommandEncoder::copy_texture_to_buffer");
        api_log!(
            "CommandEncoder::copy_texture_to_buffer {:?} -> {:?} {copy_size:?}",
            source.texture,
            destination.buffer
        );

        let hub = A::hub(self);

        let cmd_buf = CommandBuffer::get_encoder(hub, command_encoder_id)?;
        let device = &cmd_buf.device;
        if !device.is_valid() {
            return Err(TransferError::InvalidDevice(cmd_buf.device.as_info().id()).into());
        }

        let mut cmd_buf_data = cmd_buf.data.lock();
        let cmd_buf_data = cmd_buf_data.as_mut().unwrap();

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf_data.commands {
            list.push(TraceCommand::CopyTextureToBuffer {
                src: *source,
                dst: *destination,
                size: *copy_size,
            });
        }
        let encoder = &mut cmd_buf_data.encoder;
        let tracker = &mut cmd_buf_data.trackers;
        let buffer_memory_init_actions = &mut cmd_buf_data.buffer_memory_init_actions;
        let texture_memory_actions = &mut cmd_buf_data.texture_memory_actions;

        if copy_size.width == 0 || copy_size.height == 0 || copy_size.depth_or_array_layers == 0 {
            log::trace!("Ignoring copy_texture_to_buffer of size 0");
            return Ok(());
        }

        let src_texture = hub
            .textures
            .get(source.texture)
            .map_err(|_| TransferError::InvalidTexture(source.texture))?;

        let (hal_copy_size, array_layer_count) =
            validate_texture_copy_range(source, &src_texture.desc, CopySide::Source, copy_size)?;

        let (src_range, src_base) = extract_texture_selector(source, copy_size, &src_texture)?;

        // Handle texture init *before* dealing with barrier transitions so we
        // have an easier time inserting "immediate-inits" that may be required
        // by prior discards in rare cases.
        handle_src_texture_init(
            encoder,
            tracker,
            texture_memory_actions,
            device,
            source,
            copy_size,
            &src_texture,
        )?;

        let snatch_guard = device.snatchable_lock.read();

        let src_pending = tracker
            .textures
            .set_single(&src_texture, src_range, hal::TextureUses::COPY_SRC)
            .ok_or(TransferError::InvalidTexture(source.texture))?;
        let src_raw = src_texture
            .raw(&snatch_guard)
            .ok_or(TransferError::InvalidTexture(source.texture))?;
        if !src_texture.desc.usage.contains(TextureUsages::COPY_SRC) {
            return Err(TransferError::MissingCopySrcUsageFlag.into());
        }
        if src_texture.desc.sample_count != 1 {
            return Err(TransferError::InvalidSampleCount {
                sample_count: src_texture.desc.sample_count,
            }
            .into());
        }
        if source.mip_level >= src_texture.desc.mip_level_count {
            return Err(TransferError::InvalidMipLevel {
                requested: source.mip_level,
                count: src_texture.desc.mip_level_count,
            }
            .into());
        }
        let src_barrier = src_pending.map(|pending| pending.into_hal(src_raw));

        let (dst_buffer, dst_pending) = {
            let buffer_guard = hub.buffers.read();
            let dst_buffer = buffer_guard
                .get(destination.buffer)
                .map_err(|_| TransferError::InvalidBuffer(destination.buffer))?;
            tracker
                .buffers
                .set_single(dst_buffer, hal::BufferUses::COPY_DST)
                .ok_or(TransferError::InvalidBuffer(destination.buffer))?
        };
        let dst_raw = dst_buffer
            .raw
            .get(&snatch_guard)
            .ok_or(TransferError::InvalidBuffer(destination.buffer))?;
        if !dst_buffer.usage.contains(BufferUsages::COPY_DST) {
            return Err(
                TransferError::MissingCopyDstUsageFlag(Some(destination.buffer), None).into(),
            );
        }
        let dst_barrier = dst_pending.map(|pending| pending.into_hal(&dst_buffer, &snatch_guard));

        if !src_base.aspect.is_one() {
            return Err(TransferError::CopyAspectNotOne.into());
        }

        if !conv::is_valid_copy_src_texture_format(src_texture.desc.format, source.aspect) {
            return Err(TransferError::CopyFromForbiddenTextureFormat {
                format: src_texture.desc.format,
                aspect: source.aspect,
            }
            .into());
        }

        let (required_buffer_bytes_in_copy, bytes_per_array_layer) = validate_linear_texture_data(
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

        buffer_memory_init_actions.extend(dst_buffer.initialization_status.read().create_action(
            &dst_buffer,
            destination.layout.offset..(destination.layout.offset + required_buffer_bytes_in_copy),
            MemoryInitKind::ImplicitlyInitialized,
        ));

        let regions = (0..array_layer_count).map(|rel_array_layer| {
            let mut texture_base = src_base.clone();
            texture_base.array_layer += rel_array_layer;
            let mut buffer_layout = destination.layout;
            buffer_layout.offset += rel_array_layer as u64 * bytes_per_array_layer;
            hal::BufferTextureCopy {
                buffer_layout,
                texture_base,
                size: hal_copy_size,
            }
        });
        let cmd_buf_raw = encoder.open()?;
        unsafe {
            cmd_buf_raw.transition_buffers(dst_barrier.into_iter());
            cmd_buf_raw.transition_textures(src_barrier.into_iter());
            cmd_buf_raw.copy_texture_to_buffer(
                src_raw,
                hal::TextureUses::COPY_SRC,
                dst_raw,
                regions,
            );
        }
        Ok(())
    }

    pub fn command_encoder_copy_texture_to_texture<A: HalApi>(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &ImageCopyTexture,
        destination: &ImageCopyTexture,
        copy_size: &Extent3d,
    ) -> Result<(), CopyError> {
        profiling::scope!("CommandEncoder::copy_texture_to_texture");
        api_log!(
            "CommandEncoder::copy_texture_to_texture {:?} -> {:?} {copy_size:?}",
            source.texture,
            destination.texture
        );

        let hub = A::hub(self);

        let cmd_buf = CommandBuffer::get_encoder(hub, command_encoder_id)?;
        let device = &cmd_buf.device;
        if !device.is_valid() {
            return Err(TransferError::InvalidDevice(cmd_buf.device.as_info().id()).into());
        }

        let snatch_guard = device.snatchable_lock.read();

        let mut cmd_buf_data = cmd_buf.data.lock();
        let cmd_buf_data = cmd_buf_data.as_mut().unwrap();

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf_data.commands {
            list.push(TraceCommand::CopyTextureToTexture {
                src: *source,
                dst: *destination,
                size: *copy_size,
            });
        }
        let encoder = &mut cmd_buf_data.encoder;
        let tracker = &mut cmd_buf_data.trackers;
        let texture_memory_actions = &mut cmd_buf_data.texture_memory_actions;

        if copy_size.width == 0 || copy_size.height == 0 || copy_size.depth_or_array_layers == 0 {
            log::trace!("Ignoring copy_texture_to_texture of size 0");
            return Ok(());
        }

        let src_texture = hub
            .textures
            .get(source.texture)
            .map_err(|_| TransferError::InvalidTexture(source.texture))?;
        let dst_texture = hub
            .textures
            .get(destination.texture)
            .map_err(|_| TransferError::InvalidTexture(source.texture))?;

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

        let (src_copy_size, array_layer_count) =
            validate_texture_copy_range(source, &src_texture.desc, CopySide::Source, copy_size)?;
        let (dst_copy_size, _) = validate_texture_copy_range(
            destination,
            &dst_texture.desc,
            CopySide::Destination,
            copy_size,
        )?;

        let (src_range, src_tex_base) = extract_texture_selector(source, copy_size, &src_texture)?;
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

        // Handle texture init *before* dealing with barrier transitions so we
        // have an easier time inserting "immediate-inits" that may be required
        // by prior discards in rare cases.
        handle_src_texture_init(
            encoder,
            tracker,
            texture_memory_actions,
            device,
            source,
            copy_size,
            &src_texture,
        )?;
        handle_dst_texture_init(
            encoder,
            tracker,
            texture_memory_actions,
            device,
            destination,
            copy_size,
            &dst_texture,
        )?;

        let src_pending = cmd_buf_data
            .trackers
            .textures
            .set_single(&src_texture, src_range, hal::TextureUses::COPY_SRC)
            .ok_or(TransferError::InvalidTexture(source.texture))?;
        let src_raw = src_texture
            .raw(&snatch_guard)
            .ok_or(TransferError::InvalidTexture(source.texture))?;
        if !src_texture.desc.usage.contains(TextureUsages::COPY_SRC) {
            return Err(TransferError::MissingCopySrcUsageFlag.into());
        }

        //TODO: try to avoid this the collection. It's needed because both
        // `src_pending` and `dst_pending` try to hold `trackers.textures` mutably.
        let mut barriers: ArrayVec<_, 2> = src_pending
            .map(|pending| pending.into_hal(src_raw))
            .collect();

        let dst_pending = cmd_buf_data
            .trackers
            .textures
            .set_single(&dst_texture, dst_range, hal::TextureUses::COPY_DST)
            .ok_or(TransferError::InvalidTexture(destination.texture))?;
        let dst_raw = dst_texture
            .raw(&snatch_guard)
            .ok_or(TransferError::InvalidTexture(destination.texture))?;
        if !dst_texture.desc.usage.contains(TextureUsages::COPY_DST) {
            return Err(
                TransferError::MissingCopyDstUsageFlag(None, Some(destination.texture)).into(),
            );
        }

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
        let cmd_buf_raw = cmd_buf_data.encoder.open()?;
        unsafe {
            cmd_buf_raw.transition_textures(barriers.into_iter());
            cmd_buf_raw.copy_texture_to_texture(
                src_raw,
                hal::TextureUses::COPY_SRC,
                dst_raw,
                regions,
            );
        }

        Ok(())
    }
}
