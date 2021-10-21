use std::collections::hash_map::Entry;
use std::ops::Range;

use hal::CommandEncoder;

use crate::command::collect_zero_buffer_copies_for_clear_texture;
use crate::device::Device;
use crate::hub::Storage;
use crate::id;
use crate::init_tracker::*;
use crate::resource::{Buffer, Texture};
use crate::track::TextureSelector;
use crate::track::TrackerSet;
use crate::FastHashMap;

use super::{BakedCommands, DestroyedBufferError, DestroyedTextureError};

/// Surface that was discarded by `StoreOp::Discard` of a preceding renderpass.
/// Any read access to this surface needs to be preceded by a texture initialization.
#[derive(Clone)]
pub(crate) struct TextureSurfaceDiscard {
    pub texture: id::TextureId,
    pub mip_level: u32,
    pub layer: u32,
}

#[derive(Default)]
pub(crate) struct CommandBufferTextureMemoryActions {
    // init actions describe the tracker actions that we need to be executed before the command buffer is executed
    init_actions: Vec<TextureInitTrackerAction>,
    // discards describe all the discards that haven't been followed by init again within the command buffer
    // i.e. everything in this list resets the texture init state *after* the command buffer execution
    discards: Vec<TextureSurfaceDiscard>,
}

impl CommandBufferTextureMemoryActions {
    pub(crate) fn drain_init_actions(&mut self) -> std::vec::Drain<TextureInitTrackerAction> {
        self.init_actions.drain(..)
    }

    pub(crate) fn discard(&mut self, discard: TextureSurfaceDiscard) {
        self.discards.push(discard);
    }

    pub(crate) fn register_init_requirement<A: hal::Api>(
        &mut self,
        action: &TextureInitTrackerAction,
        texture_guard: &Storage<Texture<A>, id::TextureId>,
    ) {
        // TODO: Do not add MemoryInitKind::NeedsInitializedMemory to init_actions if a surface is part of the discard list
        // in that case it doesn't need to be initialized prior to the command buffer execution

        // Note that within a command buffer we may stack arbitrary memory init actions on the same texture
        // Since we react to them in sequence, they are going to be dropped again at queue submit
        self.init_actions
            .extend(match texture_guard.get(action.id) {
                Ok(texture) => texture.initialization_status.check_action(action),
                Err(_) => None,
            });

        // We expect very few discarded surfaces at any point in time which is why a linear search is likely best.
        self.discards.retain(|discarded_surface| {
            if discarded_surface.texture == action.id
                && action.range.layer_range.contains(&discarded_surface.layer)
                && action
                    .range
                    .mip_range
                    .contains(&discarded_surface.mip_level)
            {
                if let MemoryInitKind::NeedsInitializedMemory = action.kind {
                    todo!("need to immediately initialize surface!");
                    todo!("mark surface as implicitly initialized (this is relevant because it might have been uninitialized prior to discarding");
                }
                false
            } else {
                true
            }
        });
    }
}

impl<A: hal::Api> BakedCommands<A> {
    // inserts all buffer initializations that are going to be needed for executing the commands and updates resource init states accordingly
    pub(crate) fn initialize_buffer_memory(
        &mut self,
        device_tracker: &mut TrackerSet,
        buffer_guard: &mut Storage<Buffer<A>, id::BufferId>,
    ) -> Result<(), DestroyedBufferError> {
        // Gather init ranges for each buffer so we can collapse them.
        // It is not possible to do this at an earlier point since previously executed command buffer change the resource init state.
        let mut uninitialized_ranges_per_buffer = FastHashMap::default();
        for buffer_use in self.buffer_memory_init_actions.drain(..) {
            let buffer = buffer_guard
                .get_mut(buffer_use.id)
                .map_err(|_| DestroyedBufferError(buffer_use.id))?;

            // align the end to 4
            let end_remainder = buffer_use.range.end % wgt::COPY_BUFFER_ALIGNMENT;
            let end = if end_remainder == 0 {
                buffer_use.range.end
            } else {
                buffer_use.range.end + wgt::COPY_BUFFER_ALIGNMENT - end_remainder
            };
            let uninitialized_ranges = buffer
                .initialization_status
                .drain(buffer_use.range.start..end);

            match buffer_use.kind {
                MemoryInitKind::ImplicitlyInitialized => {
                    uninitialized_ranges.for_each(drop);
                }
                MemoryInitKind::NeedsInitializedMemory => {
                    match uninitialized_ranges_per_buffer.entry(buffer_use.id) {
                        Entry::Vacant(e) => {
                            e.insert(
                                uninitialized_ranges.collect::<Vec<Range<wgt::BufferAddress>>>(),
                            );
                        }
                        Entry::Occupied(mut e) => {
                            e.get_mut().extend(uninitialized_ranges);
                        }
                    }
                }
            }
        }

        for (buffer_id, mut ranges) in uninitialized_ranges_per_buffer {
            // Collapse touching ranges.
            ranges.sort_by(|a, b| a.start.cmp(&b.start));
            for i in (1..ranges.len()).rev() {
                assert!(ranges[i - 1].end <= ranges[i].start); // The memory init tracker made sure of this!
                if ranges[i].start == ranges[i - 1].end {
                    ranges[i - 1].end = ranges[i].end;
                    ranges.swap_remove(i); // Ordering not important at this point
                }
            }

            // Don't do use_replace since the buffer may already no longer have a ref_count.
            // However, we *know* that it is currently in use, so the tracker must already know about it.
            let transition = device_tracker.buffers.change_replace_tracked(
                id::Valid(buffer_id),
                (),
                hal::BufferUses::COPY_DST,
            );

            let buffer = buffer_guard
                .get_mut(buffer_id)
                .map_err(|_| DestroyedBufferError(buffer_id))?;
            let raw_buf = buffer.raw.as_ref().ok_or(DestroyedBufferError(buffer_id))?;

            unsafe {
                self.encoder
                    .transition_buffers(transition.map(|pending| pending.into_hal(buffer)));
            }

            for range in ranges.iter() {
                assert!(range.start % wgt::COPY_BUFFER_ALIGNMENT == 0, "Buffer {:?} has an uninitialized range with a start not aligned to 4 (start was {})", raw_buf, range.start);
                assert!(range.end % wgt::COPY_BUFFER_ALIGNMENT == 0, "Buffer {:?} has an uninitialized range with an end not aligned to 4 (end was {})", raw_buf, range.end);

                unsafe {
                    self.encoder.clear_buffer(raw_buf, range.clone());
                }
            }
        }
        Ok(())
    }

    // inserts all texture initializations that are going to be needed for executing the commands and updates resource init states accordingly
    // any textures that are left discarded by this command buffer will be marked as uninitialized
    pub(crate) fn initialize_texture_memory(
        &mut self,
        device_tracker: &mut TrackerSet,
        texture_guard: &mut Storage<Texture<A>, id::TextureId>,
        device: &Device<A>,
    ) -> Result<(), DestroyedTextureError> {
        let mut ranges: Vec<TextureInitRange> = Vec::new();
        for texture_use in self.texture_memory_actions.drain_init_actions() {
            let texture = texture_guard
                .get_mut(texture_use.id)
                .map_err(|_| DestroyedTextureError(texture_use.id))?;

            let use_range = texture_use.range;
            let affected_mip_trackers = texture
                .initialization_status
                .mips
                .iter_mut()
                .enumerate()
                .skip(use_range.mip_range.start as usize)
                .take((use_range.mip_range.end - use_range.mip_range.start) as usize);

            match texture_use.kind {
                MemoryInitKind::ImplicitlyInitialized => {
                    for (_, mip_tracker) in affected_mip_trackers {
                        mip_tracker
                            .drain(use_range.layer_range.clone())
                            .for_each(drop);
                    }
                }
                MemoryInitKind::NeedsInitializedMemory => {
                    ranges.clear();
                    for (mip_level, mip_tracker) in affected_mip_trackers {
                        for layer_range in mip_tracker.drain(use_range.layer_range.clone()) {
                            ranges.push(TextureInitRange {
                                mip_range: mip_level as u32..(mip_level as u32 + 1),
                                layer_range,
                            })
                        }
                    }

                    let raw_texture = texture
                        .inner
                        .as_raw()
                        .ok_or(DestroyedTextureError(texture_use.id))?;

                    debug_assert!(texture.hal_usage.contains(hal::TextureUses::COPY_DST),
                            "Every texture needs to have the COPY_DST flag. Otherwise we can't ensure initialized memory!");

                    let mut zero_buffer_copy_regions = Vec::new();
                    for range in &ranges {
                        // Don't do use_replace since the texture may already no longer have a ref_count.
                        // However, we *know* that it is currently in use, so the tracker must already know about it.
                        let transition = device_tracker.textures.change_replace_tracked(
                            id::Valid(texture_use.id),
                            TextureSelector {
                                levels: range.mip_range.clone(),
                                layers: range.layer_range.clone(),
                            },
                            hal::TextureUses::COPY_DST,
                        );

                        collect_zero_buffer_copies_for_clear_texture(
                            &texture.desc,
                            device.alignments.buffer_copy_pitch.get() as u32,
                            range.mip_range.clone(),
                            range.layer_range.clone(),
                            &mut zero_buffer_copy_regions,
                        );
                        unsafe {
                            self.encoder.transition_textures(
                                transition.map(|pending| pending.into_hal(texture)),
                            );
                        }
                    }

                    if zero_buffer_copy_regions.len() > 0 {
                        unsafe {
                            self.encoder.copy_buffer_to_texture(
                                &device.zero_buffer,
                                raw_texture,
                                zero_buffer_copy_regions.into_iter(),
                            );
                        }
                    }
                }
            }
        }

        // Now that all buffers/textures have the proper init state for before cmdbuf start, we discard init states for textures it left discarded after its execution.
        for surface_discard in self.texture_memory_actions.discards.iter() {
            let texture = texture_guard
                .get_mut(surface_discard.texture)
                .map_err(|_| DestroyedTextureError(surface_discard.texture))?;
            texture
                .initialization_status
                .discard(surface_discard.mip_level, surface_discard.layer);
        }

        Ok(())
    }
}
