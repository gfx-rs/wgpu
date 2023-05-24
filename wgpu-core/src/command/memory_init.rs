use std::{collections::hash_map::Entry, ops::Range, vec::Drain};

use hal::CommandEncoder;

use crate::{
    device::Device,
    hal_api::HalApi,
    id::{self, TextureId},
    init_tracker::*,
    resource::{Buffer, Texture},
    storage::Storage,
    track::{TextureTracker, Tracker},
    FastHashMap,
};

use super::{clear::clear_texture, BakedCommands, DestroyedBufferError, DestroyedTextureError};

/// Surface that was discarded by `StoreOp::Discard` of a preceding renderpass.
/// Any read access to this surface needs to be preceded by a texture initialization.
#[derive(Clone)]
pub(crate) struct TextureSurfaceDiscard {
    pub texture: TextureId,
    pub mip_level: u32,
    pub layer: u32,
}

pub(crate) type SurfacesInDiscardState = Vec<TextureSurfaceDiscard>;

#[derive(Default)]
pub(crate) struct CommandBufferTextureMemoryActions {
    /// The tracker actions that we need to be executed before the command
    /// buffer is executed.
    init_actions: Vec<TextureInitTrackerAction>,
    /// All the discards that haven't been followed by init again within the
    /// command buffer i.e. everything in this list resets the texture init
    /// state *after* the command buffer execution
    discards: Vec<TextureSurfaceDiscard>,
}

impl CommandBufferTextureMemoryActions {
    pub(crate) fn drain_init_actions(&mut self) -> Drain<TextureInitTrackerAction> {
        self.init_actions.drain(..)
    }

    pub(crate) fn discard(&mut self, discard: TextureSurfaceDiscard) {
        self.discards.push(discard);
    }

    // Registers a TextureInitTrackerAction.
    // Returns previously discarded surface that need to be initialized *immediately* now.
    // Only returns a non-empty list if action is MemoryInitKind::NeedsInitializedMemory.
    #[must_use]
    pub(crate) fn register_init_action<A: hal::Api>(
        &mut self,
        action: &TextureInitTrackerAction,
        texture_guard: &Storage<Texture<A>, TextureId>,
    ) -> SurfacesInDiscardState {
        let mut immediately_necessary_clears = SurfacesInDiscardState::new();

        // Note that within a command buffer we may stack arbitrary memory init
        // actions on the same texture Since we react to them in sequence, they
        // are going to be dropped again at queue submit
        //
        // We don't need to add MemoryInitKind::NeedsInitializedMemory to
        // init_actions if a surface is part of the discard list. But that would
        // mean splitting up the action which is more than we'd win here.
        self.init_actions
            .extend(match texture_guard.get(action.id) {
                Ok(texture) => texture.initialization_status.check_action(action),
                Err(_) => return immediately_necessary_clears, // texture no longer exists
            });

        // We expect very few discarded surfaces at any point in time which is
        // why a simple linear search is likely best. (i.e. most of the time
        // self.discards is empty!)
        let init_actions = &mut self.init_actions;
        self.discards.retain(|discarded_surface| {
            if discarded_surface.texture == action.id
                && action.range.layer_range.contains(&discarded_surface.layer)
                && action
                    .range
                    .mip_range
                    .contains(&discarded_surface.mip_level)
            {
                if let MemoryInitKind::NeedsInitializedMemory = action.kind {
                    immediately_necessary_clears.push(discarded_surface.clone());

                    // Mark surface as implicitly initialized (this is relevant
                    // because it might have been uninitialized prior to
                    // discarding
                    init_actions.push(TextureInitTrackerAction {
                        id: discarded_surface.texture,
                        range: TextureInitRange {
                            mip_range: discarded_surface.mip_level
                                ..(discarded_surface.mip_level + 1),
                            layer_range: discarded_surface.layer..(discarded_surface.layer + 1),
                        },
                        kind: MemoryInitKind::ImplicitlyInitialized,
                    });
                }
                false
            } else {
                true
            }
        });

        immediately_necessary_clears
    }

    // Shortcut for register_init_action when it is known that the action is an
    // implicit init, not requiring any immediate resource init.
    pub(crate) fn register_implicit_init<A: hal::Api>(
        &mut self,
        id: id::Valid<TextureId>,
        range: TextureInitRange,
        texture_guard: &Storage<Texture<A>, TextureId>,
    ) {
        let must_be_empty = self.register_init_action(
            &TextureInitTrackerAction {
                id: id.0,
                range,
                kind: MemoryInitKind::ImplicitlyInitialized,
            },
            texture_guard,
        );
        assert!(must_be_empty.is_empty());
    }
}

// Utility function that takes discarded surfaces from (several calls to)
// register_init_action and initializes them on the spot.
//
// Takes care of barriers as well!
pub(crate) fn fixup_discarded_surfaces<
    A: HalApi,
    InitIter: Iterator<Item = TextureSurfaceDiscard>,
>(
    inits: InitIter,
    encoder: &mut A::CommandEncoder,
    texture_guard: &Storage<Texture<A>, TextureId>,
    texture_tracker: &mut TextureTracker<A>,
    device: &Device<A>,
) {
    for init in inits {
        clear_texture(
            texture_guard,
            id::Valid(init.texture),
            TextureInitRange {
                mip_range: init.mip_level..(init.mip_level + 1),
                layer_range: init.layer..(init.layer + 1),
            },
            encoder,
            texture_tracker,
            &device.alignments,
            &device.zero_buffer,
        )
        .unwrap();
    }
}

impl<A: HalApi> BakedCommands<A> {
    // inserts all buffer initializations that are going to be needed for
    // executing the commands and updates resource init states accordingly
    pub(crate) fn initialize_buffer_memory(
        &mut self,
        device_tracker: &mut Tracker<A>,
        buffer_guard: &mut Storage<Buffer<A>, id::BufferId>,
    ) -> Result<(), DestroyedBufferError> {
        // Gather init ranges for each buffer so we can collapse them.
        // It is not possible to do this at an earlier point since previously
        // executed command buffer change the resource init state.
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
                MemoryInitKind::ImplicitlyInitialized => {}
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
            ranges.sort_by_key(|r| r.start);
            for i in (1..ranges.len()).rev() {
                // The memory init tracker made sure of this!
                assert!(ranges[i - 1].end <= ranges[i].start);
                if ranges[i].start == ranges[i - 1].end {
                    ranges[i - 1].end = ranges[i].end;
                    ranges.swap_remove(i); // Ordering not important at this point
                }
            }

            // Don't do use_replace since the buffer may already no longer have
            // a ref_count.
            //
            // However, we *know* that it is currently in use, so the tracker
            // must already know about it.
            let transition = device_tracker
                .buffers
                .set_single(buffer_guard, buffer_id, hal::BufferUses::COPY_DST)
                .unwrap()
                .1;

            let buffer = buffer_guard
                .get_mut(buffer_id)
                .map_err(|_| DestroyedBufferError(buffer_id))?;
            let raw_buf = buffer.raw.as_ref().ok_or(DestroyedBufferError(buffer_id))?;

            unsafe {
                self.encoder.transition_buffers(
                    transition
                        .map(|pending| pending.into_hal(buffer))
                        .into_iter(),
                );
            }

            for range in ranges.iter() {
                assert!(
                    range.start % wgt::COPY_BUFFER_ALIGNMENT == 0,
                    "Buffer {:?} has an uninitialized range with a start \
                         not aligned to 4 (start was {})",
                    raw_buf,
                    range.start
                );
                assert!(
                    range.end % wgt::COPY_BUFFER_ALIGNMENT == 0,
                    "Buffer {:?} has an uninitialized range with an end \
                         not aligned to 4 (end was {})",
                    raw_buf,
                    range.end
                );

                unsafe {
                    self.encoder.clear_buffer(raw_buf, range.clone());
                }
            }
        }
        Ok(())
    }

    // inserts all texture initializations that are going to be needed for
    // executing the commands and updates resource init states accordingly any
    // textures that are left discarded by this command buffer will be marked as
    // uninitialized
    pub(crate) fn initialize_texture_memory(
        &mut self,
        device_tracker: &mut Tracker<A>,
        texture_guard: &mut Storage<Texture<A>, TextureId>,
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
                        mip_tracker.drain(use_range.layer_range.clone());
                    }
                }
                MemoryInitKind::NeedsInitializedMemory => {
                    for (mip_level, mip_tracker) in affected_mip_trackers {
                        for layer_range in mip_tracker.drain(use_range.layer_range.clone()) {
                            ranges.push(TextureInitRange {
                                mip_range: (mip_level as u32)..(mip_level as u32 + 1),
                                layer_range,
                            });
                        }
                    }
                }
            }

            // TODO: Could we attempt some range collapsing here?
            for range in ranges.drain(..) {
                clear_texture(
                    texture_guard,
                    id::Valid(texture_use.id),
                    range,
                    &mut self.encoder,
                    &mut device_tracker.textures,
                    &device.alignments,
                    &device.zero_buffer,
                )
                .unwrap();
            }
        }

        // Now that all buffers/textures have the proper init state for before
        // cmdbuf start, we discard init states for textures it left discarded
        // after its execution.
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
