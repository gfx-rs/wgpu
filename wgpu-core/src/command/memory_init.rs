use alloc::{
    sync::Arc,
    vec::{Drain, Vec},
};
use core::ops::Range;

use hashbrown::hash_map::Entry;

use crate::{
    device::{Device, DeviceError},
    init_tracker::*,
    resource::{ParentDevice, RawResourceAccess, Texture, Trackable},
    snatch::SnatchGuard,
    track::{DeviceTracker, TextureTracker},
    FastHashMap,
};

use super::{clear_texture, BakedCommands, ClearError};

/// Surface that was discarded by `StoreOp::Discard` of a preceding renderpass.
/// Any read access to this surface needs to be preceded by a texture initialization.
#[derive(Clone)]
pub(crate) struct TextureSurfaceDiscard {
    pub texture: Arc<Texture>,
    pub mip_level: u32,

    /// For 3D textures, this field is a depth slice, otherwise, it is an array
    /// layer. Unlike the primary initialization tracker, we _do_ track
    /// individual discarded depth slices during encoding of a command buffer.
    pub layer_or_depth_slice: u32,
}

pub(crate) type SurfacesInDiscardState = Vec<TextureSurfaceDiscard>;

#[derive(Default)]
pub(crate) struct CommandBufferTextureMemoryActions {
    /// The tracker actions that we need to be executed before the command
    /// buffer is executed.
    init_actions: Vec<TextureInitTrackerAction>,
    /// Tracks surfaces that were previously discarded within this command buffer.
    ///
    /// If a later pass reads from one of these surfaces, we must insert an immediate
    /// clear operation in the command sequence. (Typically, memory initialization is done
    /// in a dedicated pass prepended to the entire command buffer, but the discards we are
    /// tracking here occur after that). Any discarded surfaces that are not reinitialized
    /// prior to the end of the command buffer, will have their `initialization_status` set
    /// to uninitialized at that point, except for depth slices, which must be reinitialized
    /// prior to the end of the command buffer.
    ///
    /// We do a linear scan of the discarded surface list for _each_ initialization
    /// action, i.e., we assume that most of the time there are no discarded surfaces.
    /// If this list has more than a few items, performance will suffer.
    discards: Vec<TextureSurfaceDiscard>,
}

impl CommandBufferTextureMemoryActions {
    pub(crate) fn drain_init_actions(&mut self) -> Drain<'_, TextureInitTrackerAction> {
        self.init_actions.drain(..)
    }

    pub(crate) fn discard(&mut self, discard: TextureSurfaceDiscard) {
        self.discards.push(discard);
    }

    /// Registers a [`TextureInitTrackerAction`].
    ///
    /// Returns previously discarded surfaces that need to be initialized
    /// *immediately*. Only returns a non-empty list if `action.kind` is
    /// [`MemoryInitKind::NeedsInitializedMemory`]. These surfaces are removed
    /// from the pending discard list, so the caller takes on the obligation to
    /// clear them (via [`fixup_discarded_surfaces`]).
    ///
    /// `depth_slices` is the range of depth slices that `action` physically
    /// accesses, or `None` if the access is to all slices of a 3D texture or
    /// to some other texture type. The depth slice information does not flow to
    /// the global init tracker, whose granularity is a whole mip level, but it
    /// is important in deciding which pending discards have to be repaired
    /// ahead of this action, as opposed to at the end of the command buffer.
    #[must_use]
    pub(crate) fn register_init_action(
        &mut self,
        action: &TextureInitTrackerAction,
        depth_slices: Option<Range<u32>>,
    ) -> SurfacesInDiscardState {
        let is_3d = action.texture.desc.dimension == wgt::TextureDimension::D3;
        debug_assert!(depth_slices.is_none() || is_3d);

        // Texture subresources from `self.discards` that were discarded earlier in this
        // command buffer and which the present `action` requires be initialized. These
        // require inline initialization, which will be done by `fixup_discarded_surfaces`.
        let mut immediately_necessary_clears = SurfacesInDiscardState::new();

        // Note that within a command buffer we may stack arbitrary memory init
        // actions on the same texture Since we react to them in sequence, they
        // are going to be dropped again at queue submit
        //
        // We don't need to add MemoryInitKind::NeedsInitializedMemory to
        // init_actions if a surface is part of the discard list. But that would
        // mean splitting up the action which is more than we'd win here.
        self.init_actions.extend(
            action
                .texture
                .initialization_status
                .read()
                .check_action(action),
        );

        // We expect very few discarded surfaces at any point in time which is
        // why a simple linear search is likely best. (i.e. most of the time
        // self.discards is empty!)
        let init_actions = &mut self.init_actions;
        self.discards.retain(|discarded_surface| {
            if !discarded_surface.texture.is_equal(&action.texture)
                || !action
                    .range
                    .mip_range
                    .contains(&discarded_surface.mip_level)
            {
                return true;
            }

            let overlaps_discard = if is_3d {
                // The `layer_range` for a `TextureInitTrackerAction` does not identify
                // depth slices, so the caller passed that information separately.
                depth_slices
                    .as_ref()
                    .is_none_or(|slices| slices.contains(&discarded_surface.layer_or_depth_slice))
            } else {
                action
                    .range
                    .layer_range
                    .contains(&discarded_surface.layer_or_depth_slice)
            };
            if !overlaps_discard {
                return true;
            }

            if let MemoryInitKind::NeedsInitializedMemory = action.kind {
                immediately_necessary_clears.push(discarded_surface.clone());

                // Mark surface as implicitly initialized. This matters for non-3D textures
                // where the discarded layer range may differ from the action layer range,
                // and may have been uninitialized prior to discarding. For 3D textures,
                // init state does not vary per layer, so we either emitted an init action
                // above for the whole mip level, or it was already initialized and none
                // is necessary.
                if !is_3d {
                    let layer = discarded_surface.layer_or_depth_slice;
                    init_actions.push(TextureInitTrackerAction {
                        texture: discarded_surface.texture.clone(),
                        range: TextureInitRange {
                            mip_range: discarded_surface.mip_level
                                ..(discarded_surface.mip_level + 1),
                            layer_range: layer..(layer + 1),
                        },
                        kind: MemoryInitKind::ImplicitlyInitialized,
                    });
                }
            }
            false
        });

        immediately_necessary_clears
    }

    // Shortcut for register_init_action when it is known that the action is an
    // implicit init, not requiring any immediate resource init.
    pub(crate) fn register_implicit_init(
        &mut self,
        texture: &Arc<Texture>,
        range: TextureInitRange,
    ) {
        let must_be_empty = self.register_init_action(
            &TextureInitTrackerAction {
                texture: texture.clone(),
                range,
                kind: MemoryInitKind::ImplicitlyInitialized,
            },
            None,
        );
        assert!(must_be_empty.is_empty());
    }
}

// Utility function that takes discarded surfaces from (several calls to)
// register_init_action and initializes them on the spot.
//
// Takes care of barriers as well!
pub(crate) fn fixup_discarded_surfaces<InitIter: Iterator<Item = TextureSurfaceDiscard>>(
    inits: InitIter,
    encoder: &mut dyn hal::DynCommandEncoder,
    texture_tracker: &mut TextureTracker,
    device: &Device,
    snatch_guard: &SnatchGuard<'_>,
) {
    for init in inits {
        let (layer_range, depth_slice) = if init.texture.desc.dimension == wgt::TextureDimension::D3
        {
            (0..1, Some(init.layer_or_depth_slice))
        } else {
            (
                init.layer_or_depth_slice..(init.layer_or_depth_slice + 1),
                None,
            )
        };
        clear_texture(
            &init.texture,
            TextureInitRange {
                mip_range: init.mip_level..(init.mip_level + 1),
                layer_range,
            },
            depth_slice,
            encoder,
            texture_tracker,
            &device.alignments,
            device.zero_buffer.as_ref(),
            snatch_guard,
            device.instance_flags,
        )
        .unwrap();
    }
}

impl BakedCommands {
    /// Initialize buffers.
    ///
    /// Inserts all buffer initializations that are going to be needed for
    /// executing the commands, and updates resource init states accordingly.
    ///
    /// The caller is responsible for checking that any buffer this may touch has not been
    /// destroyed, and must have done that check under the same snatch guard that is passed
    /// to this function.
    ///
    /// # Panics
    /// If a destroyed buffer is encountered.
    pub(crate) fn initialize_buffer_memory(
        &mut self,
        device_tracker: &mut DeviceTracker,
        snatch_guard: &SnatchGuard<'_>,
    ) {
        profiling::scope!("initialize_buffer_memory");

        // Gather init ranges for each buffer so we can collapse them.
        // It is not possible to do this at an earlier point since previously
        // executed command buffer change the resource init state.
        let mut uninitialized_ranges_per_buffer = FastHashMap::default();
        for buffer_use in self.buffer_memory_init_actions.drain(..) {
            let mut initialization_status = buffer_use.buffer.initialization_status.write();

            // align the end to 4
            let end_remainder = buffer_use.range.end % wgt::COPY_BUFFER_ALIGNMENT;
            let end = if end_remainder == 0 {
                buffer_use.range.end
            } else {
                buffer_use.range.end + wgt::COPY_BUFFER_ALIGNMENT - end_remainder
            };
            let uninitialized_ranges = initialization_status.drain(buffer_use.range.start..end);

            match buffer_use.kind {
                MemoryInitKind::ImplicitlyInitialized => {}
                MemoryInitKind::NeedsInitializedMemory => {
                    match uninitialized_ranges_per_buffer.entry(buffer_use.buffer.tracker_index()) {
                        Entry::Vacant(e) => {
                            e.insert((
                                buffer_use.buffer.clone(),
                                uninitialized_ranges.collect::<Vec<Range<wgt::BufferAddress>>>(),
                            ));
                        }
                        Entry::Occupied(mut e) => {
                            e.get_mut().1.extend(uninitialized_ranges);
                        }
                    }
                }
            }
        }

        for (buffer, mut ranges) in uninitialized_ranges_per_buffer.into_values() {
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
                .set_single(&buffer, wgt::BufferUses::COPY_DST);

            let raw_buf = buffer
                .try_raw(snatch_guard)
                .expect("attempt to initialize a destroyed buffer");

            unsafe {
                self.encoder.raw.transition_buffers(
                    transition
                        .map(|pending| pending.into_hal(&buffer, snatch_guard))
                        .as_slice(),
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
                    self.encoder.raw.clear_buffer(raw_buf, range.clone());
                }
            }
        }
    }

    /// Initialize textures.
    ///
    /// Inserts all texture initializations that are going to be needed for
    /// executing the commands, and updates resource init states accordingly. Any
    /// non-3D textures that are left discarded by this command buffer will be marked as
    /// uninitialized, and a list of any 3D depth slices is returned, to be reinitialized
    /// just prior to the end of the command buffer.
    ///
    /// The caller is responsible for checking that any texture this may touch has not been
    /// destroyed, and must have done that check under the same snatch guard that is passed
    /// to this function.
    ///
    /// Note that any error returned from this function will become device loss in
    /// [`crate::device::queue::Queue::submit`].
    ///
    /// # Panics
    /// If a destroyed texture is encountered.
    pub(crate) fn initialize_texture_memory(
        &mut self,
        device_tracker: &mut DeviceTracker,
        device: &Device,
        snatch_guard: &SnatchGuard<'_>,
    ) -> Result<SurfacesInDiscardState, ClearError> {
        profiling::scope!("initialize_texture_memory");

        let mut depth_slice_discards = SurfacesInDiscardState::new();

        let mut ranges: Vec<TextureInitRange> = Vec::new();
        for texture_use in self.texture_memory_actions.drain_init_actions() {
            {
                let mut initialization_status = texture_use.texture.initialization_status.write();
                let use_range = texture_use.range;
                let affected_mip_trackers = initialization_status
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
            }

            // TODO: Could we attempt some range collapsing here?
            for range in ranges.drain(..) {
                let clear_result = clear_texture(
                    &texture_use.texture,
                    range,
                    None,
                    self.encoder.raw.as_mut(),
                    &mut device_tracker.textures,
                    &device.alignments,
                    device.zero_buffer.as_ref(),
                    snatch_guard,
                    device.instance_flags,
                );

                // We panic on destroyed textures for symmetry with buffer
                // initialization. It should not happen, but supposing it did,
                // it would also be fine to return the error and lose the
                // device in queue submit.
                if matches!(clear_result, Err(ClearError::DestroyedResource(_))) {
                    panic!("attempt to initialize a destroyed texture");
                } else {
                    clear_result?;
                }
            }
        }

        // Process any surfaces that remain in discarded state after
        // the command buffer executes.
        for surface_discard in self.texture_memory_actions.discards.drain(..) {
            if surface_discard.texture.desc.dimension == wgt::TextureDimension::D3 {
                // Depth slices are below the resolution of the init tracker, so
                // collect a list of them to be initialized just prior to the
                // end of the command buffer.
                //
                // We could optimize this by checking whether the entire mip is
                // uninitialized (command buffer either did Clear+Discard when it was
                // already uninitialized, or discarded every slice), and if so, ignore the
                // pending discard, but it's not clear that happens enough for the
                // optimization to be worth it.
                depth_slice_discards.push(surface_discard);
            } else {
                // Anything else, record the discarded state in the initialization tracker.
                surface_discard
                    .texture
                    .initialization_status
                    .write()
                    .discard(
                        surface_discard.mip_level,
                        surface_discard.layer_or_depth_slice,
                    );
            }
        }

        Ok(depth_slice_discards)
    }

    /// Reinitialize any depth slices that were discarded during the command buffer and not
    /// subsequently reinitialized.
    ///
    /// This is necessary because the initialization tracker does not track the status of
    /// individual depth slices.
    ///
    /// We do not optimize the case where _every_ depth slice of a 3D texture is discarded.
    pub(crate) fn initialize_discarded_depth_slices(
        &mut self,
        discards: SurfacesInDiscardState,
        device_tracker: &mut DeviceTracker,
        device: &Device,
        snatch_guard: &SnatchGuard<'_>,
    ) -> Result<(), ClearError> {
        for discard in discards {
            assert!(
                discard.texture.desc.dimension == wgt::TextureDimension::D3,
                "unexpected texture dimension {:?} in initialize_discarded_depth_slices",
                discard.texture.desc.dimension,
            );
            let range = TextureInitRange {
                mip_range: discard.mip_level..(discard.mip_level + 1),
                layer_range: 0..1,
            };
            let clear_result = clear_texture(
                &discard.texture,
                range,
                Some(discard.layer_or_depth_slice),
                self.encoder.raw.as_mut(),
                &mut device_tracker.textures,
                &device.alignments,
                device.zero_buffer.as_ref(),
                snatch_guard,
                device.instance_flags,
            );
            // We panic on destroyed textures for symmetry with the main buffer
            // and initialization pass. It should not happen, but supposing it
            // did, it would also be fine to return the error and lose the
            // device in queue submit.
            if matches!(clear_result, Err(ClearError::DestroyedResource(_))) {
                panic!("attempt to initialize a destroyed texture");
            } else {
                clear_result?;
            }
        }

        Ok(())
    }

    pub(crate) fn process_deferred_query_set_resolves(
        &mut self,
        device: &Device,
        snatch_guard: &SnatchGuard<'_>,
    ) -> Result<(), DeviceError> {
        profiling::scope!("process_deferred_query_set_resolves");

        for mut resolve in self.deferred_query_set_resolves.drain(..).rev() {
            let raw_dst = resolve.dst_buffer.try_raw(snatch_guard).unwrap();
            let raw_query_set = resolve.query_set.try_raw(snatch_guard).unwrap();

            let raw_encoder = self.encoder.open_pass(crate::hal_label(
                Some("(wgpu internal) Deferred query set resolve"),
                device.instance_flags,
            ))?;

            let initialized_slots_guard = resolve.query_set.initialized_slots.lock();
            let initialized_slots =
                if let Some(query_set_writes) = resolve.query_set_writes.as_mut() {
                    query_set_writes.or(&initialized_slots_guard);
                    &*query_set_writes
                } else {
                    &*initialized_slots_guard
                };

            let mut start = resolve.start_query;
            while start < resolve.end_query {
                let is_initialized = initialized_slots[start as usize];
                let end = (start + 1..resolve.end_query)
                    .find(|&i| initialized_slots[i as usize] != is_initialized)
                    .unwrap_or(resolve.end_query);

                let byte_offset = resolve.destination_offset
                    + (start - resolve.start_query) as u64 * resolve.stride;
                let byte_len = (end - start) as u64 * resolve.stride;

                if is_initialized {
                    unsafe {
                        raw_encoder.copy_query_results(
                            raw_query_set,
                            start..end,
                            raw_dst,
                            byte_offset,
                            wgt::BufferSize::new_unchecked(resolve.stride),
                        );
                    }
                } else {
                    unsafe {
                        raw_encoder.clear_buffer(raw_dst, byte_offset..byte_offset + byte_len);
                    }
                }

                start = end;
            }
            drop(initialized_slots_guard);

            self.encoder.close_and_insert_at(resolve.insertion_point)?;
        }

        // Update query set initialization state.
        for query_set in self.trackers.query_sets.used_resources() {
            if let Some(slots) = self.query_set_writes.get(&query_set.tracker_index()) {
                let mut initialized = query_set.initialized_slots.lock();
                initialized.or(slots);
            }
        }

        Ok(())
    }
}
