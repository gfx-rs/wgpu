use crate::{
    binding_model::{
        BindGroup, BindGroupLayouts, LateMinBufferBindingSizeMismatch, PipelineLayout,
    },
    device::SHADER_STAGE_COUNT,
    hal_api::HalApi,
    id::{BindGroupId, PipelineLayoutId, Valid},
    pipeline::LateSizedBufferGroup,
    storage::Storage,
    Stored,
};

use arrayvec::ArrayVec;

type BindGroupMask = u8;

mod compat {
    use crate::{
        binding_model::BindGroupLayouts,
        id::{BindGroupLayoutId, Valid},
    };
    use std::ops::Range;

    #[derive(Debug, Default)]
    struct Entry {
        assigned: Option<Valid<BindGroupLayoutId>>,
        expected: Option<Valid<BindGroupLayoutId>>,
    }

    impl Entry {
        fn is_active(&self) -> bool {
            self.assigned.is_some() && self.expected.is_some()
        }

        fn is_valid<A: hal::Api>(&self, bind_group_layouts: &BindGroupLayouts<A>) -> bool {
            if self.expected.is_none() || self.expected == self.assigned {
                return true;
            }

            if let Some(id) = self.assigned {
                return bind_group_layouts[id].as_duplicate() == self.expected;
            }

            false
        }
    }

    #[derive(Debug)]
    pub(crate) struct BoundBindGroupLayouts {
        entries: [Entry; hal::MAX_BIND_GROUPS],
    }

    impl BoundBindGroupLayouts {
        pub fn new() -> Self {
            Self {
                entries: Default::default(),
            }
        }

        fn make_range(&self, start_index: usize) -> Range<usize> {
            // find first incompatible entry
            let end = self
                .entries
                .iter()
                .position(|e| e.expected.is_none() || e.assigned != e.expected)
                .unwrap_or(self.entries.len());
            start_index..end.max(start_index)
        }

        pub fn update_expectations(
            &mut self,
            expectations: &[Valid<BindGroupLayoutId>],
        ) -> Range<usize> {
            let start_index = self
                .entries
                .iter()
                .zip(expectations)
                .position(|(e, &expect)| e.expected != Some(expect))
                .unwrap_or(expectations.len());
            for (e, &expect) in self.entries[start_index..]
                .iter_mut()
                .zip(expectations[start_index..].iter())
            {
                e.expected = Some(expect);
            }
            for e in self.entries[expectations.len()..].iter_mut() {
                e.expected = None;
            }
            self.make_range(start_index)
        }

        pub fn assign(&mut self, index: usize, value: Valid<BindGroupLayoutId>) -> Range<usize> {
            self.entries[index].assigned = Some(value);
            self.make_range(index)
        }

        pub fn list_active(&self) -> impl Iterator<Item = usize> + '_ {
            self.entries
                .iter()
                .enumerate()
                .filter_map(|(i, e)| if e.is_active() { Some(i) } else { None })
        }

        pub fn invalid_mask<A: hal::Api>(
            &self,
            bind_group_layouts: &BindGroupLayouts<A>,
        ) -> super::BindGroupMask {
            self.entries.iter().enumerate().fold(0, |mask, (i, entry)| {
                if entry.is_valid(bind_group_layouts) {
                    mask
                } else {
                    mask | 1u8 << i
                }
            })
        }
    }

    #[test]
    fn test_compatibility() {
        fn id(val: u32) -> Valid<BindGroupLayoutId> {
            BindGroupLayoutId::dummy(val)
        }

        let mut man = BoundBindGroupLayouts::new();
        man.entries[0] = Entry {
            expected: Some(id(3)),
            assigned: Some(id(2)),
        };
        man.entries[1] = Entry {
            expected: Some(id(1)),
            assigned: Some(id(1)),
        };
        man.entries[2] = Entry {
            expected: Some(id(4)),
            assigned: Some(id(5)),
        };
        // check that we rebind [1] after [0] became compatible
        assert_eq!(man.assign(0, id(3)), 0..2);
        // check that nothing is rebound
        assert_eq!(man.update_expectations(&[id(3), id(2)]), 1..1);
        // check that [1] and [2] are rebound on expectations change
        assert_eq!(man.update_expectations(&[id(3), id(1), id(5)]), 1..3);
        // reset the first two bindings
        assert_eq!(man.update_expectations(&[id(4), id(6), id(5)]), 0..0);
        // check that nothing is rebound, even if there is a match,
        // since earlier binding is incompatible.
        assert_eq!(man.assign(1, id(6)), 1..1);
        // finally, bind everything
        assert_eq!(man.assign(0, id(4)), 0..3);
    }
}

#[derive(Debug)]
struct LateBufferBinding {
    shader_expect_size: wgt::BufferAddress,
    bound_size: wgt::BufferAddress,
}

#[derive(Debug, Default)]
pub(super) struct EntryPayload {
    pub(super) group_id: Option<Stored<BindGroupId>>,
    pub(super) dynamic_offsets: Vec<wgt::DynamicOffset>,
    late_buffer_bindings: Vec<LateBufferBinding>,
    /// Since `LateBufferBinding` may contain information about the bindings
    /// not used by the pipeline, we need to know when to stop validating.
    pub(super) late_bindings_effective_count: usize,
}

impl EntryPayload {
    fn reset(&mut self) {
        self.group_id = None;
        self.dynamic_offsets.clear();
        self.late_buffer_bindings.clear();
        self.late_bindings_effective_count = 0;
    }
}

#[derive(Debug)]
pub(super) struct Binder {
    pub(super) pipeline_layout_id: Option<Valid<PipelineLayoutId>>, //TODO: strongly `Stored`
    manager: compat::BoundBindGroupLayouts,
    payloads: [EntryPayload; hal::MAX_BIND_GROUPS],
}

impl Binder {
    pub(super) fn new() -> Self {
        Self {
            pipeline_layout_id: None,
            manager: compat::BoundBindGroupLayouts::new(),
            payloads: Default::default(),
        }
    }

    pub(super) fn reset(&mut self) {
        self.pipeline_layout_id = None;
        self.manager = compat::BoundBindGroupLayouts::new();
        for payload in self.payloads.iter_mut() {
            payload.reset();
        }
    }

    pub(super) fn change_pipeline_layout<'a, A: HalApi>(
        &'a mut self,
        guard: &Storage<PipelineLayout<A>, PipelineLayoutId>,
        new_id: Valid<PipelineLayoutId>,
        late_sized_buffer_groups: &[LateSizedBufferGroup],
    ) -> (usize, &'a [EntryPayload]) {
        let old_id_opt = self.pipeline_layout_id.replace(new_id);
        let new = &guard[new_id];

        let mut bind_range = self.manager.update_expectations(&new.bind_group_layout_ids);

        // Update the buffer binding sizes that are required by shaders.
        for (payload, late_group) in self.payloads.iter_mut().zip(late_sized_buffer_groups) {
            payload.late_bindings_effective_count = late_group.shader_sizes.len();
            for (late_binding, &shader_expect_size) in payload
                .late_buffer_bindings
                .iter_mut()
                .zip(late_group.shader_sizes.iter())
            {
                late_binding.shader_expect_size = shader_expect_size;
            }
            if late_group.shader_sizes.len() > payload.late_buffer_bindings.len() {
                for &shader_expect_size in
                    late_group.shader_sizes[payload.late_buffer_bindings.len()..].iter()
                {
                    payload.late_buffer_bindings.push(LateBufferBinding {
                        shader_expect_size,
                        bound_size: 0,
                    });
                }
            }
        }

        if let Some(old_id) = old_id_opt {
            let old = &guard[old_id];
            // root constants are the base compatibility property
            if old.push_constant_ranges != new.push_constant_ranges {
                bind_range.start = 0;
            }
        }

        (bind_range.start, &self.payloads[bind_range])
    }

    pub(super) fn assign_group<'a, A: HalApi>(
        &'a mut self,
        index: usize,
        bind_group_id: Valid<BindGroupId>,
        bind_group: &BindGroup<A>,
        offsets: &[wgt::DynamicOffset],
    ) -> &'a [EntryPayload] {
        log::trace!("\tBinding [{}] = group {:?}", index, bind_group_id);
        debug_assert_eq!(A::VARIANT, bind_group_id.0.backend());

        let payload = &mut self.payloads[index];
        payload.group_id = Some(Stored {
            value: bind_group_id,
            ref_count: bind_group.life_guard.add_ref(),
        });
        payload.dynamic_offsets.clear();
        payload.dynamic_offsets.extend_from_slice(offsets);

        // Fill out the actual binding sizes for buffers,
        // whose layout doesn't specify `min_binding_size`.
        for (late_binding, late_size) in payload
            .late_buffer_bindings
            .iter_mut()
            .zip(bind_group.late_buffer_binding_sizes.iter())
        {
            late_binding.bound_size = late_size.get();
        }
        if bind_group.late_buffer_binding_sizes.len() > payload.late_buffer_bindings.len() {
            for late_size in
                bind_group.late_buffer_binding_sizes[payload.late_buffer_bindings.len()..].iter()
            {
                payload.late_buffer_bindings.push(LateBufferBinding {
                    shader_expect_size: 0,
                    bound_size: late_size.get(),
                });
            }
        }

        let bind_range = self.manager.assign(index, bind_group.layout_id);
        &self.payloads[bind_range]
    }

    pub(super) fn list_active(&self) -> impl Iterator<Item = Valid<BindGroupId>> + '_ {
        let payloads = &self.payloads;
        self.manager
            .list_active()
            .map(move |index| payloads[index].group_id.as_ref().unwrap().value)
    }

    pub(super) fn invalid_mask<A: hal::Api>(
        &self,
        bind_group_layouts: &BindGroupLayouts<A>,
    ) -> BindGroupMask {
        self.manager.invalid_mask(bind_group_layouts)
    }

    /// Scan active buffer bindings corresponding to layouts without `min_binding_size` specified.
    pub(super) fn check_late_buffer_bindings(
        &self,
    ) -> Result<(), LateMinBufferBindingSizeMismatch> {
        for group_index in self.manager.list_active() {
            let payload = &self.payloads[group_index];
            for (compact_index, late_binding) in payload.late_buffer_bindings
                [..payload.late_bindings_effective_count]
                .iter()
                .enumerate()
            {
                if late_binding.bound_size < late_binding.shader_expect_size {
                    return Err(LateMinBufferBindingSizeMismatch {
                        group_index: group_index as u32,
                        compact_index,
                        shader_size: late_binding.shader_expect_size,
                        bound_size: late_binding.bound_size,
                    });
                }
            }
        }
        Ok(())
    }
}

struct PushConstantChange {
    stages: wgt::ShaderStages,
    offset: u32,
    enable: bool,
}

/// Break up possibly overlapping push constant ranges into a set of
/// non-overlapping ranges which contain all the stage flags of the
/// original ranges. This allows us to zero out (or write any value)
/// to every possible value.
pub fn compute_nonoverlapping_ranges(
    ranges: &[wgt::PushConstantRange],
) -> ArrayVec<wgt::PushConstantRange, { SHADER_STAGE_COUNT * 2 }> {
    if ranges.is_empty() {
        return ArrayVec::new();
    }
    debug_assert!(ranges.len() <= SHADER_STAGE_COUNT);

    let mut breaks: ArrayVec<PushConstantChange, { SHADER_STAGE_COUNT * 2 }> = ArrayVec::new();
    for range in ranges {
        breaks.push(PushConstantChange {
            stages: range.stages,
            offset: range.range.start,
            enable: true,
        });
        breaks.push(PushConstantChange {
            stages: range.stages,
            offset: range.range.end,
            enable: false,
        });
    }
    breaks.sort_unstable_by_key(|change| change.offset);

    let mut output_ranges = ArrayVec::new();
    let mut position = 0_u32;
    let mut stages = wgt::ShaderStages::NONE;

    for bk in breaks {
        if bk.offset - position > 0 && !stages.is_empty() {
            output_ranges.push(wgt::PushConstantRange {
                stages,
                range: position..bk.offset,
            })
        }
        position = bk.offset;
        stages.set(bk.stages, bk.enable);
    }

    output_ranges
}
