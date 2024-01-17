use std::sync::Arc;

use crate::{
    binding_model::{BindGroup, LateMinBufferBindingSizeMismatch, PipelineLayout},
    device::SHADER_STAGE_COUNT,
    hal_api::HalApi,
    id::BindGroupId,
    pipeline::LateSizedBufferGroup,
    resource::Resource,
};

use arrayvec::ArrayVec;

type BindGroupMask = u8;

mod compat {
    use arrayvec::ArrayVec;

    use crate::{binding_model::BindGroupLayout, device::bgl, hal_api::HalApi, resource::Resource};
    use std::{ops::Range, sync::Arc};

    #[derive(Debug, Clone)]
    struct Entry<A: HalApi> {
        assigned: Option<Arc<BindGroupLayout<A>>>,
        expected: Option<Arc<BindGroupLayout<A>>>,
    }

    impl<A: HalApi> Entry<A> {
        fn empty() -> Self {
            Self {
                assigned: None,
                expected: None,
            }
        }
        fn is_active(&self) -> bool {
            self.assigned.is_some() && self.expected.is_some()
        }

        fn is_valid(&self) -> bool {
            if self.expected.is_none() {
                return true;
            }
            if let Some(expected_bgl) = self.expected.as_ref() {
                if let Some(assigned_bgl) = self.assigned.as_ref() {
                    if expected_bgl.is_equal(assigned_bgl) {
                        return true;
                    }
                }
            }
            false
        }

        fn is_incompatible(&self) -> bool {
            self.expected.is_none() || !self.is_valid()
        }

        // Describe how bind group layouts are incompatible, for validation
        // error message.
        fn bgl_diff(&self) -> Vec<String> {
            let mut diff = Vec::new();

            if let Some(expected_bgl) = self.expected.as_ref() {
                let expected_bgl_type = match expected_bgl.origin {
                    bgl::Origin::Derived => "implicit",
                    bgl::Origin::Pool => "explicit",
                };
                let expected_label = expected_bgl.label();
                diff.push(format!(
                    "Should be compatible an with an {expected_bgl_type} bind group layout {}",
                    if expected_label.is_empty() {
                        "without label".to_string()
                    } else {
                        format!("with label = `{}`", expected_label)
                    }
                ));
                if let Some(assigned_bgl) = self.assigned.as_ref() {
                    let assigned_bgl_type = match assigned_bgl.origin {
                        bgl::Origin::Derived => "implicit",
                        bgl::Origin::Pool => "explicit",
                    };
                    let assigned_label = assigned_bgl.label();
                    diff.push(format!(
                        "Assigned {assigned_bgl_type} bind group layout {}",
                        if assigned_label.is_empty() {
                            "without label".to_string()
                        } else {
                            format!("with label = `{}`", assigned_label)
                        }
                    ));
                    for (id, e_entry) in expected_bgl.entries.iter() {
                        if let Some(a_entry) = assigned_bgl.entries.get(*id) {
                            if a_entry.binding != e_entry.binding {
                                diff.push(format!(
                                    "Entry {id} binding expected {}, got {}",
                                    e_entry.binding, a_entry.binding
                                ));
                            }
                            if a_entry.count != e_entry.count {
                                diff.push(format!(
                                    "Entry {id} count expected {:?}, got {:?}",
                                    e_entry.count, a_entry.count
                                ));
                            }
                            if a_entry.ty != e_entry.ty {
                                diff.push(format!(
                                    "Entry {id} type expected {:?}, got {:?}",
                                    e_entry.ty, a_entry.ty
                                ));
                            }
                            if a_entry.visibility != e_entry.visibility {
                                diff.push(format!(
                                    "Entry {id} visibility expected {:?}, got {:?}",
                                    e_entry.visibility, a_entry.visibility
                                ));
                            }
                        } else {
                            diff.push(format!(
                                "Entry {id} not found in assigned bind group layout"
                            ))
                        }
                    }

                    assigned_bgl.entries.iter().for_each(|(id, _e_entry)| {
                        if !expected_bgl.entries.contains_key(*id) {
                            diff.push(format!(
                                "Entry {id} not found in expected bind group layout"
                            ))
                        }
                    });

                    if expected_bgl.origin != assigned_bgl.origin {
                        diff.push(format!("Expected {expected_bgl_type} bind group layout, got {assigned_bgl_type}"))
                    }
                } else {
                    diff.push("Assigned bind group layout not found (internal error)".to_owned());
                }
            } else {
                diff.push("Expected bind group layout not found (internal error)".to_owned());
            }

            diff
        }
    }

    #[derive(Debug, Default)]
    pub(crate) struct BoundBindGroupLayouts<A: HalApi> {
        entries: ArrayVec<Entry<A>, { hal::MAX_BIND_GROUPS }>,
    }

    impl<A: HalApi> BoundBindGroupLayouts<A> {
        pub fn new() -> Self {
            Self {
                entries: (0..hal::MAX_BIND_GROUPS).map(|_| Entry::empty()).collect(),
            }
        }
        fn make_range(&self, start_index: usize) -> Range<usize> {
            // find first incompatible entry
            let end = self
                .entries
                .iter()
                .position(|e| e.is_incompatible())
                .unwrap_or(self.entries.len());
            start_index..end.max(start_index)
        }

        pub fn update_expectations(
            &mut self,
            expectations: &[Arc<BindGroupLayout<A>>],
        ) -> Range<usize> {
            let start_index = self
                .entries
                .iter()
                .zip(expectations)
                .position(|(e, expect)| {
                    e.expected.is_none() || !e.expected.as_ref().unwrap().is_equal(expect)
                })
                .unwrap_or(expectations.len());
            for (e, expect) in self.entries[start_index..]
                .iter_mut()
                .zip(expectations[start_index..].iter())
            {
                e.expected = Some(expect.clone());
            }
            for e in self.entries[expectations.len()..].iter_mut() {
                e.expected = None;
            }
            self.make_range(start_index)
        }

        pub fn assign(&mut self, index: usize, value: Arc<BindGroupLayout<A>>) -> Range<usize> {
            self.entries[index].assigned = Some(value);
            self.make_range(index)
        }

        pub fn list_active(&self) -> impl Iterator<Item = usize> + '_ {
            self.entries
                .iter()
                .enumerate()
                .filter_map(|(i, e)| if e.is_active() { Some(i) } else { None })
        }

        pub fn invalid_mask(&self) -> super::BindGroupMask {
            self.entries.iter().enumerate().fold(0, |mask, (i, entry)| {
                if entry.is_valid() {
                    mask
                } else {
                    mask | 1u8 << i
                }
            })
        }

        pub fn bgl_diff(&self) -> Vec<String> {
            for e in &self.entries {
                if !e.is_valid() {
                    return e.bgl_diff();
                }
            }
            vec![String::from("No differences detected? (internal error)")]
        }
    }
}

#[derive(Debug)]
struct LateBufferBinding {
    shader_expect_size: wgt::BufferAddress,
    bound_size: wgt::BufferAddress,
}

#[derive(Debug)]
pub(super) struct EntryPayload<A: HalApi> {
    pub(super) group: Option<Arc<BindGroup<A>>>,
    pub(super) dynamic_offsets: Vec<wgt::DynamicOffset>,
    late_buffer_bindings: Vec<LateBufferBinding>,
    /// Since `LateBufferBinding` may contain information about the bindings
    /// not used by the pipeline, we need to know when to stop validating.
    pub(super) late_bindings_effective_count: usize,
}

impl<A: HalApi> Default for EntryPayload<A> {
    fn default() -> Self {
        Self {
            group: None,
            dynamic_offsets: Default::default(),
            late_buffer_bindings: Default::default(),
            late_bindings_effective_count: Default::default(),
        }
    }
}

impl<A: HalApi> EntryPayload<A> {
    fn reset(&mut self) {
        self.group = None;
        self.dynamic_offsets.clear();
        self.late_buffer_bindings.clear();
        self.late_bindings_effective_count = 0;
    }
}

#[derive(Debug, Default)]
pub(super) struct Binder<A: HalApi> {
    pub(super) pipeline_layout: Option<Arc<PipelineLayout<A>>>,
    manager: compat::BoundBindGroupLayouts<A>,
    payloads: [EntryPayload<A>; hal::MAX_BIND_GROUPS],
}

impl<A: HalApi> Binder<A> {
    pub(super) fn new() -> Self {
        Self {
            pipeline_layout: None,
            manager: compat::BoundBindGroupLayouts::new(),
            payloads: Default::default(),
        }
    }
    pub(super) fn reset(&mut self) {
        self.pipeline_layout = None;
        self.manager = compat::BoundBindGroupLayouts::new();
        for payload in self.payloads.iter_mut() {
            payload.reset();
        }
    }

    pub(super) fn change_pipeline_layout<'a>(
        &'a mut self,
        new: &Arc<PipelineLayout<A>>,
        late_sized_buffer_groups: &[LateSizedBufferGroup],
    ) -> (usize, &'a [EntryPayload<A>]) {
        let old_id_opt = self.pipeline_layout.replace(new.clone());

        let mut bind_range = self.manager.update_expectations(&new.bind_group_layouts);

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

        if let Some(old) = old_id_opt {
            // root constants are the base compatibility property
            if old.push_constant_ranges != new.push_constant_ranges {
                bind_range.start = 0;
            }
        }

        (bind_range.start, &self.payloads[bind_range])
    }

    pub(super) fn assign_group<'a>(
        &'a mut self,
        index: usize,
        bind_group: &Arc<BindGroup<A>>,
        offsets: &[wgt::DynamicOffset],
    ) -> &'a [EntryPayload<A>] {
        let bind_group_id = bind_group.as_info().id();
        log::trace!("\tBinding [{}] = group {:?}", index, bind_group_id);
        debug_assert_eq!(A::VARIANT, bind_group_id.backend());

        let payload = &mut self.payloads[index];
        payload.group = Some(bind_group.clone());
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

        let bind_range = self.manager.assign(index, bind_group.layout.clone());
        &self.payloads[bind_range]
    }

    pub(super) fn list_active(&self) -> impl Iterator<Item = BindGroupId> + '_ {
        let payloads = &self.payloads;
        self.manager
            .list_active()
            .map(move |index| payloads[index].group.as_ref().unwrap().as_info().id())
    }

    pub(super) fn invalid_mask(&self) -> BindGroupMask {
        self.manager.invalid_mask()
    }

    pub(super) fn bgl_diff(&self) -> Vec<String> {
        self.manager.bgl_diff()
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
