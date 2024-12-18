use std::sync::Arc;

use crate::{
    binding_model::{BindGroup, LateMinBufferBindingSizeMismatch, PipelineLayout},
    device::SHADER_STAGE_COUNT,
    pipeline::LateSizedBufferGroup,
    resource::{Labeled, ResourceErrorIdent},
};

use arrayvec::ArrayVec;
use thiserror::Error;

mod compat {
    use arrayvec::ArrayVec;
    use thiserror::Error;
    use wgt::{BindingType, ShaderStages};

    use crate::{
        binding_model::BindGroupLayout,
        error::MultiError,
        resource::{Labeled, ParentDevice, ResourceErrorIdent},
    };
    use std::{
        num::NonZeroU32,
        ops::Range,
        sync::{Arc, Weak},
    };

    pub(crate) enum Error {
        Incompatible {
            expected_bgl: ResourceErrorIdent,
            assigned_bgl: ResourceErrorIdent,
            inner: MultiError,
        },
        Missing,
    }

    #[derive(Debug, Clone)]
    struct Entry {
        assigned: Option<Arc<BindGroupLayout>>,
        expected: Option<Arc<BindGroupLayout>>,
    }

    impl Entry {
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
            if let Some(expected_bgl) = self.expected.as_ref() {
                if let Some(assigned_bgl) = self.assigned.as_ref() {
                    expected_bgl.is_equal(assigned_bgl)
                } else {
                    false
                }
            } else {
                true
            }
        }

        fn is_incompatible(&self) -> bool {
            self.expected.is_none() || !self.is_valid()
        }

        fn check(&self) -> Result<(), Error> {
            if let Some(expected_bgl) = self.expected.as_ref() {
                if let Some(assigned_bgl) = self.assigned.as_ref() {
                    if expected_bgl.is_equal(assigned_bgl) {
                        Ok(())
                    } else {
                        #[derive(Clone, Debug, Error)]
                        #[error(
                            "Exclusive pipelines don't match: expected {expected}, got {assigned}"
                        )]
                        struct IncompatibleExclusivePipelines {
                            expected: String,
                            assigned: String,
                        }

                        use crate::binding_model::ExclusivePipeline;
                        match (
                            expected_bgl.exclusive_pipeline.get().unwrap(),
                            assigned_bgl.exclusive_pipeline.get().unwrap(),
                        ) {
                            (ExclusivePipeline::None, ExclusivePipeline::None) => {}
                            (
                                ExclusivePipeline::Render(e_pipeline),
                                ExclusivePipeline::Render(a_pipeline),
                            ) if Weak::ptr_eq(e_pipeline, a_pipeline) => {}
                            (
                                ExclusivePipeline::Compute(e_pipeline),
                                ExclusivePipeline::Compute(a_pipeline),
                            ) if Weak::ptr_eq(e_pipeline, a_pipeline) => {}
                            (expected, assigned) => {
                                return Err(Error::Incompatible {
                                    expected_bgl: expected_bgl.error_ident(),
                                    assigned_bgl: assigned_bgl.error_ident(),
                                    inner: MultiError::new(core::iter::once(
                                        IncompatibleExclusivePipelines {
                                            expected: expected.to_string(),
                                            assigned: assigned.to_string(),
                                        },
                                    ))
                                    .unwrap(),
                                });
                            }
                        }

                        #[derive(Clone, Debug, Error)]
                        enum EntryError {
                            #[error("Entries with binding {binding} differ in visibility: expected {expected:?}, got {assigned:?}")]
                            Visibility {
                                binding: u32,
                                expected: ShaderStages,
                                assigned: ShaderStages,
                            },
                            #[error("Entries with binding {binding} differ in type: expected {expected:?}, got {assigned:?}")]
                            Type {
                                binding: u32,
                                expected: BindingType,
                                assigned: BindingType,
                            },
                            #[error("Entries with binding {binding} differ in count: expected {expected:?}, got {assigned:?}")]
                            Count {
                                binding: u32,
                                expected: Option<NonZeroU32>,
                                assigned: Option<NonZeroU32>,
                            },
                            #[error("Expected entry with binding {binding} not found in assigned bind group layout")]
                            ExtraExpected { binding: u32 },
                            #[error("Assigned entry with binding {binding} not found in expected bind group layout")]
                            ExtraAssigned { binding: u32 },
                        }

                        let mut errors = Vec::new();

                        for (&binding, expected_entry) in expected_bgl.entries.iter() {
                            if let Some(assigned_entry) = assigned_bgl.entries.get(binding) {
                                if assigned_entry.visibility != expected_entry.visibility {
                                    errors.push(EntryError::Visibility {
                                        binding,
                                        expected: expected_entry.visibility,
                                        assigned: assigned_entry.visibility,
                                    });
                                }
                                if assigned_entry.ty != expected_entry.ty {
                                    errors.push(EntryError::Type {
                                        binding,
                                        expected: expected_entry.ty,
                                        assigned: assigned_entry.ty,
                                    });
                                }
                                if assigned_entry.count != expected_entry.count {
                                    errors.push(EntryError::Count {
                                        binding,
                                        expected: expected_entry.count,
                                        assigned: assigned_entry.count,
                                    });
                                }
                            } else {
                                errors.push(EntryError::ExtraExpected { binding });
                            }
                        }

                        for (&binding, _) in assigned_bgl.entries.iter() {
                            if !expected_bgl.entries.contains_key(binding) {
                                errors.push(EntryError::ExtraAssigned { binding });
                            }
                        }

                        Err(Error::Incompatible {
                            expected_bgl: expected_bgl.error_ident(),
                            assigned_bgl: assigned_bgl.error_ident(),
                            inner: MultiError::new(errors.drain(..)).unwrap(),
                        })
                    }
                } else {
                    Err(Error::Missing)
                }
            } else {
                Ok(())
            }
        }
    }

    #[derive(Debug, Default)]
    pub(crate) struct BoundBindGroupLayouts {
        entries: ArrayVec<Entry, { hal::MAX_BIND_GROUPS }>,
    }

    impl BoundBindGroupLayouts {
        pub fn new() -> Self {
            Self {
                entries: (0..hal::MAX_BIND_GROUPS).map(|_| Entry::empty()).collect(),
            }
        }

        pub fn num_valid_entries(&self) -> usize {
            // find first incompatible entry
            self.entries
                .iter()
                .position(|e| e.is_incompatible())
                .unwrap_or(self.entries.len())
        }

        fn make_range(&self, start_index: usize) -> Range<usize> {
            let end = self.num_valid_entries();
            start_index..end.max(start_index)
        }

        pub fn update_expectations(
            &mut self,
            expectations: &[Arc<BindGroupLayout>],
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

        pub fn assign(&mut self, index: usize, value: Arc<BindGroupLayout>) -> Range<usize> {
            self.entries[index].assigned = Some(value);
            self.make_range(index)
        }

        pub fn list_active(&self) -> impl Iterator<Item = usize> + '_ {
            self.entries
                .iter()
                .enumerate()
                .filter_map(|(i, e)| if e.is_active() { Some(i) } else { None })
        }

        #[allow(clippy::result_large_err)]
        pub fn get_invalid(&self) -> Result<(), (usize, Error)> {
            for (index, entry) in self.entries.iter().enumerate() {
                entry.check().map_err(|e| (index, e))?;
            }
            Ok(())
        }
    }
}

#[derive(Clone, Debug, Error)]
pub enum BinderError {
    #[error("The current set {pipeline} expects a BindGroup to be set at index {index}")]
    MissingBindGroup {
        index: usize,
        pipeline: ResourceErrorIdent,
    },
    #[error("The {assigned_bgl} of current set {assigned_bg} at index {index} is not compatible with the corresponding {expected_bgl} of {pipeline}")]
    IncompatibleBindGroup {
        expected_bgl: ResourceErrorIdent,
        assigned_bgl: ResourceErrorIdent,
        assigned_bg: ResourceErrorIdent,
        index: usize,
        pipeline: ResourceErrorIdent,
        #[source]
        inner: crate::error::MultiError,
    },
}

#[derive(Debug)]
struct LateBufferBinding {
    shader_expect_size: wgt::BufferAddress,
    bound_size: wgt::BufferAddress,
}

#[derive(Debug, Default)]
pub(super) struct EntryPayload {
    pub(super) group: Option<Arc<BindGroup>>,
    pub(super) dynamic_offsets: Vec<wgt::DynamicOffset>,
    late_buffer_bindings: Vec<LateBufferBinding>,
    /// Since `LateBufferBinding` may contain information about the bindings
    /// not used by the pipeline, we need to know when to stop validating.
    pub(super) late_bindings_effective_count: usize,
}

impl EntryPayload {
    fn reset(&mut self) {
        self.group = None;
        self.dynamic_offsets.clear();
        self.late_buffer_bindings.clear();
        self.late_bindings_effective_count = 0;
    }
}

#[derive(Debug, Default)]
pub(super) struct Binder {
    pub(super) pipeline_layout: Option<Arc<PipelineLayout>>,
    manager: compat::BoundBindGroupLayouts,
    payloads: [EntryPayload; hal::MAX_BIND_GROUPS],
}

impl Binder {
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
        new: &Arc<PipelineLayout>,
        late_sized_buffer_groups: &[LateSizedBufferGroup],
    ) -> (usize, &'a [EntryPayload]) {
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
        bind_group: &Arc<BindGroup>,
        offsets: &[wgt::DynamicOffset],
    ) -> &'a [EntryPayload] {
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

    pub(super) fn list_active<'a>(&'a self) -> impl Iterator<Item = &'a Arc<BindGroup>> + 'a {
        let payloads = &self.payloads;
        self.manager
            .list_active()
            .map(move |index| payloads[index].group.as_ref().unwrap())
    }

    #[cfg(feature = "indirect-validation")]
    pub(super) fn list_valid<'a>(&'a self) -> impl Iterator<Item = (usize, &'a EntryPayload)> + 'a {
        self.payloads
            .iter()
            .take(self.manager.num_valid_entries())
            .enumerate()
    }

    pub(super) fn check_compatibility<T: Labeled>(
        &self,
        pipeline: &T,
    ) -> Result<(), Box<BinderError>> {
        self.manager.get_invalid().map_err(|(index, error)| {
            Box::new(match error {
                compat::Error::Incompatible {
                    expected_bgl,
                    assigned_bgl,
                    inner,
                } => BinderError::IncompatibleBindGroup {
                    expected_bgl,
                    assigned_bgl,
                    assigned_bg: self.payloads[index].group.as_ref().unwrap().error_ident(),
                    index,
                    pipeline: pipeline.error_ident(),
                    inner,
                },
                compat::Error::Missing => BinderError::MissingBindGroup {
                    index,
                    pipeline: pipeline.error_ident(),
                },
            })
        })
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
