/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    binding_model::{BindGroup, PipelineLayout},
    device::SHADER_STAGE_COUNT,
    hub::{GfxBackend, Storage},
    id::{BindGroupId, BindGroupLayoutId, PipelineLayoutId, Valid},
    Stored, MAX_BIND_GROUPS,
};

use arrayvec::ArrayVec;

type BindGroupMask = u8;

mod compat {
    use std::ops::Range;

    #[derive(Debug)]
    struct Entry<T> {
        assigned: Option<T>,
        expected: Option<T>,
    }
    impl<T> Default for Entry<T> {
        fn default() -> Self {
            Entry {
                assigned: None,
                expected: None,
            }
        }
    }
    impl<T: Copy + PartialEq> Entry<T> {
        fn is_active(&self) -> bool {
            self.assigned.is_some() && self.expected.is_some()
        }

        fn is_valid(&self) -> bool {
            self.expected.is_none() || self.expected == self.assigned
        }
    }

    #[derive(Debug)]
    pub struct Manager<T> {
        entries: [Entry<T>; crate::MAX_BIND_GROUPS],
    }

    impl<T: Copy + PartialEq> Manager<T> {
        pub fn new() -> Self {
            Manager {
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

        pub fn update_expectations(&mut self, expectations: &[T]) -> Range<usize> {
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

        pub fn assign(&mut self, index: usize, value: T) -> Range<usize> {
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
    }

    #[test]
    fn test_compatibility() {
        let mut man = Manager::<i32>::new();
        man.entries[0] = Entry {
            expected: Some(3),
            assigned: Some(2),
        };
        man.entries[1] = Entry {
            expected: Some(1),
            assigned: Some(1),
        };
        man.entries[2] = Entry {
            expected: Some(4),
            assigned: Some(5),
        };
        // check that we rebind [1] after [0] became compatible
        assert_eq!(man.assign(0, 3), 0..2);
        // check that nothing is rebound
        assert_eq!(man.update_expectations(&[3, 2]), 1..1);
        // check that [1] and [2] are rebound on expectations change
        assert_eq!(man.update_expectations(&[3, 1, 5]), 1..3);
        // reset the first two bindings
        assert_eq!(man.update_expectations(&[4, 6, 5]), 0..0);
        // check that nothing is rebound, even if there is a match,
        // since earlier binding is incompatible.
        assert_eq!(man.assign(1, 6), 1..1);
        // finally, bind everything
        assert_eq!(man.assign(0, 4), 0..3);
    }
}

#[derive(Debug, Default)]
pub(super) struct EntryPayload {
    pub(super) group_id: Option<Stored<BindGroupId>>,
    pub(super) dynamic_offsets: Vec<wgt::DynamicOffset>,
}

#[derive(Debug)]
pub(super) struct Binder {
    pub(super) pipeline_layout_id: Option<Valid<PipelineLayoutId>>, //TODO: strongly `Stored`
    manager: compat::Manager<Valid<BindGroupLayoutId>>,
    payloads: [EntryPayload; MAX_BIND_GROUPS],
}

impl Binder {
    pub(super) fn new() -> Self {
        Binder {
            pipeline_layout_id: None,
            manager: compat::Manager::new(),
            payloads: Default::default(),
        }
    }

    pub(super) fn reset(&mut self) {
        self.pipeline_layout_id = None;
        self.manager = compat::Manager::new();
        for payload in self.payloads.iter_mut() {
            payload.group_id = None;
            payload.dynamic_offsets.clear();
        }
    }

    pub(super) fn change_pipeline_layout<'a, B: GfxBackend>(
        &'a mut self,
        guard: &Storage<PipelineLayout<B>, PipelineLayoutId>,
        new_id: Valid<PipelineLayoutId>,
    ) -> (usize, &'a [EntryPayload]) {
        let old_id_opt = self.pipeline_layout_id.replace(new_id);
        let new = &guard[new_id];

        let mut bind_range = self.manager.update_expectations(&new.bind_group_layout_ids);

        if let Some(old_id) = old_id_opt {
            let old = &guard[old_id];
            // root constants are the base compatibility property
            if old.push_constant_ranges != new.push_constant_ranges {
                bind_range.start = 0;
            }
        }

        (bind_range.start, &self.payloads[bind_range])
    }

    pub(super) fn assign_group<'a, B: GfxBackend>(
        &'a mut self,
        index: usize,
        bind_group_id: Valid<BindGroupId>,
        bind_group: &BindGroup<B>,
        offsets: &[wgt::DynamicOffset],
    ) -> &'a [EntryPayload] {
        log::trace!("\tBinding [{}] = group {:?}", index, bind_group_id);
        debug_assert_eq!(B::VARIANT, bind_group_id.0.backend());

        let payload = &mut self.payloads[index];
        payload.group_id = Some(Stored {
            value: bind_group_id,
            ref_count: bind_group.life_guard.add_ref(),
        });
        payload.dynamic_offsets.clear();
        payload.dynamic_offsets.extend_from_slice(offsets);

        let bind_range = self.manager.assign(index, bind_group.layout_id);
        &self.payloads[bind_range]
    }

    pub(super) fn list_active(&self) -> impl Iterator<Item = Valid<BindGroupId>> + '_ {
        let payloads = &self.payloads;
        self.manager
            .list_active()
            .map(move |index| payloads[index].group_id.as_ref().unwrap().value)
    }

    pub(super) fn invalid_mask(&self) -> BindGroupMask {
        self.manager.invalid_mask()
    }
}

struct PushConstantChange {
    stages: wgt::ShaderStage,
    offset: u32,
    enable: bool,
}

/// Break up possibly overlapping push constant ranges into a set of non-overlapping ranges
/// which contain all the stage flags of the original ranges. This allows us to zero out (or write any value)
/// to every possible value.
pub fn compute_nonoverlapping_ranges(
    ranges: &[wgt::PushConstantRange],
) -> ArrayVec<[wgt::PushConstantRange; SHADER_STAGE_COUNT * 2]> {
    if ranges.is_empty() {
        return ArrayVec::new();
    }
    debug_assert!(ranges.len() <= SHADER_STAGE_COUNT);

    let mut breaks: ArrayVec<[PushConstantChange; SHADER_STAGE_COUNT * 2]> = ArrayVec::new();
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
    let mut stages = wgt::ShaderStage::NONE;

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
