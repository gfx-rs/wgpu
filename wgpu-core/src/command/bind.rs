/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    binding_model::BindGroup,
    hub::GfxBackend,
    id::{BindGroupId, BindGroupLayoutId, PipelineLayoutId},
    Stored,
};

use smallvec::{smallvec, SmallVec};
use std::slice;
use wgt::DynamicOffset;

pub const DEFAULT_BIND_GROUPS: usize = 4;
type BindGroupMask = u8;

#[derive(Clone, Debug)]
pub struct BindGroupPair {
    layout_id: BindGroupLayoutId,
    group_id: Stored<BindGroupId>,
}

#[derive(Debug)]
pub enum LayoutChange<'a> {
    Unchanged,
    Match(BindGroupId, &'a [DynamicOffset]),
    Mismatch,
}

#[derive(Debug)]
pub enum Provision {
    Unchanged,
    Changed { was_compatible: bool },
}

#[derive(Clone)]
pub struct FollowUpIter<'a> {
    iter: slice::Iter<'a, BindGroupEntry>,
}
impl<'a> Iterator for FollowUpIter<'a> {
    type Item = (BindGroupId, &'a [DynamicOffset]);
    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .and_then(|entry| Some((entry.actual_value()?, entry.dynamic_offsets.as_slice())))
    }
}

#[derive(Clone, Default, Debug)]
pub struct BindGroupEntry {
    expected_layout_id: Option<BindGroupLayoutId>,
    provided: Option<BindGroupPair>,
    dynamic_offsets: Vec<DynamicOffset>,
}

impl BindGroupEntry {
    fn provide<B: GfxBackend>(
        &mut self,
        bind_group_id: BindGroupId,
        bind_group: &BindGroup<B>,
        offsets: &[DynamicOffset],
    ) -> Provision {
        debug_assert_eq!(B::VARIANT, bind_group_id.backend());

        let was_compatible = match self.provided {
            Some(BindGroupPair {
                layout_id,
                ref group_id,
            }) => {
                if group_id.value == bind_group_id && offsets == self.dynamic_offsets.as_slice() {
                    assert_eq!(layout_id, bind_group.layout_id);
                    return Provision::Unchanged;
                }
                self.expected_layout_id == Some(layout_id)
            }
            None => false,
        };

        self.provided = Some(BindGroupPair {
            layout_id: bind_group.layout_id,
            group_id: Stored {
                value: bind_group_id,
                ref_count: bind_group.life_guard.add_ref(),
            },
        });
        //TODO: validate the count of dynamic offsets to match the layout
        self.dynamic_offsets.clear();
        self.dynamic_offsets.extend_from_slice(offsets);

        Provision::Changed { was_compatible }
    }

    pub fn expect_layout(&mut self, bind_group_layout_id: BindGroupLayoutId) -> LayoutChange {
        let some = Some(bind_group_layout_id);
        if self.expected_layout_id != some {
            self.expected_layout_id = some;
            match self.provided {
                Some(BindGroupPair {
                    layout_id,
                    ref group_id,
                }) if layout_id == bind_group_layout_id => {
                    LayoutChange::Match(group_id.value, &self.dynamic_offsets)
                }
                Some(_) | None => LayoutChange::Mismatch,
            }
        } else {
            LayoutChange::Unchanged
        }
    }

    fn is_valid(&self) -> bool {
        match (self.expected_layout_id, self.provided.as_ref()) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(layout), Some(pair)) => layout == pair.layout_id,
        }
    }

    fn actual_value(&self) -> Option<BindGroupId> {
        self.expected_layout_id.and_then(|layout_id| {
            self.provided.as_ref().and_then(|pair| {
                if pair.layout_id == layout_id {
                    Some(pair.group_id.value)
                } else {
                    None
                }
            })
        })
    }
}

#[derive(Debug)]
pub struct Binder {
    pub(crate) pipeline_layout_id: Option<PipelineLayoutId>, //TODO: strongly `Stored`
    pub(crate) entries: SmallVec<[BindGroupEntry; DEFAULT_BIND_GROUPS]>,
}

impl Binder {
    pub(crate) fn new(max_bind_groups: u32) -> Self {
        Self {
            pipeline_layout_id: None,
            entries: smallvec![Default::default(); max_bind_groups as usize],
        }
    }

    pub(crate) fn reset_expectations(&mut self, length: usize) {
        for entry in self.entries[length..].iter_mut() {
            entry.expected_layout_id = None;
        }
    }

    /// Attempt to set the value of the specified bind group index.
    /// Returns Some() when the new bind group is ready to be actually bound
    /// (i.e. compatible with current expectations). Also returns an iterator
    /// of bind group IDs to be bound with it: those are compatible bind groups
    /// that were previously blocked because the current one was incompatible.
    pub(crate) fn provide_entry<'a, B: GfxBackend>(
        &'a mut self,
        index: usize,
        bind_group_id: BindGroupId,
        bind_group: &BindGroup<B>,
        offsets: &[DynamicOffset],
    ) -> Option<(PipelineLayoutId, FollowUpIter<'a>)> {
        log::trace!("\tBinding [{}] = group {:?}", index, bind_group_id);
        debug_assert_eq!(B::VARIANT, bind_group_id.backend());

        match self.entries[index].provide(bind_group_id, bind_group, offsets) {
            Provision::Unchanged => None,
            Provision::Changed { was_compatible, .. } => {
                let compatible_count = self.compatible_count();
                if index < compatible_count {
                    let end = compatible_count.min(if was_compatible {
                        index + 1
                    } else {
                        self.entries.len()
                    });
                    log::trace!("\t\tbinding up to {}", end);
                    Some((
                        self.pipeline_layout_id?,
                        FollowUpIter {
                            iter: self.entries[index + 1..end].iter(),
                        },
                    ))
                } else {
                    log::trace!("\t\tskipping above compatible {}", compatible_count);
                    None
                }
            }
        }
    }

    pub(crate) fn invalid_mask(&self) -> BindGroupMask {
        self.entries.iter().enumerate().fold(0, |mask, (i, entry)| {
            if entry.is_valid() {
                mask
            } else {
                mask | 1u8 << i
            }
        })
    }

    fn compatible_count(&self) -> usize {
        self.entries
            .iter()
            .position(|entry| !entry.is_valid())
            .unwrap_or_else(|| self.entries.len())
    }
}
