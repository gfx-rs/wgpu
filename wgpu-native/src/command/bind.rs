use crate::{BindGroupHandle, BindGroupId, BindGroupLayoutId, PipelineLayoutId, Stored};

use log::trace;

use std::convert::identity;


pub const MAX_BIND_GROUPS: usize = 4;
type BindGroupMask = u8;


pub struct BindGroupPair {
    layout_id: BindGroupLayoutId,
    group_id: Stored<BindGroupId>,
}

pub enum Expectation {
    Unchanged,
    Match(BindGroupId),
    Mismatch,
}

pub enum Provision {
    Unchanged,
    Changed {
        was_compatible: bool,
        now_compatible: bool,
    },
}


struct TakeSome<I> {
    iter: I,
}
impl<T, I> Iterator for TakeSome<I>
where
    I: Iterator<Item = Option<T>>,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.iter.next().and_then(identity)
    }
}

#[derive(Default)]
pub struct BindGroupEntry {
    expected_layout_id: Option<BindGroupLayoutId>,
    provided: Option<BindGroupPair>,
}

impl BindGroupEntry {
    fn provide(&mut self, bind_group_id: BindGroupId, bind_group: &BindGroupHandle) -> Provision {
        let was_compatible = match self.provided {
            Some(BindGroupPair { layout_id, ref group_id }) => {
                if group_id.value == bind_group_id {
                    assert_eq!(layout_id, bind_group.layout_id);
                    return Provision::Unchanged;
                }
                self.expected_layout_id == Some(layout_id)
            }
            None => true
        };

        self.provided = Some(BindGroupPair {
            layout_id: bind_group.layout_id,
            group_id: Stored {
                value: bind_group_id,
                ref_count: bind_group.life_guard.ref_count.clone(),
            },
        });

        Provision::Changed {
            was_compatible,
            now_compatible: self.expected_layout_id == Some(bind_group.layout_id),
        }
    }

    pub fn expect_layout(
        &mut self,
        bind_group_layout_id: BindGroupLayoutId,
    ) -> Expectation {
        let some = Some(bind_group_layout_id);
        if self.expected_layout_id != some {
            self.expected_layout_id = some;
            match self.provided {
                Some(BindGroupPair {
                    layout_id,
                    ref group_id,
                }) if layout_id == bind_group_layout_id => Expectation::Match(group_id.value),
                Some(_) | None => Expectation::Mismatch,
            }
        } else {
            Expectation::Unchanged
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
        self.expected_layout_id
            .and_then(|layout_id| self.provided.as_ref().and_then(|pair| {
                if pair.layout_id == layout_id {
                    Some(pair.group_id.value)
                } else {
                    None
                }
            }))
    }
}

#[derive(Default)]
pub struct Binder {
    pub(crate) pipeline_layout_id: Option<PipelineLayoutId>, //TODO: strongly `Stored`
    pub(crate) entries: [BindGroupEntry; MAX_BIND_GROUPS],
}

impl Binder {
    pub(crate) fn cut_expectations(&mut self, length: usize) {
        for entry in self.entries[length ..].iter_mut() {
            entry.expected_layout_id = None;
        }
    }

    /// Attemt to set the value of the specified bind group index.
    /// Returns Some() when the new bind group is ready to be actually bound
    /// (i.e. compatible with current expectations). Also returns an iterator
    /// of bind group IDs to be bound with it: those are compatible bind groups
    /// that were previously blocked because the current one was incompatible.
    pub(crate) fn provide_entry<'a>(
        &'a mut self,
        index: usize,
        bind_group_id: BindGroupId,
        bind_group: &BindGroupHandle,
    ) -> Option<(PipelineLayoutId, impl 'a + Iterator<Item = BindGroupId>)> {
        trace!("\tBinding [{}] = group {:?}", index, bind_group_id);
        match self.entries[index].provide(bind_group_id, bind_group) {
            Provision::Unchanged => {
                None
            }
            Provision::Changed { now_compatible: false, .. } => {
                trace!("\t\tnot compatible");
                None
            }
            Provision::Changed { was_compatible, .. } => {
                if self.entries[.. index].iter().all(|entry| entry.is_valid()) {
                    self.pipeline_layout_id.map(move |pipeline_layout_id| {
                        let end = if was_compatible {
                            trace!("\t\tgenerating follow-up sequence");
                            MAX_BIND_GROUPS
                        } else {
                            index + 1
                        };
                        (pipeline_layout_id, TakeSome {
                            iter: self.entries[index + 1 .. end]
                                .iter()
                                .map(|entry| entry.actual_value()),
                        })
                    })
                } else {
                    trace!("\t\tbehind an incompatible");
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
}
