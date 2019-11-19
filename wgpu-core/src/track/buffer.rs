/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::{PendingTransition, ResourceState, Stitch, Unit};
use crate::{conv, id::BufferId, resource::BufferUsage};
use std::ops::Range;

//TODO: store `hal::buffer::State` here to avoid extra conversions
pub type BufferState = Unit<BufferUsage>;

impl PendingTransition<BufferState> {
    /// Produce the gfx-hal buffer states corresponding to the transition.
    pub fn to_states(&self) -> Range<hal::buffer::State> {
        conv::map_buffer_state(self.usage.start) .. conv::map_buffer_state(self.usage.end)
    }
}

impl ResourceState for BufferState {
    type Id = BufferId;
    type Selector = ();
    type Usage = BufferUsage;

    fn new(_full_selector: &Self::Selector) -> Self {
        BufferState {
            init: BufferUsage::empty(),
            last: BufferUsage::empty(),
        }
    }

    fn query(&self, _selector: Self::Selector) -> Option<Self::Usage> {
        Some(self.last)
    }

    fn change(
        &mut self,
        id: Self::Id,
        _selector: Self::Selector,
        usage: Self::Usage,
        output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        let old = self.last;
        if usage != old || !BufferUsage::ORDERED.contains(usage) {
            let pending = PendingTransition {
                id,
                selector: (),
                usage: old .. usage,
            };
            self.last = match output {
                Some(transitions) => {
                    transitions.push(pending);
                    usage
                }
                None => {
                    if !old.is_empty()
                        && old != usage
                        && BufferUsage::WRITE_ALL.intersects(old | usage)
                    {
                        return Err(pending);
                    }
                    old | usage
                }
            };
        }
        Ok(())
    }

    fn merge(
        &mut self,
        id: Self::Id,
        other: &Self,
        stitch: Stitch,
        output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        let old = self.last;
        let new = other.select(stitch);
        self.last = if old == new && BufferUsage::ORDERED.contains(new) {
            other.last
        } else {
            let pending = PendingTransition {
                id,
                selector: (),
                usage: old .. new,
            };
            match output {
                Some(transitions) => {
                    transitions.push(pending);
                    other.last
                }
                None => {
                    if !old.is_empty() && BufferUsage::WRITE_ALL.intersects(old | new) {
                        return Err(pending);
                    }
                    old | new
                }
            }
        };
        Ok(())
    }

    fn optimize(&mut self) {}
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{id::TypedId, Backend};

    #[test]
    fn change() {
        let mut bs = Unit {
            init: BufferUsage::INDEX,
            last: BufferUsage::STORAGE,
        };
        let id = TypedId::zip(0, 0, Backend::Empty);
        assert!(bs.change(id, (), BufferUsage::VERTEX, None).is_err());
        bs.change(id, (), BufferUsage::VERTEX, Some(&mut Vec::new()))
            .unwrap();
        bs.change(id, (), BufferUsage::INDEX, None).unwrap();
        assert_eq!(bs.last, BufferUsage::VERTEX | BufferUsage::INDEX);
    }
}
