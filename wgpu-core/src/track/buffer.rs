/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::{PendingTransition, ResourceState, Unit};
use crate::{id::BufferId, resource::BufferUsage};

//TODO: store `hal::buffer::State` here to avoid extra conversions
pub type BufferState = Unit<BufferUsage>;

impl PendingTransition<BufferState> {
    fn collapse(self) -> Result<BufferUsage, Self> {
        if self.usage.start.is_empty()
            || self.usage.start == self.usage.end
            || !BufferUsage::WRITE_ALL.intersects(self.usage.start | self.usage.end)
        {
            Ok(self.usage.start | self.usage.end)
        } else {
            Err(self)
        }
    }
}

impl Default for BufferState {
    fn default() -> Self {
        BufferState {
            first: None,
            last: BufferUsage::empty(),
        }
    }
}

impl BufferState {
    pub fn with_usage(usage: BufferUsage) -> Self {
        Unit::new(usage)
    }
}

impl ResourceState for BufferState {
    type Id = BufferId;
    type Selector = ();
    type Usage = BufferUsage;

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
                None => pending.collapse()?,
                Some(transitions) => {
                    transitions.push(pending);
                    if self.first.is_none() {
                        self.first = Some(old);
                    }
                    usage
                }
            }
        }
        Ok(())
    }

    fn merge(
        &mut self,
        id: Self::Id,
        other: &Self,
        output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        let old = self.last;
        let new = other.port();
        self.last = if old == new && BufferUsage::ORDERED.contains(new) {
            if self.first.is_none() {
                self.first = Some(old);
            }
            other.last
        } else {
            let pending = PendingTransition {
                id,
                selector: (),
                usage: old .. new,
            };
            match output {
                None => pending.collapse()?,
                Some(transitions) => {
                    transitions.push(pending);
                    if self.first.is_none() {
                        self.first = Some(old);
                    }
                    other.last
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
            first: Some(BufferUsage::INDEX),
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
