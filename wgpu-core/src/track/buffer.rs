/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::{PendingTransition, ResourceState, Unit};
use crate::{id::BufferId, resource::BufferUse};

//TODO: store `hal::buffer::State` here to avoid extra conversions
pub(crate) type BufferState = Unit<BufferUse>;

impl PendingTransition<BufferState> {
    fn collapse(self) -> Result<BufferUse, Self> {
        if self.usage.start.is_empty()
            || self.usage.start == self.usage.end
            || !BufferUse::WRITE_ALL.intersects(self.usage.start | self.usage.end)
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
            last: BufferUse::empty(),
        }
    }
}

impl BufferState {
    pub fn with_usage(usage: BufferUse) -> Self {
        Unit::new(usage)
    }
}

impl ResourceState for BufferState {
    type Id = BufferId;
    type Selector = ();
    type Usage = BufferUse;

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
        if old != usage || !BufferUse::ORDERED.contains(usage) {
            let pending = PendingTransition {
                id,
                selector: (),
                usage: old..usage,
            };
            *self = match output {
                None => {
                    assert_eq!(
                        self.first, None,
                        "extending a state that is already a transition"
                    );
                    Unit::new(pending.collapse()?)
                }
                Some(transitions) => {
                    transitions.push(pending);
                    Unit {
                        first: self.first.or(Some(old)),
                        last: usage,
                    }
                }
            };
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
        if old == new && BufferUse::ORDERED.contains(new) {
            if output.is_some() && self.first.is_none() {
                self.first = Some(old);
            }
        } else {
            let pending = PendingTransition {
                id,
                selector: (),
                usage: old..new,
            };
            *self = match output {
                None => {
                    assert_eq!(
                        self.first, None,
                        "extending a state that is already a transition"
                    );
                    Unit::new(pending.collapse()?)
                }
                Some(transitions) => {
                    transitions.push(pending);
                    Unit {
                        first: self.first.or(Some(old)),
                        last: other.last,
                    }
                }
            };
        }
        Ok(())
    }

    fn optimize(&mut self) {}
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::id::Id;

    #[test]
    fn change_extend() {
        let mut bs = Unit {
            first: None,
            last: BufferUse::INDEX,
        };
        let id = Id::default();
        assert_eq!(
            bs.change(id, (), BufferUse::STORAGE_STORE, None),
            Err(PendingTransition {
                id,
                selector: (),
                usage: BufferUse::INDEX..BufferUse::STORAGE_STORE,
            }),
        );
        bs.change(id, (), BufferUse::VERTEX, None).unwrap();
        bs.change(id, (), BufferUse::INDEX, None).unwrap();
        assert_eq!(bs, Unit::new(BufferUse::VERTEX | BufferUse::INDEX));
    }

    #[test]
    fn change_replace() {
        let mut bs = Unit {
            first: None,
            last: BufferUse::STORAGE_STORE,
        };
        let id = Id::default();
        let mut list = Vec::new();
        bs.change(id, (), BufferUse::VERTEX, Some(&mut list))
            .unwrap();
        assert_eq!(
            &list,
            &[PendingTransition {
                id,
                selector: (),
                usage: BufferUse::STORAGE_STORE..BufferUse::VERTEX,
            }],
        );
        assert_eq!(
            bs,
            Unit {
                first: Some(BufferUse::STORAGE_STORE),
                last: BufferUse::VERTEX,
            }
        );

        list.clear();
        bs.change(id, (), BufferUse::STORAGE_STORE, Some(&mut list))
            .unwrap();
        assert_eq!(
            &list,
            &[PendingTransition {
                id,
                selector: (),
                usage: BufferUse::VERTEX..BufferUse::STORAGE_STORE,
            }],
        );
        assert_eq!(
            bs,
            Unit {
                first: Some(BufferUse::STORAGE_STORE),
                last: BufferUse::STORAGE_STORE,
            }
        );
    }
}
