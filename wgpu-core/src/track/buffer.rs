use super::{PendingTransition, ResourceState, Unit};
use crate::id::{BufferId, Valid};
use hal::BufferUses;

pub(crate) type BufferState = Unit<BufferUses>;

impl PendingTransition<BufferState> {
    fn collapse(self) -> Result<BufferUses, Self> {
        if self.usage.start.is_empty()
            || self.usage.start == self.usage.end
            || !BufferUses::EXCLUSIVE.intersects(self.usage.start | self.usage.end)
        {
            Ok(self.usage.start | self.usage.end)
        } else {
            Err(self)
        }
    }
}

impl Default for BufferState {
    fn default() -> Self {
        Self {
            first: None,
            last: BufferUses::empty(),
        }
    }
}

impl BufferState {
    pub fn with_usage(usage: BufferUses) -> Self {
        Unit::new(usage)
    }
}

impl ResourceState for BufferState {
    type Id = BufferId;
    type Selector = ();
    type Usage = BufferUses;

    fn query(&self, _selector: Self::Selector) -> Option<Self::Usage> {
        Some(self.last)
    }

    fn change(
        &mut self,
        id: Valid<Self::Id>,
        _selector: Self::Selector,
        usage: Self::Usage,
        output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        let old = self.last;
        if old != usage || !BufferUses::ORDERED.contains(usage) {
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
        id: Valid<Self::Id>,
        other: &Self,
        output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        let old = self.last;
        let new = other.port();
        if old == new && BufferUses::ORDERED.contains(new) {
            if output.is_some() && self.first.is_none() {
                *self = Unit {
                    first: Some(old),
                    last: other.last,
                };
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
            last: BufferUses::INDEX,
        };
        let id = Id::dummy();
        assert_eq!(
            bs.change(id, (), BufferUses::STORAGE_WRITE, None),
            Err(PendingTransition {
                id,
                selector: (),
                usage: BufferUses::INDEX..BufferUses::STORAGE_WRITE,
            }),
        );
        bs.change(id, (), BufferUses::VERTEX, None).unwrap();
        bs.change(id, (), BufferUses::INDEX, None).unwrap();
        assert_eq!(bs, Unit::new(BufferUses::VERTEX | BufferUses::INDEX));
    }

    #[test]
    fn change_replace() {
        let mut bs = Unit {
            first: None,
            last: BufferUses::STORAGE_WRITE,
        };
        let id = Id::dummy();
        let mut list = Vec::new();
        bs.change(id, (), BufferUses::VERTEX, Some(&mut list))
            .unwrap();
        assert_eq!(
            &list,
            &[PendingTransition {
                id,
                selector: (),
                usage: BufferUses::STORAGE_WRITE..BufferUses::VERTEX,
            }],
        );
        assert_eq!(
            bs,
            Unit {
                first: Some(BufferUses::STORAGE_WRITE),
                last: BufferUses::VERTEX,
            }
        );

        list.clear();
        bs.change(id, (), BufferUses::STORAGE_WRITE, Some(&mut list))
            .unwrap();
        assert_eq!(
            &list,
            &[PendingTransition {
                id,
                selector: (),
                usage: BufferUses::VERTEX..BufferUses::STORAGE_WRITE,
            }],
        );
        assert_eq!(
            bs,
            Unit {
                first: Some(BufferUses::STORAGE_WRITE),
                last: BufferUses::STORAGE_WRITE,
            }
        );
    }

    #[test]
    fn merge_replace() {
        let mut bs = Unit {
            first: None,
            last: BufferUses::empty(),
        };
        let other_smooth = Unit {
            first: Some(BufferUses::empty()),
            last: BufferUses::COPY_DST,
        };
        let id = Id::dummy();
        let mut list = Vec::new();
        bs.merge(id, &other_smooth, Some(&mut list)).unwrap();
        assert!(list.is_empty());
        assert_eq!(
            bs,
            Unit {
                first: Some(BufferUses::empty()),
                last: BufferUses::COPY_DST,
            }
        );

        let other_rough = Unit {
            first: Some(BufferUses::empty()),
            last: BufferUses::UNIFORM,
        };
        bs.merge(id, &other_rough, Some(&mut list)).unwrap();
        assert_eq!(
            &list,
            &[PendingTransition {
                id,
                selector: (),
                usage: BufferUses::COPY_DST..BufferUses::empty(),
            }],
        );
        assert_eq!(
            bs,
            Unit {
                first: Some(BufferUses::empty()),
                last: BufferUses::UNIFORM,
            }
        );
    }
}
