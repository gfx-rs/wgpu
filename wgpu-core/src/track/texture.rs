use super::{range::RangedStates, PendingTransition, ResourceState, Unit};
use crate::id::{TextureId, Valid};
use hal::TextureUses;

use arrayvec::ArrayVec;

use std::{iter, ops::Range};

type PlaneStates = RangedStates<u32, Unit<TextureUses>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextureSelector {
    //TODO: rename to `mip_levels` and `array_layers` for consistency
    //pub aspects: hal::FormatAspects,
    pub levels: Range<u32>,
    pub layers: Range<u32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct TextureState {
    mips: ArrayVec<PlaneStates, { hal::MAX_MIP_LEVELS as usize }>,
    /// True if we have the information about all the subresources here
    full: bool,
}

impl PendingTransition<TextureState> {
    fn collapse(self) -> Result<TextureUses, Self> {
        if self.usage.start.is_empty()
            || self.usage.start == self.usage.end
            || !TextureUses::EXCLUSIVE.intersects(self.usage.start | self.usage.end)
        {
            Ok(self.usage.start | self.usage.end)
        } else {
            Err(self)
        }
    }
}

impl TextureState {
    pub fn new(mip_level_count: u32, array_layer_count: u32) -> Self {
        Self {
            mips: iter::repeat_with(|| {
                PlaneStates::from_range(0..array_layer_count, Unit::new(TextureUses::UNINITIALIZED))
            })
            .take(mip_level_count as usize)
            .collect(),
            full: true,
        }
    }
}

impl ResourceState for TextureState {
    type Id = TextureId;
    type Selector = TextureSelector;
    type Usage = TextureUses;

    fn query(&self, selector: Self::Selector) -> Option<Self::Usage> {
        let mut result = None;
        // Note: we only consider the subresources tracked by `self`.
        // If some are not known to `self`, it means the can assume the
        // initial state to whatever we need, which we can always make
        // to be the same as the query result for the known subresources.
        let num_levels = self.mips.len();
        if self.full {
            assert!(num_levels >= selector.levels.end as usize);
        }
        let mip_start = num_levels.min(selector.levels.start as usize);
        let mip_end = num_levels.min(selector.levels.end as usize);
        for mip in self.mips[mip_start..mip_end].iter() {
            match mip.query(&selector.layers, |unit| unit.last) {
                None => {}
                Some(Ok(usage)) if result == Some(usage) => {}
                Some(Ok(usage)) if result.is_none() => {
                    result = Some(usage);
                }
                Some(Ok(_)) | Some(Err(())) => return None,
            }
        }
        result
    }

    fn change(
        &mut self,
        id: Valid<Self::Id>,
        selector: Self::Selector,
        usage: Self::Usage,
        mut output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        if self.full {
            assert!(self.mips.len() >= selector.levels.end as usize);
        } else {
            while self.mips.len() < selector.levels.end as usize {
                self.mips.push(PlaneStates::empty());
            }
        }
        for (mip_id, mip) in self.mips[selector.levels.start as usize..selector.levels.end as usize]
            .iter_mut()
            .enumerate()
        {
            let level = selector.levels.start + mip_id as u32;
            let layers = mip.isolate(&selector.layers, Unit::new(usage));
            for &mut (ref range, ref mut unit) in layers {
                if unit.last == usage && TextureUses::ORDERED.contains(usage) {
                    continue;
                }
                // TODO: Can't satisfy clippy here unless we modify
                // `TextureSelector` to use `std::ops::RangeBounds`.
                #[allow(clippy::range_plus_one)]
                let pending = PendingTransition {
                    id,
                    selector: TextureSelector {
                        levels: level..level + 1,
                        layers: range.clone(),
                    },
                    usage: unit.last..usage,
                };

                *unit = match output {
                    None => {
                        assert_eq!(
                            unit.first, None,
                            "extending a state that is already a transition"
                        );
                        Unit::new(pending.collapse()?)
                    }
                    Some(ref mut out) => {
                        out.push(pending);
                        Unit {
                            first: unit.first.or(Some(unit.last)),
                            last: usage,
                        }
                    }
                };
            }
        }
        Ok(())
    }

    fn merge(
        &mut self,
        id: Valid<Self::Id>,
        other: &Self,
        mut output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        let mut temp = Vec::new();
        if self.full {
            assert!(self.mips.len() >= other.mips.len());
        } else {
            while self.mips.len() < other.mips.len() {
                self.mips.push(PlaneStates::empty());
            }
        }

        for (mip_id, (mip_self, mip_other)) in self.mips.iter_mut().zip(&other.mips).enumerate() {
            let level = mip_id as u32;
            temp.extend(mip_self.merge(mip_other, 0));
            mip_self.clear();

            for (layers, states) in temp.drain(..) {
                let unit = match states {
                    Range {
                        start: None,
                        end: None,
                    } => unreachable!(),
                    Range {
                        start: Some(start),
                        end: None,
                    } => start,
                    Range {
                        start: None,
                        end: Some(end),
                    } => end,
                    Range {
                        start: Some(start),
                        end: Some(end),
                    } => {
                        let to_usage = end.port();
                        if start.last == to_usage && TextureUses::ORDERED.contains(to_usage) {
                            Unit {
                                first: match output {
                                    None => start.first,
                                    Some(_) => start.first.or(Some(start.last)),
                                },
                                last: end.last,
                            }
                        } else {
                            // TODO: Can't satisfy clippy here unless we modify
                            // `TextureSelector` to use `std::ops::RangeBounds`.
                            #[allow(clippy::range_plus_one)]
                            let pending = PendingTransition {
                                id,
                                selector: TextureSelector {
                                    levels: level..level + 1,
                                    layers: layers.clone(),
                                },
                                usage: start.last..to_usage,
                            };

                            match output {
                                None => {
                                    assert_eq!(
                                        start.first, None,
                                        "extending a state that is already a transition"
                                    );
                                    Unit::new(pending.collapse()?)
                                }
                                Some(ref mut out) => {
                                    out.push(pending);
                                    Unit {
                                        // this has to leave a valid `first` state
                                        first: start.first.or(Some(start.last)),
                                        last: end.last,
                                    }
                                }
                            }
                        }
                    }
                };
                mip_self.append(layers, unit);
            }
        }

        Ok(())
    }

    fn optimize(&mut self) {
        for mip in self.mips.iter_mut() {
            mip.coalesce();
        }
    }
}

#[cfg(test)]
mod test {
    //TODO: change() tests
    use super::*;
    use crate::id::Id;

    #[test]
    fn query() {
        let mut ts = TextureState::default();
        ts.mips.push(PlaneStates::empty());
        ts.mips.push(PlaneStates::from_slice(&[
            (1..3, Unit::new(TextureUses::RESOURCE)),
            (3..5, Unit::new(TextureUses::RESOURCE)),
            (5..6, Unit::new(TextureUses::STORAGE_READ)),
        ]));

        assert_eq!(
            ts.query(TextureSelector {
                levels: 1..2,
                layers: 2..5,
            }),
            // level 1 matches
            Some(TextureUses::RESOURCE),
        );
        assert_eq!(
            ts.query(TextureSelector {
                levels: 0..2,
                layers: 2..5,
            }),
            // level 0 is empty, level 1 matches
            Some(TextureUses::RESOURCE),
        );
        assert_eq!(
            ts.query(TextureSelector {
                levels: 1..2,
                layers: 1..5,
            }),
            // level 1 matches with gaps
            Some(TextureUses::RESOURCE),
        );
        assert_eq!(
            ts.query(TextureSelector {
                levels: 1..2,
                layers: 4..6,
            }),
            // level 1 doesn't match
            None,
        );
    }

    #[test]
    fn merge() {
        let id = Id::dummy();
        let mut ts1 = TextureState::default();
        ts1.mips.push(PlaneStates::from_slice(&[(
            1..3,
            Unit::new(TextureUses::RESOURCE),
        )]));
        let mut ts2 = TextureState::default();
        assert_eq!(
            ts1.merge(id, &ts2, None),
            Ok(()),
            "failed to merge with an empty"
        );

        ts2.mips.push(PlaneStates::from_slice(&[(
            1..2,
            Unit::new(TextureUses::COPY_SRC),
        )]));
        assert_eq!(
            ts1.merge(Id::dummy(), &ts2, None),
            Ok(()),
            "failed to extend a compatible state"
        );
        assert_eq!(
            ts1.mips[0].query(&(1..2), |&v| v),
            Some(Ok(Unit {
                first: None,
                last: TextureUses::RESOURCE | TextureUses::COPY_SRC,
            })),
            "wrong extension result"
        );

        ts2.mips[0] = PlaneStates::from_slice(&[(1..2, Unit::new(TextureUses::COPY_DST))]);
        assert_eq!(
            ts1.clone().merge(Id::dummy(), &ts2, None),
            Err(PendingTransition {
                id,
                selector: TextureSelector {
                    levels: 0..1,
                    layers: 1..2,
                },
                usage: TextureUses::RESOURCE | TextureUses::COPY_SRC..TextureUses::COPY_DST,
            }),
            "wrong error on extending with incompatible state"
        );

        let mut list = Vec::new();
        ts2.mips[0] = PlaneStates::from_slice(&[
            (1..2, Unit::new(TextureUses::COPY_DST)),
            (
                2..3,
                Unit {
                    first: Some(TextureUses::COPY_SRC),
                    last: TextureUses::COLOR_TARGET,
                },
            ),
        ]);
        ts1.merge(Id::dummy(), &ts2, Some(&mut list)).unwrap();
        assert_eq!(
            &list,
            &[
                PendingTransition {
                    id,
                    selector: TextureSelector {
                        levels: 0..1,
                        layers: 1..2,
                    },
                    usage: TextureUses::RESOURCE | TextureUses::COPY_SRC..TextureUses::COPY_DST,
                },
                PendingTransition {
                    id,
                    selector: TextureSelector {
                        levels: 0..1,
                        layers: 2..3,
                    },
                    // the transition links the end of the base rage (..SAMPLED)
                    // with the start of the next range (COPY_SRC..)
                    usage: TextureUses::RESOURCE..TextureUses::COPY_SRC,
                },
            ],
            "replacing produced wrong transitions"
        );
        assert_eq!(
            ts1.mips[0].query(&(1..2), |&v| v),
            Some(Ok(Unit {
                first: Some(TextureUses::RESOURCE | TextureUses::COPY_SRC),
                last: TextureUses::COPY_DST,
            })),
            "wrong final layer 1 state"
        );
        assert_eq!(
            ts1.mips[0].query(&(2..3), |&v| v),
            Some(Ok(Unit {
                first: Some(TextureUses::RESOURCE),
                last: TextureUses::COLOR_TARGET,
            })),
            "wrong final layer 2 state"
        );

        list.clear();
        ts2.mips[0] = PlaneStates::from_slice(&[(
            2..3,
            Unit {
                first: Some(TextureUses::COLOR_TARGET),
                last: TextureUses::COPY_SRC,
            },
        )]);
        ts1.merge(Id::dummy(), &ts2, Some(&mut list)).unwrap();
        assert_eq!(&list, &[], "unexpected replacing transition");

        list.clear();
        ts2.mips[0] = PlaneStates::from_slice(&[(
            2..3,
            Unit {
                first: Some(TextureUses::COPY_DST),
                last: TextureUses::COPY_DST,
            },
        )]);
        ts1.merge(Id::dummy(), &ts2, Some(&mut list)).unwrap();
        assert_eq!(
            &list,
            &[PendingTransition {
                id,
                selector: TextureSelector {
                    levels: 0..1,
                    layers: 2..3,
                },
                usage: TextureUses::COPY_SRC..TextureUses::COPY_DST,
            },],
            "invalid replacing transition"
        );
        assert_eq!(
            ts1.mips[0].query(&(2..3), |&v| v),
            Some(Ok(Unit {
                // the initial state here is never expected to change
                first: Some(TextureUses::RESOURCE),
                last: TextureUses::COPY_DST,
            })),
            "wrong final layer 2 state"
        );
    }
}
