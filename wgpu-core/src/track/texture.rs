/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::{range::RangedStates, PendingTransition, ResourceState, Unit};
use crate::{device::MAX_MIP_LEVELS, id::TextureId};

use arrayvec::ArrayVec;
use wgt::TextureUsage;

use std::{iter, ops::Range};

//TODO: store `hal::image::State` here to avoid extra conversions
type PlaneStates = RangedStates<hal::image::Layer, Unit<TextureUsage>>;

#[derive(Clone, Debug, Default)]
pub struct TextureState {
    mips: ArrayVec<[PlaneStates; MAX_MIP_LEVELS]>,
    /// True if we have the information about all the subresources here
    full: bool,
}

impl PendingTransition<TextureState> {
    fn collapse(self) -> Result<TextureUsage, Self> {
        if self.usage.start.is_empty()
            || self.usage.start == self.usage.end
            || !TextureUsage::WRITE_ALL.intersects(self.usage.start | self.usage.end)
        {
            Ok(self.usage.start | self.usage.end)
        } else {
            Err(self)
        }
    }
}

impl TextureState {
    pub fn with_range(range: &hal::image::SubresourceRange) -> Self {
        debug_assert_eq!(range.layers.start, 0);
        debug_assert_eq!(range.levels.start, 0);
        TextureState {
            mips: iter::repeat_with(|| {
                PlaneStates::from_range(0..range.layers.end, Unit::new(TextureUsage::UNINITIALIZED))
            })
            .take(range.levels.end as usize)
            .collect(),
            full: true,
        }
    }
}

impl ResourceState for TextureState {
    type Id = TextureId;
    type Selector = hal::image::SubresourceRange;
    type Usage = TextureUsage;

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
        id: Self::Id,
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
            let level = selector.levels.start + mip_id as hal::image::Level;
            let layers = mip.isolate(&selector.layers, Unit::new(usage));
            for &mut (ref range, ref mut unit) in layers {
                if unit.last == usage && TextureUsage::ORDERED.contains(usage) {
                    continue;
                }
                // TODO: Can't satisfy clippy here unless we modify
                // `hal::image::SubresourceRange` in gfx to use
                // `std::ops::RangeBounds`.
                #[allow(clippy::range_plus_one)]
                let pending = PendingTransition {
                    id,
                    selector: hal::image::SubresourceRange {
                        aspects: hal::format::Aspects::empty(),
                        levels: level..level + 1,
                        layers: range.clone(),
                    },
                    usage: unit.last..usage,
                };

                unit.last = match output {
                    None => pending.collapse()?,
                    Some(ref mut out) => {
                        out.push(pending);
                        if unit.first.is_none() {
                            unit.first = Some(unit.last);
                        }
                        usage
                    }
                };
            }
        }
        Ok(())
    }

    fn merge(
        &mut self,
        id: Self::Id,
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
            let level = mip_id as hal::image::Level;
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
                        if start.last == to_usage && TextureUsage::ORDERED.contains(to_usage) {
                            Unit {
                                first: start.first,
                                last: end.last,
                            }
                        } else {
                            // TODO: Can't satisfy clippy here unless we modify
                            // `hal::image::SubresourceRange` in gfx to use
                            // `std::ops::RangeBounds`.
                            #[allow(clippy::range_plus_one)]
                            let pending = PendingTransition {
                                id,
                                selector: hal::image::SubresourceRange {
                                    aspects: hal::format::Aspects::empty(),
                                    levels: level..level + 1,
                                    layers: layers.clone(),
                                },
                                usage: start.last..to_usage,
                            };

                            match output {
                                None => Unit {
                                    first: start.first,
                                    last: pending.collapse()?,
                                },
                                Some(ref mut out) => {
                                    out.push(pending);
                                    Unit {
                                        first: Some(start.last),
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
    //TODO: change() and merge() tests
    //use crate::TypedId;
    use super::*;
    use hal::{format::Aspects, image::SubresourceRange};

    #[test]
    fn query() {
        let mut ts = TextureState::default();
        ts.mips.push(PlaneStates::empty());
        ts.mips.push(PlaneStates::from_slice(&[
            (1..3, Unit::new(TextureUsage::SAMPLED)),
            (3..5, Unit::new(TextureUsage::SAMPLED)),
            (5..6, Unit::new(TextureUsage::STORAGE)),
        ]));

        assert_eq!(
            ts.query(SubresourceRange {
                aspects: Aspects::COLOR,
                levels: 1..2,
                layers: 2..5,
            }),
            // level 1 matches
            Some(TextureUsage::SAMPLED),
        );
        assert_eq!(
            ts.query(SubresourceRange {
                aspects: Aspects::COLOR,
                levels: 0..2,
                layers: 2..5,
            }),
            // level 0 is empty, level 1 matches
            Some(TextureUsage::SAMPLED),
        );
        assert_eq!(
            ts.query(SubresourceRange {
                aspects: Aspects::COLOR,
                levels: 1..2,
                layers: 1..5,
            }),
            // level 1 matches with gaps
            Some(TextureUsage::SAMPLED),
        );
        assert_eq!(
            ts.query(SubresourceRange {
                aspects: Aspects::COLOR,
                levels: 1..2,
                layers: 4..6,
            }),
            // level 1 doesn't match
            None,
        );
    }
}
