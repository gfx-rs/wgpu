/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::{SEPARATE_DEPTH_STENCIL_STATES, range::RangedStates, PendingTransition, ResourceState, Stitch, Unit};
use crate::{conv, device::MAX_MIP_LEVELS, id::TextureId, resource::TextureUsage};

use arrayvec::ArrayVec;

use std::ops::Range;


//TODO: store `hal::image::State` here to avoid extra conversions
type PlaneStates = RangedStates<hal::image::Layer, Unit<TextureUsage>>;
type MipState = ArrayVec<[(hal::format::Aspects, PlaneStates); 2]>;

#[derive(Clone, Debug)]
pub struct TextureState {
    mips: ArrayVec<[MipState; MAX_MIP_LEVELS]>,
}

impl PendingTransition<TextureState> {
    /// Produce the gfx-hal image states corresponding to the transition.
    pub fn to_states(&self) -> Range<hal::image::State> {
        conv::map_texture_state(self.usage.start, self.selector.aspects)
            .. conv::map_texture_state(self.usage.end, self.selector.aspects)
    }

    //TODO: make this less awkward!
    /// Check for the validity of `self` with regards to the presence of `output`.
    ///
    /// Return the end usage if the `output` is provided and pushes self to it.
    /// Otherwise, return the extended usage, or an error if extension is impossible.
    ///
    /// When a transition is generated, returns the specified `replace` usage.
    fn record(
        self,
        output: Option<&mut &mut Vec<Self>>,
        replace: TextureUsage,
    ) -> Result<TextureUsage, Self> {
        let u = self.usage.clone();
        match output {
            Some(out) => {
                out.push(self);
                Ok(replace)
            }
            None => {
                if !u.start.is_empty()
                    && u.start != u.end
                    && TextureUsage::WRITE_ALL.intersects(u.start | u.end)
                {
                    Err(self)
                } else {
                    Ok(u.start | u.end)
                }
            }
        }
    }
}

impl ResourceState for TextureState {
    type Id = TextureId;
    type Selector = hal::image::SubresourceRange;
    type Usage = TextureUsage;

    fn new(full_selector: &Self::Selector) -> Self {
        TextureState {
            mips: (0 .. full_selector.levels.end)
                .map(|_| {
                    let mut slices = ArrayVec::new();
                    let aspects_without_stencil = full_selector.aspects & !hal::format::Aspects::STENCIL;
                    if SEPARATE_DEPTH_STENCIL_STATES && full_selector.aspects != aspects_without_stencil {
                        slices.push((aspects_without_stencil, PlaneStates::default()));
                        slices.push((hal::format::Aspects::STENCIL, PlaneStates::default()));
                    } else {
                        slices.push((full_selector.aspects, PlaneStates::default()))
                    }
                    slices
                })
                .collect()
        }
    }

    fn query(&self, selector: Self::Selector) -> Option<Self::Usage> {
        let mut result = None;
        let num_levels = self.mips.len();
        let mip_start = num_levels.min(selector.levels.start as usize);
        let mip_end = num_levels.min(selector.levels.end as usize);
        for mip in self.mips[mip_start .. mip_end].iter() {
            for &(aspects, ref plane_states) in mip {
                if !selector.aspects.intersects(aspects) {
                    continue;
                }
                match plane_states.query(&selector.layers, |unit| unit.last) {
                    None => {}
                    Some(Ok(usage)) if result == Some(usage) => {}
                    Some(Ok(usage)) if result.is_none() => {
                        result = Some(usage);
                    }
                    Some(Ok(_)) | Some(Err(())) => return None,
                }
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
        for (mip_id, mip) in self.mips
            [selector.levels.start as usize .. selector.levels.end as usize]
            .iter_mut()
            .enumerate()
        {
            let level = selector.levels.start + mip_id as hal::image::Level;
            for &mut (mip_aspects, ref mut plane_states) in mip {
                let aspects = selector.aspects & mip_aspects;
                if aspects.is_empty() {
                    continue;
                }
                debug_assert_eq!(aspects, mip_aspects);
                let layers = plane_states.isolate(&selector.layers, Unit::new(usage));
                for &mut (ref range, ref mut unit) in layers {
                    if unit.last == usage && TextureUsage::ORDERED.contains(usage) {
                        continue;
                    }
                    let pending = PendingTransition {
                        id,
                        selector: hal::image::SubresourceRange {
                            aspects,
                            levels: level .. level + 1,
                            layers: range.clone(),
                        },
                        usage: unit.last .. usage,
                    };
                    unit.last = pending.record(output.as_mut(), usage)?;
                }
            }
        }
        Ok(())
    }

    fn merge(
        &mut self,
        id: Self::Id,
        other: &Self,
        stitch: Stitch,
        mut output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        assert!(output.is_some() || stitch == Stitch::Last);

        let mut temp = Vec::new();
        while self.mips.len() < other.mips.len() as usize {
            self.mips.push(MipState::default());
        }

        for (mip_id, (mip_self, mip_other)) in self.mips.iter_mut().zip(&other.mips).enumerate() {
            let level = mip_id as hal::image::Level;
            for (&mut (aspects, ref mut planes_self), &(aspects_other, ref planes_other)) in mip_self.iter_mut().zip(mip_other) {
                debug_assert_eq!(aspects, aspects_other);
                temp.extend(planes_self.merge(planes_other, 0));
                planes_self.clear();

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
                            let mut final_usage = end.select(stitch);
                            if start.last != final_usage
                                || !TextureUsage::ORDERED.contains(final_usage)
                            {
                                let pending = PendingTransition {
                                    id,
                                    selector: hal::image::SubresourceRange {
                                        aspects,
                                        levels: level .. level+1,
                                        layers: layers.clone(),
                                    },
                                    usage: start.last .. final_usage,
                                };
                                final_usage = pending.record(output.as_mut(), end.last)?;
                            }
                            Unit {
                                init: start.init,
                                last: final_usage,
                            }
                        }
                    };
                    planes_self.append(layers, unit);
                }
            }
        }

        Ok(())
    }

    fn optimize(&mut self) {
        for mip in self.mips.iter_mut() {
            for &mut (_, ref mut planes) in mip.iter_mut() {
                planes.coalesce();
            }
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
        let mut ts = TextureState::new(&SubresourceRange {
            aspects: Aspects::COLOR,
            levels: 0 .. 2,
            layers: 0 .. 10,
        });
        ts.mips[1][0].1 = PlaneStates::new(&[
            (1 .. 3, Unit::new(TextureUsage::SAMPLED)),
            (3 .. 5, Unit::new(TextureUsage::SAMPLED)),
            (5 .. 6, Unit::new(TextureUsage::STORAGE)),
        ]);
        assert_eq!(
            ts.query(SubresourceRange {
                aspects: Aspects::COLOR,
                levels: 1 .. 2,
                layers: 2 .. 5,
            }),
            // level 1 matches
            Some(TextureUsage::SAMPLED),
        );
        assert_eq!(
            ts.query(SubresourceRange {
                aspects: Aspects::DEPTH,
                levels: 1 .. 2,
                layers: 2 .. 5,
            }),
            // no depth found
            None,
        );
        assert_eq!(
            ts.query(SubresourceRange {
                aspects: Aspects::COLOR,
                levels: 0 .. 2,
                layers: 2 .. 5,
            }),
            // level 0 is empty, level 1 matches
            Some(TextureUsage::SAMPLED),
        );
        assert_eq!(
            ts.query(SubresourceRange {
                aspects: Aspects::COLOR,
                levels: 1 .. 2,
                layers: 1 .. 5,
            }),
            // level 1 matches with gaps
            Some(TextureUsage::SAMPLED),
        );
        assert_eq!(
            ts.query(SubresourceRange {
                aspects: Aspects::COLOR,
                levels: 1 .. 2,
                layers: 4 .. 6,
            }),
            // level 1 doesn't match
            None,
        );
    }
}
