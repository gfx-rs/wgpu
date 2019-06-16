use crate::{
    conv,
    device::MAX_MIP_LEVELS,
    resource::TextureUsage,
    TextureId,
};
use super::{range::RangedStates, PendingTransition, ResourceState, Stitch, Unit};

use arrayvec::ArrayVec;

use std::ops::Range;


type PlaneStates = RangedStates<hal::image::Layer, Unit<TextureUsage>>;


//TODO: store `hal::image::State` here to avoid extra conversions
#[derive(Clone, Debug, Default)]
struct MipState {
    color: PlaneStates,
    depth: PlaneStates,
    stencil: PlaneStates,
}

#[derive(Clone, Debug, Default)]
pub struct TextureState {
    mips: ArrayVec<[MipState; MAX_MIP_LEVELS]>,
}

impl PendingTransition<TextureState> {
    /// Produce the gfx-hal image states corresponding to the transition.
    pub fn to_states(&self) -> Range<hal::image::State> {
        conv::map_texture_state(self.usage.start, self.selector.aspects) ..
        conv::map_texture_state(self.usage.end, self.selector.aspects)
    }

    //TODO: make this less awkward!
    /// Check for the validity of `self` with regards to the presence of `output`.
    ///
    /// Return the end usage if the `output` is provided and pushes self to it.
    /// Otherwise, return the extended usage, or an error if extension is impossible.
    ///
    /// When a transition is generated, returns the specified `replace` usage.
    fn record(
        self, output: Option<&mut &mut Vec<Self>>, replace: TextureUsage
    ) -> Result<TextureUsage, Self> {
        let u = self.usage.clone();
        match output {
            Some(out) => {
                out.push(self);
                Ok(replace)
            }
            None => {
                if !u.start.is_empty() && TextureUsage::WRITE_ALL.intersects(u.start | u.end) {
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

    fn query(
        &self,
        selector: Self::Selector,
    ) -> Option<Self::Usage> {
        let mut result = None;
        let num_levels = self.mips.len();
        let mip_start = num_levels.min(selector.levels.start as usize);
        let mip_end = num_levels.min(selector.levels.end as usize);
        for mip in self.mips[mip_start .. mip_end].iter() {
            for &(aspect, plane_states) in &[
                (hal::format::Aspects::COLOR, &mip.color),
                (hal::format::Aspects::DEPTH, &mip.depth),
                (hal::format::Aspects::STENCIL, &mip.stencil),
            ] {
                if !selector.aspects.contains(aspect) {
                    continue
                }
                match plane_states.query(&selector.layers, |unit| unit.last) {
                    None => {}
                    Some(Ok(usage)) if result == Some(usage) => {}
                    Some(Ok(usage)) if result.is_none() => {
                        result = Some(usage);
                    }
                    Some(Ok(_)) |
                    Some(Err(())) => return None,
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
        while self.mips.len() < selector.levels.end as usize {
            self.mips.push(MipState::default());
        }
        for (mip_id, mip) in self
            .mips[selector.levels.start as usize .. selector.levels.end as usize]
            .iter_mut()
            .enumerate()
        {
            let level = selector.levels.start + mip_id as hal::image::Level;
            for &mut (aspect, ref mut plane_states) in &mut [
                (hal::format::Aspects::COLOR, &mut mip.color),
                (hal::format::Aspects::DEPTH, &mut mip.depth),
                (hal::format::Aspects::STENCIL, &mut mip.stencil),
            ] {
                if !selector.aspects.contains(aspect) {
                    continue
                }
                let layers = plane_states.isolate(&selector.layers, Unit::new(usage));
                for &mut (ref range, ref mut unit) in layers {
                    if unit.last == usage {
                        continue
                    }
                    let pending = PendingTransition {
                        id,
                        selector: hal::image::SubresourceRange {
                            aspects: hal::format::Aspects::COLOR,
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
        let mut temp = Vec::new();
        while self.mips.len() < other.mips.len() as usize {
            self.mips.push(MipState::default());
        }

        for (mip_id, (mip_self, mip_other)) in self.mips
            .iter_mut()
            .zip(&other.mips)
            .enumerate()
        {
            let level = mip_id as hal::image::Level;
            for &mut (aspects, ref mut planes_self, planes_other) in &mut [
                (hal::format::Aspects::COLOR, &mut mip_self.color, &mip_other.color),
                (hal::format::Aspects::DEPTH, &mut mip_self.depth, &mip_other.depth),
                (hal::format::Aspects::STENCIL, &mut mip_self.stencil, &mip_other.stencil),
            ] {
                temp.extend(planes_self.merge(planes_other, 0));
                planes_self.clear();

                for (layers, states) in temp.drain(..) {
                    let unit = match states {
                        Range { start: None, end: None } => unreachable!(),
                        Range { start: Some(start), end: None } => start,
                        Range { start: None, end: Some(end) } => Unit::new(end.select(stitch)),
                        Range { start: Some(start), end: Some(end) } => {
                            let mut final_usage = end.select(stitch);
                            if start.last != final_usage {
                                let pending = PendingTransition {
                                    id,
                                    selector: hal::image::SubresourceRange {
                                        aspects,
                                        levels: level .. level + 1,
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
            mip.color.coalesce();
            mip.depth.coalesce();
            mip.stencil.coalesce();
        }
    }
}


#[cfg(test)]
mod test {
    //TODO: change() and merge() tests
    //use crate::TypedId;
    use super::*;
    use hal::{
        format::Aspects,
        image::SubresourceRange,
    };

    #[test]
    fn query() {
        let mut ts = TextureState::default();
        ts.mips.push(MipState::default());
        ts.mips.push(MipState::default());
        ts.mips[1].color = PlaneStates::new(&[
            (1..3, Unit::new(TextureUsage::SAMPLED)),
            (3..5, Unit::new(TextureUsage::SAMPLED)),
            (5..6, Unit::new(TextureUsage::STORAGE)),
        ]);
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
                aspects: Aspects::DEPTH,
                levels: 1..2,
                layers: 2..5,
            }),
            // no depth found
            None,
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
