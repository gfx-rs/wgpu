use crate::{
    conv,
    device::MAX_MIP_LEVELS,
    resource::TextureUsage,
    TextureId,
};
use super::{range::RangedStates, PendingTransition, ResourceState, Stitch, Unit};

use arrayvec::ArrayVec;

use std::ops::Range;


type PlaneStates<T> = RangedStates<hal::image::Layer, T>;

//TODO: store `hal::image::State` here to avoid extra conversions
#[derive(Clone, Copy, Debug, PartialEq)]
struct DepthStencilState {
    depth: Unit<TextureUsage>,
    stencil: Unit<TextureUsage>,
}

#[derive(Clone, Debug, Default)]
pub struct TextureStates {
    color_mips: ArrayVec<[PlaneStates<Unit<TextureUsage>>; MAX_MIP_LEVELS]>,
    depth_stencil: PlaneStates<DepthStencilState>,
}

impl PendingTransition<TextureStates> {
    /// Produce the gfx-hal image states corresponding to the transition.
    pub fn to_states(&self) -> Range<hal::image::State> {
        conv::map_texture_state(self.usage.start, self.selector.aspects) ..
        conv::map_texture_state(self.usage.end, self.selector.aspects)
    }

    /// Check for the validity of `self` with regards to the presense of `output`.
    ///
    /// Return the end usage if the `output` is provided and pushes self to it.
    /// Otherwise, return the extended usage, or an error if extension is impossible.
    fn record(self, output: Option<&mut &mut Vec<Self>>) -> Result<TextureUsage, Self> {
        let u = self.usage.clone();
        match output {
            Some(out) => {
                out.push(self);
                Ok(u.end)
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

impl ResourceState for TextureStates {
    type Id = TextureId;
    type Selector = hal::image::SubresourceRange;
    type Usage = TextureUsage;

    fn query(
        &self,
        selector: Self::Selector,
    ) -> Option<Self::Usage> {
        let mut usage = None;
        if selector.aspects.contains(hal::format::Aspects::COLOR) {
            let num_levels = self.color_mips.len();
            let layer_start = num_levels.min(selector.levels.start as usize);
            let layer_end = num_levels.min(selector.levels.end as usize);
            for layer in self.color_mips[layer_start .. layer_end].iter() {
                for &(ref range, ref unit) in layer.iter() {
                    if range.end > selector.layers.start && range.start < selector.layers.end {
                        let old = usage.replace(unit.last);
                        if old.is_some() && old != usage {
                            return None
                        }
                    }
                }
            }
        }
        if selector.aspects.intersects(hal::format::Aspects::DEPTH | hal::format::Aspects::STENCIL) {
            for &(ref range, ref ds) in self.depth_stencil.iter() {
                if range.end > selector.layers.start && range.start < selector.layers.end {
                    if selector.aspects.contains(hal::format::Aspects::DEPTH) {
                        let old = usage.replace(ds.depth.last);
                        if old.is_some() && old != usage {
                            return None
                        }
                    }
                    if selector.aspects.contains(hal::format::Aspects::STENCIL) {
                        let old = usage.replace(ds.stencil.last);
                        if old.is_some() && old != usage {
                            return None
                        }
                    }
                }
            }
        }
        usage
    }

    fn change(
        &mut self,
        id: Self::Id,
        selector: Self::Selector,
        usage: Self::Usage,
        mut output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        if selector.aspects.contains(hal::format::Aspects::COLOR) {
            while self.color_mips.len() < selector.levels.end as usize {
                self.color_mips.push(PlaneStates::default());
            }
            for level in selector.levels.clone() {
                let layers = self
                    .color_mips[level as usize]
                    .isolate(&selector.layers, Unit::new(usage));
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
                    unit.last = pending.record(output.as_mut())?;
                }
            }
        }
        if selector.aspects.intersects(hal::format::Aspects::DEPTH | hal::format::Aspects::STENCIL) {
            for level in selector.levels.clone() {
                let ds_state = DepthStencilState {
                    depth: Unit::new(usage),
                    stencil: Unit::new(usage),
                };
                for &mut (ref range, ref mut unit) in self.depth_stencil
                    .isolate(&selector.layers, ds_state)
                {
                    //TODO: check if anything needs to be done when only one of the depth/stencil
                    // is selected?
                    if unit.depth.last != usage && selector.aspects.contains(hal::format::Aspects::DEPTH) {
                        let pending = PendingTransition {
                            id,
                            selector: hal::image::SubresourceRange {
                                aspects: hal::format::Aspects::DEPTH,
                                levels: level .. level + 1,
                                layers: range.clone(),
                            },
                            usage: unit.depth.last .. usage,
                        };
                        unit.depth.last = pending.record(output.as_mut())?;
                    }
                    if unit.stencil.last != usage && selector.aspects.contains(hal::format::Aspects::STENCIL) {
                        let pending = PendingTransition {
                            id,
                            selector: hal::image::SubresourceRange {
                                aspects: hal::format::Aspects::STENCIL,
                                levels: level .. level + 1,
                                layers: range.clone(),
                            },
                            usage: unit.stencil.last .. usage,
                        };
                        unit.stencil.last = pending.record(output.as_mut())?;
                    }
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
        let mut temp_color = Vec::new();
        while self.color_mips.len() < other.color_mips.len() {
            self.color_mips.push(PlaneStates::default());
        }
        for (mip_id, (mip_self, mip_other)) in self.color_mips
            .iter_mut()
            .zip(&other.color_mips)
            .enumerate()
        {
            temp_color.extend(mip_self.merge(mip_other, 0));
            mip_self.clear();
            for (layers, states) in temp_color.drain(..) {
                let mut color_usage = states.start.last .. states.end.select(stitch);
                if color_usage.start != color_usage.end {
                    let level = mip_id as hal::image::Level;
                    let pending = PendingTransition {
                        id,
                        selector: hal::image::SubresourceRange {
                            aspects: hal::format::Aspects::COLOR,
                            levels: level .. level + 1,
                            layers: layers.clone(),
                        },
                        usage: color_usage.clone(),
                    };
                    color_usage.end = pending.record(output.as_mut())?;
                }
                mip_self.append(layers, Unit {
                    init: states.start.init,
                    last: color_usage.end,
                });
            }
        }

        let mut temp_ds = Vec::new();
        temp_ds.extend(self.depth_stencil.merge(&other.depth_stencil, 0));
        self.depth_stencil.clear();
        for (layers, states) in temp_ds.drain(..) {
            let mut usage_depth = states.start.depth.last .. states.end.depth.select(stitch);
            let mut usage_stencil = states.start.stencil.last .. states.end.stencil.select(stitch);
            if usage_depth.start != usage_depth.end {
                let pending = PendingTransition {
                    id,
                    selector: hal::image::SubresourceRange {
                        aspects: hal::format::Aspects::DEPTH,
                        levels: 0 .. 1,
                        layers: layers.clone(),
                    },
                    usage: usage_depth.clone(),
                };
                usage_depth.end = pending.record(output.as_mut())?;
            }
            if usage_stencil.start != usage_stencil.end {
                let pending = PendingTransition {
                    id,
                    selector: hal::image::SubresourceRange {
                        aspects: hal::format::Aspects::STENCIL,
                        levels: 0 .. 1,
                        layers: layers.clone(),
                    },
                    usage: usage_stencil.clone(),
                };
                usage_stencil.end = pending.record(output.as_mut())?;
            }
            self.depth_stencil.append(layers, DepthStencilState {
                depth: Unit {
                    init: states.start.depth.init,
                    last: usage_depth.end,
                },
                stencil: Unit {
                    init: states.start.stencil.init,
                    last: usage_stencil.end,
                },
            });
        }

        Ok(())
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
        let mut ts = TextureStates::default();
        ts.color_mips.push(PlaneStates::default());
        ts.color_mips.push(PlaneStates::new(&[
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
