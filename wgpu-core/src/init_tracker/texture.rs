use super::{InitTracker, MemoryInitKind};
use crate::{id::TextureId, track::TextureSelector};
use arrayvec::ArrayVec;
use std::ops::Range;

#[derive(Debug, Clone)]
pub(crate) struct TextureInitRange {
    pub(crate) mip_range: Range<u32>,
    pub(crate) layer_range: Range<u32>,
}

impl From<TextureSelector> for TextureInitRange {
    fn from(selector: TextureSelector) -> Self {
        TextureInitRange {
            mip_range: selector.levels,
            layer_range: selector.layers,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TextureInitTrackerAction {
    pub(crate) id: TextureId,
    pub(crate) range: TextureInitRange,
    pub(crate) kind: MemoryInitKind,
}

pub(crate) type TextureLayerInitTracker = InitTracker<u32>;

#[derive(Debug)]
pub(crate) struct TextureInitTracker {
    mips: ArrayVec<TextureLayerInitTracker, { hal::MAX_MIP_LEVELS as usize }>,
}

impl TextureInitTracker {
    pub(crate) fn new(depth_or_array_layers: u32, mip_level_count: u32) -> Self {
        TextureInitTracker {
            mips: std::iter::repeat(TextureLayerInitTracker::new(depth_or_array_layers))
                .take(mip_level_count as usize)
                .collect(),
        }
    }

    pub(crate) fn check_action(
        &self,
        action: &TextureInitTrackerAction,
    ) -> Option<TextureInitTrackerAction> {
        let mut mip_range_start = action.range.mip_range.start as usize;
        let mut mip_range_end = mip_range_start;
        let mut layer_range_start = std::u32::MAX;
        let mut layer_range_end = std::u32::MIN;

        for (i, mip_tracker) in self
            .mips
            .iter()
            .enumerate()
            .skip(action.range.mip_range.start as usize)
            .take((action.range.mip_range.end - action.range.mip_range.start) as usize)
        {
            match mip_tracker.check(action.range.layer_range.clone()) {
                Some(uninitialized_layer_range) => {
                    mip_range_end = i + 1;
                    layer_range_start = layer_range_start.min(uninitialized_layer_range.start);
                    layer_range_end = layer_range_end.max(uninitialized_layer_range.end);
                }
                None => {
                    mip_range_start += 1;
                }
            };
        }

        if mip_range_start < mip_range_end && layer_range_start < layer_range_end {
            Some(TextureInitTrackerAction {
                id: action.id,
                range: TextureInitRange {
                    mip_range: mip_range_start as u32..mip_range_end as u32,
                    layer_range: layer_range_start..layer_range_end,
                },
                kind: action.kind,
            })
        } else {
            None
        }
    }

    pub(crate) fn check_layer_range(
        &self,
        mip_level: u32,
        layer_range: Range<u32>,
    ) -> Option<Range<u32>> {
        self.mips[mip_level as usize].check(layer_range)
    }
}
