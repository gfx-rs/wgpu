/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    device::SHADER_STAGE_COUNT,
    id::{BindGroupLayoutId, BufferId, DeviceId, SamplerId, TextureViewId},
    track::{TrackerSet, DUMMY_SELECTOR},
    FastHashMap, LifeGuard, MultiRefCount, RefCount, Stored, MAX_BIND_GROUPS,
};

use arrayvec::ArrayVec;
use gfx_descriptor::{DescriptorCounts, DescriptorSet};

#[cfg(feature = "replay")]
use serde::Deserialize;
#[cfg(feature = "trace")]
use serde::Serialize;

use std::{borrow::Borrow, fmt};

#[derive(Clone, Debug)]
pub enum BindGroupLayoutError {
    ConflictBinding(u32),
    MissingFeature(wgt::Features),
    /// Arrays of bindings can't be 0 elements long
    ZeroCount,
    /// Arrays of bindings unsupported for this type of binding
    ArrayUnsupported,
    /// Bindings go over binding count limits
    TooManyBindings(BindingTypeMaxCountError),
}

#[derive(Clone, Debug)]
pub enum BindGroupError {
    /// Number of bindings in bind group descriptor does not match
    /// the number of bindings defined in the bind group layout.
    BindingsNumMismatch { actual: usize, expected: usize },
    /// Unable to find a corresponding declaration for the given binding,
    MissingBindingDeclaration(u32),
    /// The given binding has a different type than the one in the layout.
    WrongBindingType {
        // Index of the binding
        binding: u32,
        // The type given to the function
        actual: wgt::BindingType,
        // Human-readable description of expected types
        expected: &'static str,
    },
    /// The given sampler is/is not a comparison sampler,
    /// while the layout type indicates otherwise.
    WrongSamplerComparison,
    /// Uniform buffer binding range exceeds [`wgt::Limits::max_uniform_buffer_binding_size`] limit
    UniformBufferRangeTooLarge,
}

#[derive(Clone, Debug)]
pub struct BindingTypeMaxCountError {
    pub kind: BindingTypeMaxCountErrorKind,
    pub stage: wgt::ShaderStage,
    pub count: u32,
}

#[derive(Clone, Debug)]
pub enum BindingTypeMaxCountErrorKind {
    DynamicUniformBuffers,
    DynamicStorageBuffers,
    SampledTextures,
    Samplers,
    StorageBuffers,
    StorageTextures,
    UniformBuffers,
}

#[derive(Debug, Default)]
pub(crate) struct PerStageBindingTypeCounter {
    vertex: u32,
    fragment: u32,
    compute: u32,
}
impl PerStageBindingTypeCounter {
    pub(crate) fn add(&mut self, stage: wgt::ShaderStage, count: u32) {
        if stage.contains(wgt::ShaderStage::VERTEX) {
            self.vertex += count;
        }
        if stage.contains(wgt::ShaderStage::FRAGMENT) {
            self.fragment += count;
        }
        if stage.contains(wgt::ShaderStage::COMPUTE) {
            self.compute += count;
        }
    }

    pub(crate) fn max(&self) -> (wgt::ShaderStage, u32) {
        let max_value = self.vertex.max(self.fragment.max(self.compute));
        let mut stage = wgt::ShaderStage::NONE;
        if max_value == self.vertex {
            stage |= wgt::ShaderStage::VERTEX
        }
        if max_value == self.fragment {
            stage |= wgt::ShaderStage::FRAGMENT
        }
        if max_value == self.compute {
            stage |= wgt::ShaderStage::COMPUTE
        }
        (stage, max_value)
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        self.vertex = self.vertex.max(other.vertex);
        self.fragment = self.fragment.max(other.fragment);
        self.compute = self.compute.max(other.compute);
    }

    pub(crate) fn validate(
        &self,
        limit: u32,
        kind: BindingTypeMaxCountErrorKind,
    ) -> Result<(), BindingTypeMaxCountError> {
        let (stage, count) = self.max();
        if limit < count {
            Err(BindingTypeMaxCountError { kind, stage, count })
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct BindingTypeMaxCountValidator {
    dynamic_uniform_buffers: u32,
    dynamic_storage_buffers: u32,
    sampled_textures: PerStageBindingTypeCounter,
    samplers: PerStageBindingTypeCounter,
    storage_buffers: PerStageBindingTypeCounter,
    storage_textures: PerStageBindingTypeCounter,
    uniform_buffers: PerStageBindingTypeCounter,
}

impl BindingTypeMaxCountValidator {
    pub(crate) fn add_binding(&mut self, binding: &wgt::BindGroupLayoutEntry) {
        let count = binding.count.unwrap_or(1);
        match binding.ty {
            wgt::BindingType::UniformBuffer { dynamic, .. } => {
                self.uniform_buffers.add(binding.visibility, count);
                if dynamic {
                    self.dynamic_uniform_buffers += count;
                }
            }
            wgt::BindingType::StorageBuffer { dynamic, .. } => {
                self.storage_buffers.add(binding.visibility, count);
                if dynamic {
                    self.dynamic_storage_buffers += count;
                }
            }
            wgt::BindingType::Sampler { .. } => {
                self.samplers.add(binding.visibility, count);
            }
            wgt::BindingType::SampledTexture { .. } => {
                self.sampled_textures.add(binding.visibility, count);
            }
            wgt::BindingType::StorageTexture { .. } => {
                self.storage_textures.add(binding.visibility, count);
            }
        }
    }

    pub(crate) fn merge(&mut self, other: &Self) {
        self.dynamic_uniform_buffers += other.dynamic_uniform_buffers;
        self.dynamic_storage_buffers += other.dynamic_storage_buffers;
        self.sampled_textures.merge(&other.sampled_textures);
        self.samplers.merge(&other.samplers);
        self.storage_buffers.merge(&other.storage_buffers);
        self.storage_textures.merge(&other.storage_textures);
        self.uniform_buffers.merge(&other.uniform_buffers);
    }

    pub(crate) fn validate(&self, limits: &wgt::Limits) -> Result<(), BindingTypeMaxCountError> {
        if limits.max_dynamic_uniform_buffers_per_pipeline_layout < self.dynamic_uniform_buffers {
            return Err(BindingTypeMaxCountError {
                kind: BindingTypeMaxCountErrorKind::DynamicUniformBuffers,
                stage: wgt::ShaderStage::NONE,
                count: self.dynamic_uniform_buffers,
            });
        }
        if limits.max_dynamic_storage_buffers_per_pipeline_layout < self.dynamic_storage_buffers {
            return Err(BindingTypeMaxCountError {
                kind: BindingTypeMaxCountErrorKind::DynamicStorageBuffers,
                stage: wgt::ShaderStage::NONE,
                count: self.dynamic_storage_buffers,
            });
        }
        self.sampled_textures.validate(
            limits.max_sampled_textures_per_shader_stage,
            BindingTypeMaxCountErrorKind::SampledTextures,
        )?;
        self.storage_buffers.validate(
            limits.max_storage_buffers_per_shader_stage,
            BindingTypeMaxCountErrorKind::StorageBuffers,
        )?;
        self.samplers.validate(
            limits.max_samplers_per_shader_stage,
            BindingTypeMaxCountErrorKind::Samplers,
        )?;
        self.storage_buffers.validate(
            limits.max_storage_buffers_per_shader_stage,
            BindingTypeMaxCountErrorKind::StorageBuffers,
        )?;
        self.storage_textures.validate(
            limits.max_storage_textures_per_shader_stage,
            BindingTypeMaxCountErrorKind::StorageTextures,
        )?;
        self.uniform_buffers.validate(
            limits.max_uniform_buffers_per_shader_stage,
            BindingTypeMaxCountErrorKind::UniformBuffers,
        )?;
        Ok(())
    }
}

pub(crate) type BindEntryMap = FastHashMap<u32, wgt::BindGroupLayoutEntry>;

#[derive(Debug)]
pub struct BindGroupLayout<B: hal::Backend> {
    pub(crate) raw: B::DescriptorSetLayout,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) multi_ref_count: MultiRefCount,
    pub(crate) entries: BindEntryMap,
    pub(crate) desc_counts: DescriptorCounts,
    pub(crate) dynamic_count: usize,
    pub(crate) count_validator: BindingTypeMaxCountValidator,
}

#[derive(Clone, Debug)]
pub enum PipelineLayoutError {
    TooManyGroups(usize),
    TooManyBindings(BindingTypeMaxCountError),
    PushConstantRangeTooLarge { index: usize },
    MoreThanOnePushConstantRangePerStage { index: usize },
    MisalignedPushConstantRange { index: usize },
    MissingFeature(wgt::Features),
}

#[derive(Clone, Debug)]
pub enum PushConstantUploadError {
    TooLarge {
        offset: u32,
        end_offset: u32,
        idx: usize,
        range: wgt::PushConstantRange,
    },
    PartialRangeMatch {
        actual: wgt::ShaderStage,
        idx: usize,
        matched: wgt::ShaderStage,
    },
    MissingStages {
        actual: wgt::ShaderStage,
        idx: usize,
        missing: wgt::ShaderStage,
    },
    UnmatchedStages {
        actual: wgt::ShaderStage,
        unmatched: wgt::ShaderStage,
    },
    Unaligned(u32),
}

impl fmt::Display for PushConstantUploadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::TooLarge { offset, end_offset, idx, range } => write!(
                f,
                "provided push constant with indices {}..{} overruns matching push constant range (index {}) with stage(s) {:?} and indices {}..{}",
                offset,
                end_offset,
                idx,
                range.stages,
                range.range.start,
                range.range.end,
            ),
            Self::PartialRangeMatch { actual, idx, matched } => write!(
                f,
                "provided push constant is for stage(s) {:?}, stage with a partial match found at index {} with stage(s) {:?}, however push constants must be complete matches",
                actual,
                idx,
                matched,
            ),
            Self::MissingStages { actual, idx, missing } => write!(
                f,
                "provided push constant is for stage(s) {:?}, but intersects a push constant range (at index {}) with stage(s) {:?}. Push constants must provide the stages for all ranges they intersect",
                actual,
                idx,
                missing,
            ),
            Self::UnmatchedStages { actual, unmatched } => write!(
                f,
                "provided push constant is for stage(s) {:?}, however the pipeline layout has no push constant range for the stage(s) {:?}",
                actual,
                unmatched,
            ),
            Self::Unaligned(offset) => write!(
                f,
                "provided push constant offset {} must be aligned to {}",
                offset,
                wgt::PUSH_CONSTANT_ALIGNMENT,
            )
        }
    }
}

#[derive(Debug)]
pub struct PipelineLayout<B: hal::Backend> {
    pub(crate) raw: B::PipelineLayout,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) life_guard: LifeGuard,
    pub(crate) bind_group_layout_ids: ArrayVec<[Stored<BindGroupLayoutId>; MAX_BIND_GROUPS]>,
    pub(crate) push_constant_ranges: ArrayVec<[wgt::PushConstantRange; SHADER_STAGE_COUNT]>,
}

impl<B: hal::Backend> PipelineLayout<B> {
    /// Validate push constants match up with expected ranges.
    pub(crate) fn validate_push_constant_ranges(
        &self,
        stages: wgt::ShaderStage,
        offset: u32,
        end_offset: u32,
    ) -> Result<(), PushConstantUploadError> {
        // Don't need to validate size against the push constant size limit here,
        // as push constant ranges are already validated to be within bounds,
        // and we validate that they are within the ranges.

        if offset % wgt::PUSH_CONSTANT_ALIGNMENT != 0 {
            return Err(PushConstantUploadError::Unaligned(offset));
        }

        // Push constant validation looks very complicated on the surface, but
        // the problem can be range-reduced pretty well.
        //
        // Push constants require (summarized from the vulkan spec):
        // 1. For each byte in the range and for each shader stage in stageFlags,
        //    there must be a push constant range in the layout that includes that
        //    byte and that stage.
        // 2. For each byte in the range and for each push constant range that overlaps that byte,
        //    `stage` must include all stages in that push constant range’s `stage`.
        //
        // However there are some additional constraints that help us:
        // 3. All push constant ranges are the only range that can access that stage.
        //    i.e. if one range has VERTEX, no other range has VERTEX
        //
        // Therefore we can simplify the checks in the following ways:
        // - Because 3 guarantees that the push constant range has a unique stage,
        //   when we check for 1, we can simply check that our entire updated range
        //   is within a push constant range. i.e. our range for a specific stage cannot
        //   intersect more than one push constant range.
        let mut used_stages = wgt::ShaderStage::NONE;
        for (idx, range) in self.push_constant_ranges.iter().enumerate() {
            // contains not intersects due to 2
            if stages.contains(range.stages) {
                if !(range.range.start <= offset && end_offset <= range.range.end) {
                    return Err(PushConstantUploadError::TooLarge {
                        offset,
                        end_offset,
                        idx,
                        range: range.clone(),
                    });
                }
                used_stages |= range.stages;
            } else if stages.intersects(range.stages) {
                // Will be caught by used stages check below, but we can do this because of 1
                // and is more helpful to the user.
                return Err(PushConstantUploadError::PartialRangeMatch {
                    actual: stages,
                    idx,
                    matched: range.stages,
                });
            }

            // The push constant range intersects range we are uploading
            if offset < range.range.end && range.range.start < end_offset {
                // But requires stages we don't provide
                if !stages.contains(range.stages) {
                    return Err(PushConstantUploadError::MissingStages {
                        actual: stages,
                        idx,
                        missing: stages,
                    });
                }
            }
        }
        if used_stages != stages {
            return Err(PushConstantUploadError::UnmatchedStages {
                actual: stages,
                unmatched: stages - used_stages,
            });
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct BufferBinding {
    pub buffer_id: BufferId,
    pub offset: wgt::BufferAddress,
    pub size: Option<wgt::BufferSize>,
}

// Note: Duplicated in `wgpu-rs` as `BindingResource`
// They're different enough that it doesn't make sense to share a common type
#[derive(Debug)]
pub enum BindingResource<'a> {
    Buffer(BufferBinding),
    Sampler(SamplerId),
    TextureView(TextureViewId),
    TextureViewArray(&'a [TextureViewId]),
}

pub type BindGroupEntry<'a> = wgt::BindGroupEntry<BindingResource<'a>>;

pub type BindGroupDescriptor<'a> =
    wgt::BindGroupDescriptor<'a, BindGroupLayoutId, BindGroupEntry<'a>>;

#[derive(Clone, Debug)]
pub enum BindError {
    MismatchedDynamicOffsetCount { actual: usize, expected: usize },
    UnalignedDynamicBinding { idx: usize, offset: u32 },
    DynamicBindingOutOfBounds { idx: usize, offset: u32, max: u64 },
}

impl fmt::Display for BindError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::MismatchedDynamicOffsetCount { actual, expected } =>
                write!(
                    f,
                    "number of dynamic offsets ({}) doesn't match the number of dynamic bindings in the bind group layout ({})",
                    actual,
                    expected,
                ),
            Self::UnalignedDynamicBinding { idx, offset } => write!(
                f,
                "dynamic binding at index {}: offset {} must be aligned to {}",
                idx,
                offset,
                wgt::BIND_BUFFER_ALIGNMENT,
            ),
            Self::DynamicBindingOutOfBounds { idx, offset, max } => write!(
                f,
                "dynamic binding at index {} with offset {} would overrun the buffer (limit: {})",
                idx,
                offset,
                max,
            ),
        }
    }
}

#[derive(Debug)]
pub struct BindGroupDynamicBindingData {
    /// The maximum value the dynamic offset can have before running off the end of the buffer.
    pub(crate) maximum_dynamic_offset: wgt::BufferAddress,
}

#[derive(Debug)]
pub struct BindGroup<B: hal::Backend> {
    pub(crate) raw: DescriptorSet<B>,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) layout_id: BindGroupLayoutId,
    pub(crate) life_guard: LifeGuard,
    pub(crate) used: TrackerSet,
    pub(crate) dynamic_binding_info: Vec<BindGroupDynamicBindingData>,
}

impl<B: hal::Backend> BindGroup<B> {
    pub(crate) fn validate_dynamic_bindings(
        &self,
        offsets: &[wgt::DynamicOffset],
    ) -> Result<(), BindError> {
        if self.dynamic_binding_info.len() != offsets.len() {
            return Err(BindError::MismatchedDynamicOffsetCount {
                expected: self.dynamic_binding_info.len(),
                actual: offsets.len(),
            });
        }

        for (idx, (info, &offset)) in self
            .dynamic_binding_info
            .iter()
            .zip(offsets.iter())
            .enumerate()
        {
            if offset as wgt::BufferAddress % wgt::BIND_BUFFER_ALIGNMENT != 0 {
                return Err(BindError::UnalignedDynamicBinding { idx, offset });
            }

            if offset as wgt::BufferAddress > info.maximum_dynamic_offset {
                return Err(BindError::DynamicBindingOutOfBounds {
                    idx,
                    offset,
                    max: info.maximum_dynamic_offset,
                });
            }
        }

        Ok(())
    }
}

impl<B: hal::Backend> Borrow<RefCount> for BindGroup<B> {
    fn borrow(&self) -> &RefCount {
        self.life_guard.ref_count.as_ref().unwrap()
    }
}

impl<B: hal::Backend> Borrow<()> for BindGroup<B> {
    fn borrow(&self) -> &() {
        &DUMMY_SELECTOR
    }
}
