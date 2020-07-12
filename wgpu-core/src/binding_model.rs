/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
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
use std::borrow::Borrow;

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
                self.storage_textures.add(binding.visibility, count);
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

#[repr(C)]
#[derive(Debug)]
pub struct PipelineLayoutDescriptor {
    pub bind_group_layouts: *const BindGroupLayoutId,
    pub bind_group_layouts_length: usize,
}

#[derive(Clone, Debug)]
pub enum PipelineLayoutError {
    TooManyGroups(usize),
    TooManyBindings(BindingTypeMaxCountError),
}

#[derive(Debug)]
pub struct PipelineLayout<B: hal::Backend> {
    pub(crate) raw: B::PipelineLayout,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) life_guard: LifeGuard,
    pub(crate) bind_group_layout_ids: ArrayVec<[Stored<BindGroupLayoutId>; MAX_BIND_GROUPS]>,
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

// Note: Duplicated in wgpu-rs as BindingResource
#[derive(Debug)]
pub enum BindingResource<'a> {
    Buffer(BufferBinding),
    Sampler(SamplerId),
    TextureView(TextureViewId),
    TextureViewArray(&'a [TextureViewId]),
}

// Note: Duplicated in wgpu-rs as Binding
#[derive(Debug)]
pub struct BindGroupEntry<'a> {
    pub binding: u32,
    pub resource: BindingResource<'a>,
}

// Note: Duplicated in wgpu-rs as BindGroupDescriptor
#[derive(Debug)]
pub struct BindGroupDescriptor<'a> {
    pub label: Option<&'a str>,
    pub layout: BindGroupLayoutId,
    pub entries: &'a [BindGroupEntry<'a>],
}

#[derive(Debug)]
pub enum BindError {
    /// Number of dynamic offsets doesn't match the number of dynamic bindings
    /// in the bind group layout.
    MismatchedDynamicOffsetCount { actual: usize, expected: usize },
    /// Expected dynamic binding alignment was not met.
    UnalignedDynamicBinding { idx: usize },
    /// Dynamic offset would cause buffer overrun.
    DynamicBindingOutOfBounds { idx: usize },
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
            log::error!(
                "BindGroup has {} dynamic bindings, but {} dynamic offsets were provided",
                self.dynamic_binding_info.len(),
                offsets.len()
            );
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
                log::error!(
                    "Dynamic buffer offset index {}: {} needs to be aligned as a multiple of {}",
                    idx,
                    offset,
                    wgt::BIND_BUFFER_ALIGNMENT
                );
                return Err(BindError::UnalignedDynamicBinding { idx });
            }

            if offset as wgt::BufferAddress > info.maximum_dynamic_offset {
                log::error!(
                    "Dynamic offset index {} with value {} overruns underlying buffer. Dynamic offset must be no more than {}",
                    idx,
                    offset,
                    info.maximum_dynamic_offset,
                );
                return Err(BindError::DynamicBindingOutOfBounds { idx });
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
