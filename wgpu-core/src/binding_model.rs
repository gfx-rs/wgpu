/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    id::{BindGroupLayoutId, BufferId, DeviceId, SamplerId, TextureViewId},
    resource::TextureViewDimension,
    track::{DUMMY_SELECTOR, TrackerSet},
    BufferAddress,
    FastHashMap,
    LifeGuard,
    RefCount,
    Stored,
};

use arrayvec::ArrayVec;
use rendy_descriptor::{DescriptorRanges, DescriptorSet};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;

pub const MAX_BIND_GROUPS: usize = 4;

bitflags::bitflags! {
    #[repr(transparent)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct ShaderStage: u32 {
        const NONE = 0;
        const VERTEX = 1;
        const FRAGMENT = 2;
        const COMPUTE = 4;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BindingType {
    UniformBuffer = 0,
    StorageBuffer = 1,
    ReadonlyStorageBuffer = 2,
    Sampler = 3,
    SampledTexture = 4,
    StorageTexture = 5,
}

#[repr(C)]
#[derive(Clone, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BindGroupLayoutBinding {
    pub binding: u32,
    pub visibility: ShaderStage,
    pub ty: BindingType,
    pub texture_dimension: TextureViewDimension,
    pub multisampled: bool,
    pub dynamic: bool,
}

#[repr(C)]
#[derive(Debug)]
pub struct BindGroupLayoutDescriptor {
    pub bindings: *const BindGroupLayoutBinding,
    pub bindings_length: usize,
}

#[derive(Debug)]
pub struct BindGroupLayout<B: hal::Backend> {
    pub(crate) raw: B::DescriptorSetLayout,
    pub(crate) bindings: FastHashMap<u32, BindGroupLayoutBinding>,
    pub(crate) desc_ranges: DescriptorRanges,
    pub(crate) dynamic_count: usize,
}

#[repr(C)]
#[derive(Debug)]
pub struct PipelineLayoutDescriptor {
    pub bind_group_layouts: *const BindGroupLayoutId,
    pub bind_group_layouts_length: usize,
}

#[derive(Debug)]
pub struct PipelineLayout<B: hal::Backend> {
    pub(crate) raw: B::PipelineLayout,
    pub(crate) bind_group_layout_ids: ArrayVec<[BindGroupLayoutId; MAX_BIND_GROUPS]>,
}

#[repr(C)]
#[derive(Debug)]
pub struct BufferBinding {
    pub buffer: BufferId,
    pub offset: BufferAddress,
    pub size: BufferAddress,
}

#[repr(C)]
#[derive(Debug)]
pub enum BindingResource {
    Buffer(BufferBinding),
    Sampler(SamplerId),
    TextureView(TextureViewId),
}

#[repr(C)]
#[derive(Debug)]
pub struct BindGroupBinding {
    pub binding: u32,
    pub resource: BindingResource,
}

#[repr(C)]
#[derive(Debug)]
pub struct BindGroupDescriptor {
    pub layout: BindGroupLayoutId,
    pub bindings: *const BindGroupBinding,
    pub bindings_length: usize,
}

#[derive(Debug)]
pub struct BindGroup<B: hal::Backend> {
    pub(crate) raw: DescriptorSet<B>,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) layout_id: BindGroupLayoutId,
    pub(crate) life_guard: LifeGuard,
    pub(crate) used: TrackerSet,
    pub(crate) dynamic_count: usize,
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
