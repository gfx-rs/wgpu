/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    id::{BindGroupLayoutId, BufferId, DeviceId, SamplerId, TextureViewId},
    track::{TrackerSet, DUMMY_SELECTOR},
    FastHashMap, LifeGuard, RefCount, Stored, MAX_BIND_GROUPS,
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
    MissingExtension(wgt::Extensions),
    MissingCapability(wgt::Capabilities),
    /// Arrays of bindings can't be 0 elements long
    ZeroCount,
    /// Arrays of bindings unsupported for this type of binding
    ArrayUnsupported,
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
}

pub(crate) type BindEntryMap = FastHashMap<u32, wgt::BindGroupLayoutEntry>;

#[derive(Debug)]
pub struct BindGroupLayout<B: hal::Backend> {
    pub(crate) raw: B::DescriptorSetLayout,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) life_guard: LifeGuard,
    pub(crate) entries: BindEntryMap,
    pub(crate) desc_counts: DescriptorCounts,
    pub(crate) dynamic_count: usize,
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
    pub size: wgt::BufferSize,
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
    pub bindings: &'a [BindGroupEntry<'a>],
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
