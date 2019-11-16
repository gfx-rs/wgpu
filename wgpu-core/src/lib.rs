/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

pub mod backend {
    #[cfg(windows)]
    pub use gfx_backend_dx11::Backend as Dx11;
    #[cfg(windows)]
    pub use gfx_backend_dx12::Backend as Dx12;
    pub use gfx_backend_empty::Backend as Empty;
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub use gfx_backend_metal::Backend as Metal;
    #[cfg(any(
        not(any(target_os = "ios", target_os = "macos")),
        feature = "gfx-backend-vulkan"
    ))]
    pub use gfx_backend_vulkan::Backend as Vulkan;
}

pub mod binding_model;
pub mod command;
pub mod conv;
pub mod device;
pub mod hub;
pub mod id;
pub mod instance;
pub mod pipeline;
pub mod resource;
pub mod swap_chain;
pub mod track;

pub use hal::pso::read_spirv;

use std::{
    os::raw::c_char,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
};

type SubmissionIndex = usize;
type Index = u32;
type Epoch = u32;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Backend {
    Empty = 0,
    Vulkan = 1,
    Metal = 2,
    Dx12 = 3,
    Dx11 = 4,
    Gl = 5,
}

pub type BufferAddress = u64;
pub type RawString = *const c_char;

//TODO: make it private. Currently used for swapchain creation impl.
#[derive(Debug)]
pub struct RefCount(ptr::NonNull<AtomicUsize>);

unsafe impl Send for RefCount {}
unsafe impl Sync for RefCount {}

impl RefCount {
    const MAX: usize = 1 << 24;

    fn load(&self) -> usize {
        unsafe { self.0.as_ref() }.load(Ordering::Acquire)
    }
}

impl Clone for RefCount {
    fn clone(&self) -> Self {
        let old_size = unsafe { self.0.as_ref() }.fetch_add(1, Ordering::Relaxed);
        assert!(old_size < Self::MAX);
        RefCount(self.0)
    }
}

impl Drop for RefCount {
    fn drop(&mut self) {
        if unsafe { self.0.as_ref() }.fetch_sub(1, Ordering::Relaxed) == 1 {
            let _ = unsafe { Box::from_raw(self.0.as_ptr()) };
        }
    }
}

#[derive(Debug)]
struct LifeGuard {
    ref_count: RefCount,
    submission_index: AtomicUsize,
}

impl LifeGuard {
    fn new() -> Self {
        let bx = Box::new(AtomicUsize::new(1));
        LifeGuard {
            ref_count: RefCount(ptr::NonNull::new(Box::into_raw(bx)).unwrap()),
            submission_index: AtomicUsize::new(0),
        }
    }
}

#[derive(Clone, Debug)]
struct Stored<T> {
    value: T,
    ref_count: RefCount,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Color {
    pub const TRANSPARENT: Self = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };
    pub const BLACK: Self = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const WHITE: Self = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const RED: Self = Color {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const GREEN: Self = Color {
        r: 0.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };
    pub const BLUE: Self = Color {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Origin3d {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Origin3d {
    pub const ZERO: Self = Origin3d {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
}

impl Default for Origin3d {
    fn default() -> Self {
        Origin3d::ZERO
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Extent3d {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[repr(C)]
#[derive(Debug)]
pub struct U32Array {
    pub bytes: *const u32,
    pub length: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Features {
    pub max_bind_groups: u32,
    pub supports_texture_d24_s8: bool,
}

#[macro_export]
macro_rules! gfx_select {
    ($id:expr => $global:ident.$method:ident( $($param:expr),+ )) => {
        match $id.backend() {
            #[cfg(any(not(any(target_os = "ios", target_os = "macos")), feature = "gfx-backend-vulkan"))]
            $crate::Backend::Vulkan => $global.$method::<$crate::backend::Vulkan>( $($param),+ ),
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            $crate::Backend::Metal => $global.$method::<$crate::backend::Metal>( $($param),+ ),
            #[cfg(windows)]
            $crate::Backend::Dx12 => $global.$method::<$crate::backend::Dx12>( $($param),+ ),
            #[cfg(windows)]
            $crate::Backend::Dx11 => $global.$method::<$crate::backend::Dx11>( $($param),+ ),
            _ => unreachable!()
        }
    };
}

/// Fast hash map used internally.
type FastHashMap<K, V> =
    std::collections::HashMap<K, V, std::hash::BuildHasherDefault<fxhash::FxHasher>>;
