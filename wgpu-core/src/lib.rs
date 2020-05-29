/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_qualifications
)]

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
mod conv;
pub mod device;
pub mod hub;
pub mod id;
pub mod instance;
pub mod pipeline;
pub mod power;
pub mod resource;
pub mod swap_chain;
mod track;

pub use hal::pso::read_spirv;

#[cfg(test)]
use loom::sync::atomic;
#[cfg(not(test))]
use std::sync::atomic;

use atomic::{AtomicUsize, Ordering};

use std::{os::raw::c_char, ptr};

type SubmissionIndex = usize;
type Index = u32;
type Epoch = u32;

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

    /// This works like `std::mem::drop`, except that it returns a boolean which is true if and only
    /// if we deallocated the underlying memory, i.e. if this was the last clone of this `RefCount`
    /// to be dropped. This is useful for loom testing because it allows us to verify that we
    /// deallocated the underlying memory exactly once.
    #[cfg(test)]
    fn rich_drop_outer(self) -> bool {
        unsafe { std::mem::ManuallyDrop::new(self).rich_drop_inner() }
    }

    /// This function exists to allow `Self::rich_drop_outer` and `Drop::drop` to share the same
    /// logic. To use this safely from outside of `Drop::drop`, the calling function must move
    /// `Self` into a `ManuallyDrop`.
    unsafe fn rich_drop_inner(&mut self) -> bool {
        if self.0.as_ref().fetch_sub(1, Ordering::Relaxed) == 1 {
            let _ = Box::from_raw(self.0.as_ptr());
            true
        } else {
            false
        }
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
        unsafe {
            self.rich_drop_inner();
        }
    }
}

#[cfg(test)]
#[test]
fn loom() {
    loom::model(move || {
        let bx = Box::new(AtomicUsize::new(1));
        let ref_count_main = ptr::NonNull::new(Box::into_raw(bx)).map(RefCount).unwrap();
        let ref_count_spawned = ref_count_main.clone();

        let join_handle = loom::thread::spawn(move || {
            let _ = ref_count_spawned.clone();
            ref_count_spawned.rich_drop_outer()
        });

        let dropped_in_main = ref_count_main.rich_drop_outer();
        let dropped_in_spawned = join_handle.join().unwrap();
        assert_ne!(
            dropped_in_main, dropped_in_spawned,
            "must drop exactly once"
        );
    });
}

#[derive(Debug)]
struct LifeGuard {
    ref_count: Option<RefCount>,
    submission_index: AtomicUsize,
}

impl LifeGuard {
    fn new() -> Self {
        let bx = Box::new(AtomicUsize::new(1));
        LifeGuard {
            ref_count: ptr::NonNull::new(Box::into_raw(bx)).map(RefCount),
            submission_index: AtomicUsize::new(0),
        }
    }

    fn add_ref(&self) -> RefCount {
        self.ref_count.clone().unwrap()
    }

    /// Returns `true` if the resource is still needed by the user.
    fn use_at(&self, submit_index: SubmissionIndex) -> bool {
        self.submission_index.store(submit_index, Ordering::Release);
        self.ref_count.is_some()
    }
}

#[derive(Clone, Debug)]
struct Stored<T> {
    value: T,
    ref_count: RefCount,
}

#[repr(C)]
#[derive(Debug)]
pub struct U32Array {
    pub bytes: *const u32,
    pub length: usize,
}

#[derive(Clone, Copy, Debug)]
struct PrivateFeatures {
    pub supports_texture_d24_s8: bool,
}

#[macro_export]
macro_rules! gfx_select {
    ($id:expr => $global:ident.$method:ident( $($param:expr),+ )) => {
        match $id.backend() {
            #[cfg(any(not(any(target_os = "ios", target_os = "macos")), feature = "gfx-backend-vulkan"))]
            wgt::Backend::Vulkan => $global.$method::<$crate::backend::Vulkan>( $($param),+ ),
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            wgt::Backend::Metal => $global.$method::<$crate::backend::Metal>( $($param),+ ),
            #[cfg(windows)]
            wgt::Backend::Dx12 => $global.$method::<$crate::backend::Dx12>( $($param),+ ),
            #[cfg(windows)]
            wgt::Backend::Dx11 => $global.$method::<$crate::backend::Dx11>( $($param),+ ),
            _ => unreachable!()
        }
    };
}

/// Fast hash map used internally.
type FastHashMap<K, V> =
    std::collections::HashMap<K, V, std::hash::BuildHasherDefault<fxhash::FxHasher>>;
