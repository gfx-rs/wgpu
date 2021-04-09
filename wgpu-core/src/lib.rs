/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#![allow(
    // We use loops for getting early-out of scope without closures.
    clippy::never_loop,
    // We don't use syntax sugar where it's not necessary.
    clippy::match_like_matches_macro,
    // Redundant matching is more explicit.
    clippy::redundant_pattern_matching,
    // Explicit lifetimes are often easier to reason about.
    clippy::needless_lifetimes,
    // No need for defaults in the internal types.
    clippy::new_without_default,
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_qualifications,
    // We don't match on a reference, unless required.
    clippy::pattern_type_mismatch,
)]

#[macro_use]
mod macros;

pub mod backend {
    pub use gfx_backend_empty::Backend as Empty;

    #[cfg(dx11)]
    pub use gfx_backend_dx11::Backend as Dx11;
    #[cfg(dx12)]
    pub use gfx_backend_dx12::Backend as Dx12;
    #[cfg(gl)]
    pub use gfx_backend_gl::Backend as Gl;
    #[cfg(metal)]
    pub use gfx_backend_metal::Backend as Metal;
    #[cfg(vulkan)]
    pub use gfx_backend_vulkan::Backend as Vulkan;
}

pub mod binding_model;
pub mod command;
mod conv;
pub mod device;
pub mod hub;
pub mod id;
pub mod instance;
mod memory_init_tracker;
pub mod pipeline;
pub mod resource;
pub mod swap_chain;
mod track;
mod validation;

#[cfg(test)]
use loom::sync::atomic;
#[cfg(not(test))]
use std::sync::atomic;

use atomic::{AtomicUsize, Ordering};

use std::{borrow::Cow, os::raw::c_char, ptr};

pub const MAX_BIND_GROUPS: usize = 8;

type SubmissionIndex = usize;
type Index = u32;
type Epoch = u32;

pub type RawString = *const c_char;
pub type Label<'a> = Option<Cow<'a, str>>;

trait LabelHelpers<'a> {
    fn to_string_or_default(&'a self) -> String;
    fn borrow_or_default(&'a self) -> &'a str;
}
impl<'a> LabelHelpers<'a> for Label<'a> {
    fn borrow_or_default(&'a self) -> &'a str {
        self.as_ref().map(|cow| cow.as_ref()).unwrap_or("")
    }
    fn to_string_or_default(&'a self) -> String {
        self.as_ref()
            .map(|cow| cow.as_ref())
            .unwrap_or("")
            .to_string()
    }
}

/// Reference count object that is 1:1 with each reference.
#[derive(Debug)]
struct RefCount(ptr::NonNull<AtomicUsize>);

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
        if self.0.as_ref().fetch_sub(1, Ordering::AcqRel) == 1 {
            let _ = Box::from_raw(self.0.as_ptr());
            true
        } else {
            false
        }
    }
}

impl Clone for RefCount {
    fn clone(&self) -> Self {
        let old_size = unsafe { self.0.as_ref() }.fetch_add(1, Ordering::AcqRel);
        assert!(old_size < Self::MAX);
        Self(self.0)
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

/// Reference count object that tracks multiple references.
/// Unlike `RefCount`, it's manually inc()/dec() called.
#[derive(Debug)]
struct MultiRefCount(ptr::NonNull<AtomicUsize>);

unsafe impl Send for MultiRefCount {}
unsafe impl Sync for MultiRefCount {}

impl MultiRefCount {
    fn new() -> Self {
        let bx = Box::new(AtomicUsize::new(1));
        let ptr = Box::into_raw(bx);
        Self(unsafe { ptr::NonNull::new_unchecked(ptr) })
    }

    fn inc(&self) {
        unsafe { self.0.as_ref() }.fetch_add(1, Ordering::AcqRel);
    }

    fn dec_and_check_empty(&self) -> bool {
        unsafe { self.0.as_ref() }.fetch_sub(1, Ordering::AcqRel) == 1
    }
}

impl Drop for MultiRefCount {
    fn drop(&mut self) {
        let _ = unsafe { Box::from_raw(self.0.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct LifeGuard {
    ref_count: Option<RefCount>,
    submission_index: AtomicUsize,
    #[cfg(debug_assertions)]
    pub(crate) label: String,
}

impl LifeGuard {
    #[allow(unused_variables)]
    fn new(label: &str) -> Self {
        let bx = Box::new(AtomicUsize::new(1));
        Self {
            ref_count: ptr::NonNull::new(Box::into_raw(bx)).map(RefCount),
            submission_index: AtomicUsize::new(0),
            #[cfg(debug_assertions)]
            label: label.to_string(),
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
    value: id::Valid<T>,
    ref_count: RefCount,
}

#[derive(Clone, Copy, Debug)]
struct PrivateFeatures {
    anisotropic_filtering: bool,
    texture_d24: bool,
    texture_d24_s8: bool,
}

const DOWNLEVEL_WARNING_MESSAGE: &str = "The underlying API or device in use does not \
support enough features to be a fully compliant implementation of WebGPU. A subset of the features can still be used. \
If you are running this program on native and not in a browser and wish to limit the features you use to the supported subset, \
call Adapter::downlevel_properties or Device::downlevel_properties to get a listing of the features the current \
platform supports.";
const DOWNLEVEL_ERROR_WARNING_MESSAGE: &str = "This is not an invalid use of WebGPU: the underlying API or device does not \
support enough features to be a fully compliant implementation. A subset of the features can still be used. \
If you are running this program on native and not in a browser and wish to work around this issue, call \
Adapter::downlevel_properties or Device::downlevel_properties to get a listing of the features the current \
platform supports.";

#[macro_export]
macro_rules! gfx_select {
    ($id:expr => $global:ident.$method:ident( $($param:expr),* )) => {
        // Note: For some reason the cfg aliases defined in build.rs don't succesfully apply in this
        // macro so we must specify their equivalents manually
        match $id.backend() {
            #[cfg(all(not(target_arch = "wasm32"), any(not(any(target_os = "ios", target_os = "macos")), feature = "gfx-backend-vulkan")))]
            wgt::Backend::Vulkan => $global.$method::<$crate::backend::Vulkan>( $($param),* ),
            #[cfg(all(not(target_arch = "wasm32"), any(target_os = "ios", target_os = "macos")))]
            wgt::Backend::Metal => $global.$method::<$crate::backend::Metal>( $($param),* ),
            #[cfg(all(not(target_arch = "wasm32"), windows))]
            wgt::Backend::Dx12 => $global.$method::<$crate::backend::Dx12>( $($param),* ),
            #[cfg(all(not(target_arch = "wasm32"), windows))]
            wgt::Backend::Dx11 => $global.$method::<$crate::backend::Dx11>( $($param),* ),
            //#[cfg(any(target_arch = "wasm32", all(unix, not(any(target_os = "ios", target_os = "macos")))))]
            //wgt::Backend::Gl => $global.$method::<$crate::backend::Gl>( $($param),+ ),
            other => panic!("Unexpected backend {:?}", other),
        }
    };
}

/// Fast hash map used internally.
type FastHashMap<K, V> =
    std::collections::HashMap<K, V, std::hash::BuildHasherDefault<fxhash::FxHasher>>;
/// Fast hash set used internally.
type FastHashSet<K> = std::collections::HashSet<K, std::hash::BuildHasherDefault<fxhash::FxHasher>>;

#[test]
fn test_default_limits() {
    let limits = wgt::Limits::default();
    assert!(limits.max_bind_groups <= MAX_BIND_GROUPS as u32);
}
