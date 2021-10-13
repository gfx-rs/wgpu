/*! This library safely implements WebGPU on native platforms.
 *  It is designed for integration into browsers, as well as wrapping
 *  into other language-specific user-friendly libraries.
 */

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
    // For some reason `rustc` can warn about these in const generics even
    // though they are required.
    unused_braces,
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

pub mod binding_model;
pub mod command;
mod conv;
pub mod device;
pub mod error;
pub mod hub;
pub mod id;
mod init_tracker;
pub mod instance;
pub mod pipeline;
pub mod present;
pub mod resource;
mod track;
mod validation;

pub use hal::{api, MAX_BIND_GROUPS, MAX_COLOR_TARGETS, MAX_VERTEX_BUFFERS};

use atomic::{AtomicUsize, Ordering};

use std::{borrow::Cow, os::raw::c_char, ptr, sync::atomic};

type SubmissionIndex = hal::FenceValue;
type Index = u32;
type Epoch = u32;

pub type RawString = *const c_char;
pub type Label<'a> = Option<Cow<'a, str>>;

trait LabelHelpers<'a> {
    fn borrow_option(&'a self) -> Option<&'a str>;
    fn borrow_or_default(&'a self) -> &'a str;
}
impl<'a> LabelHelpers<'a> for Label<'a> {
    fn borrow_option(&'a self) -> Option<&'a str> {
        self.as_ref().map(|cow| cow.as_ref())
    }
    fn borrow_or_default(&'a self) -> &'a str {
        self.borrow_option().unwrap_or_default()
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
        self.submission_index
            .store(submit_index as _, Ordering::Release);
        self.ref_count.is_some()
    }

    fn life_count(&self) -> SubmissionIndex {
        self.submission_index.load(Ordering::Acquire) as _
    }
}

#[derive(Clone, Debug)]
struct Stored<T> {
    value: id::Valid<T>,
    ref_count: RefCount,
}

const DOWNLEVEL_WARNING_MESSAGE: &str = "The underlying API or device in use does not \
support enough features to be a fully compliant implementation of WebGPU. A subset of the features can still be used. \
If you are running this program on native and not in a browser and wish to limit the features you use to the supported subset, \
call Adapter::downlevel_properties or Device::downlevel_properties to get a listing of the features the current \
platform supports.";
const DOWNLEVEL_ERROR_MESSAGE: &str = "This is not an invalid use of WebGPU: the underlying API or device does not \
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
            #[cfg(all(not(target_arch = "wasm32"), not(target_os = "ios"), not(target_os = "macos")))]
            wgt::Backend::Vulkan => $global.$method::<$crate::api::Vulkan>( $($param),* ),
            #[cfg(all(not(target_arch = "wasm32"), any(target_os = "ios", target_os = "macos")))]
            wgt::Backend::Metal => $global.$method::<$crate::api::Metal>( $($param),* ),
            #[cfg(all(not(target_arch = "wasm32"), windows))]
            wgt::Backend::Dx12 => $global.$method::<$crate::api::Dx12>( $($param),* ),
            //#[cfg(all(not(target_arch = "wasm32"), windows))]
            //wgt::Backend::Dx11 => $global.$method::<$crate::api::Dx11>( $($param),* ),
            #[cfg(any(
                all(unix, not(target_os = "macos"), not(target_os = "ios")),
                target_arch = "wasm32"
            ))]
            wgt::Backend::Gl => $global.$method::<$crate::api::Gles>( $($param),+ ),
            other => panic!("Unexpected backend {:?}", other),

        }
    };
}

/// Fast hash map used internally.
type FastHashMap<K, V> =
    std::collections::HashMap<K, V, std::hash::BuildHasherDefault<fxhash::FxHasher>>;
/// Fast hash set used internally.
type FastHashSet<K> = std::collections::HashSet<K, std::hash::BuildHasherDefault<fxhash::FxHasher>>;

#[inline]
pub(crate) fn get_lowest_common_denom(a: u32, b: u32) -> u32 {
    let gcd = if a >= b {
        get_greatest_common_divisor(a, b)
    } else {
        get_greatest_common_divisor(b, a)
    };
    a * b / gcd
}

#[inline]
pub(crate) fn get_greatest_common_divisor(mut a: u32, mut b: u32) -> u32 {
    assert!(a >= b);
    loop {
        let c = a % b;
        if c == 0 {
            return b;
        } else {
            a = b;
            b = c;
        }
    }
}

#[inline]
pub(crate) fn align_to(value: u32, alignment: u32) -> u32 {
    match value % alignment {
        0 => value,
        other => value - other + alignment,
    }
}

#[test]
fn test_lcd() {
    assert_eq!(get_lowest_common_denom(2, 2), 2);
    assert_eq!(get_lowest_common_denom(2, 3), 6);
    assert_eq!(get_lowest_common_denom(6, 4), 12);
}

#[test]
fn test_gcd() {
    assert_eq!(get_greatest_common_divisor(5, 1), 1);
    assert_eq!(get_greatest_common_divisor(4, 2), 2);
    assert_eq!(get_greatest_common_divisor(6, 4), 2);
    assert_eq!(get_greatest_common_divisor(7, 7), 7);
}
