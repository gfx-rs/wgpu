/*! This library safely implements WebGPU on native platforms.
 *  It is designed for integration into browsers, as well as wrapping
 *  into other language-specific user-friendly libraries.
 */

// When we have no backends, we end up with a lot of dead or otherwise unreachable code.
#![cfg_attr(
    all(
        not(all(feature = "vulkan", not(target_arch = "wasm32"))),
        not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))),
        not(all(feature = "dx12", windows)),
        not(all(feature = "dx11", windows)),
        not(feature = "gles"),
    ),
    allow(unused, clippy::let_and_return)
)]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]
#![allow(
    // It is much clearer to assert negative conditions with eq! false
    clippy::bool_assert_comparison,
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
    // Needless updates are more scaleable, easier to play with features.
    clippy::needless_update,
    // Need many arguments for some core functions to be able to re-use code in many situations.
    clippy::too_many_arguments,
    // For some reason `rustc` can warn about these in const generics even
    // though they are required.
    unused_braces,
    // Clashes with clippy::pattern_type_mismatch
    clippy::needless_borrowed_reference,
)]
#![warn(
    trivial_casts,
    trivial_numeric_casts,
    unsafe_op_in_unsafe_fn,
    unused_extern_crates,
    unused_qualifications,
    // We don't match on a reference, unless required.
    clippy::pattern_type_mismatch,
)]

pub mod binding_model;
pub mod command;
mod conv;
pub mod device;
pub mod error;
pub mod global;
pub mod hal_api;
pub mod hub;
pub mod id;
pub mod identity;
mod init_tracker;
pub mod instance;
pub mod pipeline;
pub mod present;
pub mod registry;
pub mod resource;
pub mod storage;
mod track;
mod validation;

pub use hal::{api, MAX_BIND_GROUPS, MAX_COLOR_ATTACHMENTS, MAX_VERTEX_BUFFERS};

use atomic::{AtomicUsize, Ordering};

use std::{borrow::Cow, os::raw::c_char, ptr, sync::atomic};

/// The index of a queue submission.
///
/// These are the values stored in `Device::fence`.
type SubmissionIndex = hal::FenceValue;

type Index = u32;
type Epoch = u32;

pub type RawString = *const c_char;
pub type Label<'a> = Option<Cow<'a, str>>;

trait LabelHelpers<'a> {
    fn borrow_option(&'a self) -> Option<&'a str>;
    fn to_hal(&'a self, flags: wgt::InstanceFlags) -> Option<&'a str>;
    fn borrow_or_default(&'a self) -> &'a str;
}
impl<'a> LabelHelpers<'a> for Label<'a> {
    fn borrow_option(&'a self) -> Option<&'a str> {
        self.as_ref().map(|cow| cow.as_ref())
    }
    fn to_hal(&'a self, flags: wgt::InstanceFlags) -> Option<&'a str> {
        if flags.contains(wgt::InstanceFlags::DISCARD_HAL_LABELS) {
            return None;
        }

        self.as_ref().map(|cow| cow.as_ref())
    }
    fn borrow_or_default(&'a self) -> &'a str {
        self.borrow_option().unwrap_or_default()
    }
}

pub fn hal_label(opt: Option<&str>, flags: wgt::InstanceFlags) -> Option<&str> {
    if flags.contains(wgt::InstanceFlags::DISCARD_HAL_LABELS) {
        return None;
    }

    opt
}

/// Reference count object that is 1:1 with each reference.
///
/// All the clones of a given `RefCount` point to the same
/// heap-allocated atomic reference count. When the count drops to
/// zero, only the count is freed. No other automatic cleanup takes
/// place; this is just a reference count, not a smart pointer.
///
/// `RefCount` values are created only by [`LifeGuard::new`] and by
/// `Clone`, so every `RefCount` is implicitly tied to some
/// [`LifeGuard`].
#[derive(Debug)]
struct RefCount(ptr::NonNull<AtomicUsize>);

unsafe impl Send for RefCount {}
unsafe impl Sync for RefCount {}

impl RefCount {
    const MAX: usize = 1 << 24;

    /// Construct a new `RefCount`, with an initial count of 1.
    fn new() -> RefCount {
        let bx = Box::new(AtomicUsize::new(1));
        Self(unsafe { ptr::NonNull::new_unchecked(Box::into_raw(bx)) })
    }

    fn load(&self) -> usize {
        unsafe { self.0.as_ref() }.load(Ordering::Acquire)
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
            if self.0.as_ref().fetch_sub(1, Ordering::AcqRel) == 1 {
                drop(Box::from_raw(self.0.as_ptr()));
            }
        }
    }
}

/// Reference count object that tracks multiple references.
/// Unlike `RefCount`, it's manually inc()/dec() called.
#[derive(Debug)]
struct MultiRefCount(AtomicUsize);

impl MultiRefCount {
    fn new() -> Self {
        Self(AtomicUsize::new(1))
    }

    fn inc(&self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }

    fn dec_and_check_empty(&self) -> bool {
        self.0.fetch_sub(1, Ordering::AcqRel) == 1
    }
}

/// Information needed to decide when it's safe to free some wgpu-core
/// resource.
///
/// Each type representing a `wgpu-core` resource, like [`Device`],
/// [`Buffer`], etc., contains a `LifeGuard` which indicates whether
/// it is safe to free.
///
/// A resource may need to be retained for any of several reasons:
///
/// - The user may hold a reference to it (via a `wgpu::Buffer`, say).
///
/// - Other resources may depend on it (a texture view's backing
///   texture, for example).
///
/// - It may be used by commands sent to the GPU that have not yet
///   finished execution.
///
/// [`Device`]: device::Device
/// [`Buffer`]: resource::Buffer
#[derive(Debug)]
pub struct LifeGuard {
    /// `RefCount` for the user's reference to this resource.
    ///
    /// When the user first creates a `wgpu-core` resource, this `RefCount` is
    /// created along with the resource's `LifeGuard`. When the user drops the
    /// resource, we swap this out for `None`. Note that the resource may
    /// still be held alive by other resources.
    ///
    /// Any `Stored<T>` value holds a clone of this `RefCount` along with the id
    /// of a `T` resource.
    ref_count: Option<RefCount>,

    /// The index of the last queue submission in which the resource
    /// was used.
    ///
    /// Each queue submission is fenced and assigned an index number
    /// sequentially. Thus, when a queue submission completes, we know any
    /// resources used in that submission and any lower-numbered submissions are
    /// no longer in use by the GPU.
    submission_index: AtomicUsize,

    /// The `label` from the descriptor used to create the resource.
    #[cfg(debug_assertions)]
    pub(crate) label: String,
}

impl LifeGuard {
    #[allow(unused_variables)]
    fn new(label: &str) -> Self {
        Self {
            ref_count: Some(RefCount::new()),
            submission_index: AtomicUsize::new(0),
            #[cfg(debug_assertions)]
            label: label.to_string(),
        }
    }

    fn add_ref(&self) -> RefCount {
        self.ref_count.clone().unwrap()
    }

    /// Record that this resource will be used by the queue submission with the
    /// given index.
    ///
    /// Returns `true` if the resource is still held by the user.
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

// #[cfg] attributes in exported macros are interesting!
//
// The #[cfg] conditions in a macro's expansion are evaluated using the
// configuration options (features, target architecture and os, etc.) in force
// where the macro is *used*, not where it is *defined*. That is, if crate A
// defines a macro like this:
//
//     #[macro_export]
//     macro_rules! if_bleep {
//         { } => {
//             #[cfg(feature = "bleep")]
//             bleep();
//         }
//     }
//
// and then crate B uses it like this:
//
//     fn f() {
//         if_bleep! { }
//     }
//
// then it is crate B's `"bleep"` feature, not crate A's, that determines
// whether the macro expands to a function call or an empty statement. The
// entire configuration predicate is evaluated in the use's context, not the
// definition's.
//
// Since `wgpu-core` selects back ends using features, we need to make sure the
// arms of the `gfx_select!` macro are pruned according to `wgpu-core`'s
// features, not those of whatever crate happens to be using `gfx_select!`. This
// means we can't use `#[cfg]` attributes in `gfx_select!`s definition itself.
// Instead, for each backend, `gfx_select!` must use a macro whose definition is
// selected by `#[cfg]` in `wgpu-core`. The configuration predicate is still
// evaluated when the macro is used; we've just moved the `#[cfg]` into a macro
// used by `wgpu-core` itself.

/// Define an exported macro named `$public` that expands to an expression if
/// the feature `$feature` is enabled, or to a panic otherwise.
///
/// This is used in the definition of `gfx_select!`, to dispatch the
/// call to the appropriate backend, but panic if that backend was not
/// compiled in.
///
/// For a call like this:
///
/// ```ignore
/// define_backend_caller! { name, private, "feature" if cfg_condition }
/// ```
///
/// define a macro `name`, used like this:
///
/// ```ignore
/// name!(expr)
/// ```
///
/// that expands to `expr` if `#[cfg(cfg_condition)]` is enabled, or a
/// panic otherwise. The panic message complains that `"feature"` is
/// not enabled.
///
/// Because of odd technical limitations on exporting macros expanded
/// by other macros, you must supply both a public-facing name for the
/// macro and a private name, `$private`, which is never used
/// outside this macro. For details:
/// <https://github.com/rust-lang/rust/pull/52234#issuecomment-976702997>
macro_rules! define_backend_caller {
    { $public:ident, $private:ident, $feature:literal if $cfg:meta } => {
        #[cfg($cfg)]
        #[macro_export]
        macro_rules! $private {
            ( $call:expr ) => ( $call )
        }

        #[cfg(not($cfg))]
        #[macro_export]
        macro_rules! $private {
            ( $call:expr ) => (
                panic!("Identifier refers to disabled backend feature {:?}", $feature)
            )
        }

        // See note about rust-lang#52234 above.
        #[doc(hidden)] pub use $private as $public;
    }
}

// Define a macro for each `gfx_select!` match arm. For example,
//
//     gfx_if_vulkan!(expr)
//
// expands to `expr` if the `"vulkan"` feature is enabled, or to a panic
// otherwise.
define_backend_caller! { gfx_if_vulkan, gfx_if_vulkan_hidden, "vulkan" if all(feature = "vulkan", not(target_arch = "wasm32")) }
define_backend_caller! { gfx_if_metal, gfx_if_metal_hidden, "metal" if all(feature = "metal", any(target_os = "macos", target_os = "ios")) }
define_backend_caller! { gfx_if_dx12, gfx_if_dx12_hidden, "dx12" if all(feature = "dx12", windows) }
define_backend_caller! { gfx_if_dx11, gfx_if_dx11_hidden, "dx11" if all(feature = "dx11", windows) }
define_backend_caller! { gfx_if_gles, gfx_if_gles_hidden, "gles" if feature = "gles" }

/// Dispatch on an [`Id`]'s backend to a backend-generic method.
///
/// Uses of this macro have the form:
///
/// ```ignore
///
///     gfx_select!(id => value.method(args...))
///
/// ```
///
/// This expands to an expression that calls `value.method::<A>(args...)` for
/// the backend `A` selected by `id`. The expansion matches on `id.backend()`,
/// with an arm for each backend type in [`wgpu_types::Backend`] which calls the
/// specialization of `method` for the given backend. This allows resource
/// identifiers to select backends dynamically, even though many `wgpu_core`
/// methods are compiled and optimized for a specific back end.
///
/// This macro is typically used to call methods on [`wgpu_core::global::Global`],
/// many of which take a single `hal::Api` type parameter. For example, to
/// create a new buffer on the device indicated by `device_id`, one would say:
///
/// ```ignore
/// gfx_select!(device_id => global.device_create_buffer(device_id, ...))
/// ```
///
/// where the `device_create_buffer` method is defined like this:
///
/// ```ignore
/// impl<...> Global<...> {
///    pub fn device_create_buffer<A: hal::Api>(&self, ...) -> ...
///    { ... }
/// }
/// ```
///
/// That `gfx_select!` call uses `device_id`'s backend to select the right
/// backend type `A` for a call to `Global::device_create_buffer<A>`.
///
/// However, there's nothing about this macro that is specific to `global::Global`.
/// For example, Firefox's embedding of `wgpu_core` defines its own types with
/// methods that take `hal::Api` type parameters. Firefox uses `gfx_select!` to
/// dynamically dispatch to the right specialization based on the resource's id.
///
/// [`wgpu_types::Backend`]: wgt::Backend
/// [`wgpu_core::global::Global`]: crate::global::Global
/// [`Id`]: id::Id
#[macro_export]
macro_rules! gfx_select {
    ($id:expr => $global:ident.$method:ident( $($param:expr),* )) => {
        match $id.backend() {
            wgt::Backend::Vulkan => $crate::gfx_if_vulkan!($global.$method::<$crate::api::Vulkan>( $($param),* )),
            wgt::Backend::Metal => $crate::gfx_if_metal!($global.$method::<$crate::api::Metal>( $($param),* )),
            wgt::Backend::Dx12 => $crate::gfx_if_dx12!($global.$method::<$crate::api::Dx12>( $($param),* )),
            wgt::Backend::Dx11 => $crate::gfx_if_dx11!($global.$method::<$crate::api::Dx11>( $($param),* )),
            wgt::Backend::Gl => $crate::gfx_if_gles!($global.$method::<$crate::api::Gles>( $($param),+ )),
            other => panic!("Unexpected backend {:?}", other),
        }
    };
}

/// Fast hash map used internally.
type FastHashMap<K, V> =
    std::collections::HashMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
/// Fast hash set used internally.
type FastHashSet<K> =
    std::collections::HashSet<K, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

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
