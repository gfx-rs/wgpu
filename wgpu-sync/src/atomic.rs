//! Provides items for atomic operations.

pub use core::sync::atomic::{compiler_fence, fence, Ordering};

cfg_if::cfg_if! {
    if #[cfg(target_has_atomic = "ptr")] {
        pub use core::sync::atomic::{AtomicIsize, AtomicPtr, AtomicUsize};
    } else if #[cfg(feature = "portable-atomic")] {
        pub use portable_atomic::{AtomicIsize, AtomicPtr, AtomicUsize};
    }
}

cfg_if::cfg_if! {
    if #[cfg(target_has_atomic = "64")] {
        pub use core::sync::atomic::{AtomicI64, AtomicU64};
    } else if #[cfg(feature = "portable-atomic")] {
        pub use portable_atomic::{AtomicI64, AtomicU64};
    }
}

cfg_if::cfg_if! {
    if #[cfg(target_has_atomic = "32")] {
        pub use core::sync::atomic::{AtomicI32, AtomicU32};
    } else if #[cfg(feature = "portable-atomic")] {
        pub use portable_atomic::{AtomicI32, AtomicU32};
    }
}

cfg_if::cfg_if! {
    if #[cfg(target_has_atomic = "16")] {
        pub use core::sync::atomic::{AtomicI16, AtomicU16};
    } else if #[cfg(feature = "portable-atomic")] {
        pub use portable_atomic::{AtomicI16, AtomicU16};
    }
}

cfg_if::cfg_if! {
    if #[cfg(target_has_atomic = "8")] {
        pub use core::sync::atomic::{AtomicBool, AtomicI8, AtomicU8};
    } else if #[cfg(feature = "portable-atomic")] {
        pub use portable_atomic::{AtomicBool, AtomicI8, AtomicU8};
    }
}
