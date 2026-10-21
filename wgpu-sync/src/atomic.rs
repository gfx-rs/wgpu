//! Provides items for atomic operations.

pub use core::sync::atomic::{compiler_fence, fence, Ordering};

cfg_select! {
    target_has_atomic = "ptr" => {
        pub use core::sync::atomic::{AtomicIsize, AtomicPtr, AtomicUsize};
    }
    feature = "portable-atomic" => {
        pub use portable_atomic::{AtomicIsize, AtomicPtr, AtomicUsize};
    }
}

cfg_select! {
    target_has_atomic = "64" => {
        pub use core::sync::atomic::{AtomicI64, AtomicU64};
    }
    feature = "portable-atomic" => {
        pub use portable_atomic::{AtomicI64, AtomicU64};
    }
}

cfg_select! {
    target_has_atomic = "32" => {
        pub use core::sync::atomic::{AtomicI32, AtomicU32};
    }
    feature = "portable-atomic" => {
        pub use portable_atomic::{AtomicI32, AtomicU32};
    }
}

cfg_select! {
    target_has_atomic = "16" => {
        pub use core::sync::atomic::{AtomicI16, AtomicU16};
    }
    feature = "portable-atomic" => {
        pub use portable_atomic::{AtomicI16, AtomicU16};
    }
}

cfg_select! {
    target_has_atomic = "8" => {
        pub use core::sync::atomic::{AtomicBool, AtomicI8, AtomicU8};
    }
    feature = "portable-atomic" => {
        pub use portable_atomic::{AtomicBool, AtomicI8, AtomicU8};
    }
}
