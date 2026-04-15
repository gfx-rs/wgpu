#![cfg_attr(docsrs, feature(doc_cfg))]
#![no_std]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod glsl;
pub mod hlsl;
pub mod msl;
pub mod spv;

use alloc::string::String;

/// Create a Markdown link definition referring to the `wgpu` crate.
///
/// This macro should be used inside a `#[doc = ...]` attribute.
/// The two arguments should be string literals or macros that expand to string literals.
/// If the module in which the item using this macro is located is not the crate root,
/// use the `../` syntax.
///
/// We cannot simply use rustdoc links to `wgpu` because it is one of our dependents.
/// This link adapts to work in locally generated documentation (`cargo doc`) by default,
/// and work with `docs.rs` URL structure when building for `docs.rs`.
///
/// Note: This macro cannot be used outside this crate, because `cfg(docsrs)` will not apply.
#[cfg(not(docsrs))]
#[macro_export]
macro_rules! link_to_wgpu_docs {
    ([$reference:expr]: $url_path:expr) => {
        concat!("[", $reference, "]: ../wgpu/", $url_path)
    };

    (../ [$reference:expr]: $url_path:expr) => {
        concat!("[", $reference, "]: ../../wgpu/", $url_path)
    };
}
#[cfg(docsrs)]
#[macro_export]
macro_rules! link_to_wgpu_docs {
    ($(../)? [$reference:expr]: $url_path:expr) => {
        concat!(
            "[",
            $reference,
            // URL path will have a base URL of https://docs.rs/
            "]: /wgpu/",
            // The version of wgpu-types is not necessarily the same as the version of wgpu
            // if a patch release of either has been published, so we cannot use the full version
            // number. docs.rs will interpret this single number as a Cargo-style version
            // requirement and redirect to the latest compatible version.
            //
            // This technique would break if `wgpu` and `wgpu-types` ever switch to having distinct
            // major version numbering. An alternative would be to hardcode the corresponding `wgpu`
            // version, but that would give us another thing to forget to update.
            env!("CARGO_PKG_VERSION_MAJOR"),
            "/wgpu/",
            $url_path
        )
    };
}

/// Create a Markdown link definition referring to an item in the `wgpu` crate.
///
/// This macro should be used inside a `#[doc = ...]` attribute.
/// See [`link_to_wgpu_docs`] for more details.
#[macro_export]
macro_rules! link_to_wgpu_item {
    ($kind:ident $name:ident) => {
        $crate::link_to_wgpu_docs!(
            [concat!("`", stringify!($name), "`")]: concat!(stringify!($kind), ".", stringify!($name), ".html")
        )
    };
}

/// Create a Markdown link definition referring to the `wgpu_core` crate.
///
/// This macro should be used inside a `#[doc = ...]` attribute.
/// See [`link_to_wgpu_docs`] for more details.
#[cfg(not(docsrs))]
#[macro_export]
macro_rules! link_to_wgc_docs {
    ([$reference:expr]: $url_path:expr) => {
        concat!("[", $reference, "]: ../wgpu_core/", $url_path)
    };

    (../ [$reference:expr]: $url_path:expr) => {
        concat!("[", $reference, "]: ../../wgpu_core/", $url_path)
    };
}
#[cfg(docsrs)]
#[macro_export]
macro_rules! link_to_wgc_docs {
    ($(../)? [$reference:expr]: $url_path:expr) => {
        concat!(
            "[",
            $reference,
            // URL path will have a base URL of https://docs.rs/
            "]: /wgpu_core/",
            // The version of wgpu-types is not necessarily the same as the version of wgpu_core
            // if a patch release of either has been published, so we cannot use the full version
            // number. docs.rs will interpret this single number as a Cargo-style version
            // requirement and redirect to the latest compatible version.
            //
            // This technique would break if `wgpu_core` and `wgpu-types` ever switch to having
            // distinct major version numbering. An alternative would be to hardcode the
            // corresponding `wgpu_core` version, but that would give us another thing to forget
            // to update.
            env!("CARGO_PKG_VERSION_MAJOR"),
            "/wgpu_core/",
            $url_path
        )
    };
}
