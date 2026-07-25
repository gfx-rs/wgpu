This crate exists to allow platform and feature specific features work correctly. The features
enabled on this crate are only enabled on `target_os = "linux"`, `target_os = "android"` and
`target_os = "freebsd"` platforms. See wgpu-hal's `Cargo.toml` for more information.
