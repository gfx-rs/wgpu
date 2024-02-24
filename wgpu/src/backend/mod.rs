#[cfg(all(webgpu, web_sys_unstable_apis))]
mod webgpu;
#[cfg(all(webgpu, web_sys_unstable_apis))]
pub(crate) use webgpu::{get_browser_gpu_property, ContextWebGpu};

#[cfg(all(webgpu, not(web_sys_unstable_apis)))]
compile_error!(
    "webgpu feature used without web_sys_unstable_apis config:
Here are some ways to resolve this:
* If you wish to use webgpu backend, create a .cargo/config.toml in the root of the repo containing:
    [build]
    rustflags = [ \"--cfg=web_sys_unstable_apis\" ]
    rustdocflags = [ \"--cfg=web_sys_unstable_apis\" ]
* If you wish to disable webgpu backend and instead use webgl backend, change your wgpu Cargo.toml entry to:
    wgpu = { version = \"\", default-features = false, features = [\"webgl\"] }
"
);

#[cfg(wgpu_core)]
mod wgpu_core;

#[cfg(wgpu_core)]
pub(crate) use wgpu_core::ContextWgpuCore;
