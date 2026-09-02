/*!
# Debugging wgpu Applications

This page describes the tricks and tools aimed at investigating issues
related to `wgpu`.

## Native API validation

`wgpu`'s goal is to guarantee that only valid workloads reach the low-level
APIs under the hood. If anything is not correct, we are going to complain
about it. You should never need to listen to the low-level API's complaints.

This is a big [work-in-progress](https://github.com/gfx-rs/wgpu/labels/area%3A%20validation),
however. So in the meantime, it's useful to hear what the low-level
validation has to say.

### Vulkan and OpenGL

On Khronos APIs we redirect all the validation output to the console. You
will see it as long as:

1. you run a debug build,
2. you have the validation layers installed (i.e. the LunarG SDK), and
3. you capture logging (with something like
   [`env_logger`](https://crates.io/crates/env_logger)).

To enable the validation layers in non-debug builds, manually turn on
validation in the [`InstanceDescriptor`] with [`InstanceFlags::VALIDATION`].
See also [Enabling Vulkan Validation Layers](crate::documentation::debugging::vulkan_validation_layers).

### D3D

#### COM

The refcount of a COM object is 9 32-bit words behind the pointer's
destination and is 4 bytes. As such, subtract 36 from the pointer's address
to get a pointer to that object's refcount. This allows you to use data
breakpoints to see all changes to the object's refcount, useful for
detecting leaks and other such things.

### Metal

You can either set `METAL_DEVICE_WRAPPER_TYPE=1` in the environment, or
launch from Xcode, where validation options are configurable in the project
menus (and enabled by default). To launch from Xcode, you can create a
project of "External Build System" type, picking `cargo` or any other
command as the build system. Then you'd need to edit the target
configuration and select the executable.

A full guide that explains how to launch from Xcode is available in
[Debugging with Xcode](crate::documentation::debugging::xcode).

#### Mac leaks

To look for leaks using "Instruments", you need to add a special entitlement
to your executable's signature. Write the following to `entitle.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "https://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
    <dict>
        <key>com.apple.security.get-task-allow</key>
        <true/>
    </dict>
</plist>
```

Then run `codesign -s - -v -f --entitlements=entitle.xml /path/to/my/app`.

([Source](https://stackoverflow.com/questions/74262518/why-required-kernel-recording-resources-are-in-use-by-another-document-error-i).)

## GPU debugging

If the validation is clean, and you are getting incorrect rendering or
computing, it's time to dive into GPU debugging. Many tools exist today:

- RenderDoc can be used on Windows and Linux to capture Vulkan, D3D, and
  OpenGL.
  - If RenderDoc is used with a compute shader without any normal rendering
    components, wrap the work in [`Device::start_graphics_debugger_capture`]
    and [`Device::stop_graphics_debugger_capture`] so RenderDoc picks up the
    pipeline.
  - Vulkan initialization may fail in RenderDoc under Wayland (see
    [issue 3889](https://github.com/gfx-rs/wgpu/issues/3889)). To work around
    it, force winit to use X11 by setting the `WAYLAND_DISPLAY` environment
    variable to an empty string for winit versions `0.28.2` and onward, or
    the `WINIT_UNIX_BACKEND` environment variable to `"x11"` for winit
    versions up to `0.29.1`. When in doubt, do both.
- PIX can be used on Windows to capture D3D12.
- Xcode has powerful Metal GPU capture support with full shader debugging.

In order to run your application from Xcode, you can create an empty project,
select your executable binary, and provide the command-line parameters in
the project settings.

### Capturing WebGPU in Chrome with PIX

<https://gist.github.com/Popov72/41f71cbf8d55f2cb8cae93f439eee347>

### Capturing WebGPU in Chrome with Xcode

<https://ced.quest/chromium-metal-debugger.html>

## CPU debugging

As `wgpu` is a Rust project, all the regular methods of debugging work. It's
convenient to launch an application from Xcode (e.g. as an "External Build
System" type project), or Visual Studio (just go to "Open Project" and
select the executable, adjusting the path and parameters).

When looking at a crash/panic, the first step is typically enabling
`RUST_BACKTRACE=1` in the environment. This will spew out the stack trace,
which helps you locate the problem.

## Rubber-duck debugging

Hop on to [`#wgpu:matrix.org`](https://matrix.to/#/#wgpu:matrix.org) and try
to explain what happens and why. Maybe we listen. Maybe you'll figure out
the answer by then ;)

## Tracing infrastructure

API tracing is built into `wgpu-core` under the `trace` feature. To record a
trace, first make sure this feature is enabled, then set the `trace` field
of the [`DeviceDescriptor`] to [`Trace::Directory`](crate::Trace) with an existing folder
path when calling [`Adapter::request_device`]. The device will then record a
trace of everything that's going on to that folder.

Once the trace is recorded, you can zip the folder and attach it to an
issue. This should allow anyone, including the developers, to reproduce the
issue on their machines, using the `player`:

```bash
cd player
cargo run --features winit -- <trace_folder_path>
```

If the program crashed or was interrupted in the midst of recording, you may
need to edit the trace folder's `trace.ron` file and add a closing `]`
bracket to the end. When that bracket is missing, the player exits with a
message like:

```text
thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value:
Error { code: Eof, position: Position { line: 96233, col: 1 } }', player/src/bin/play.rs:28:70
```

### Running in Gecko

If you encounter a problem with something that runs fine outside of Gecko
but fails to run in Gecko, there is a way to trace the API use by the
browser. Simply run Firefox Nightly (with prefs enabled for WebGPU,
obviously) under `WGPU_TRACE` pointing to a writable folder, and it will
record the API trace. Setting `MOZ_DISABLE_GPU_SANDBOX=1` might also be
required if the GPU process is sandboxed by default. This trace can be
replayed as usual, or even compared to a trace you get by running the same
application natively, to find out what the browser doesn't do right.

To replay a capture from Firefox and inspect it in RenderDoc or similar
tools, it is best not to build the player with the `winit` cargo feature.

When Firefox renders a WebGPU canvas, the result is done in a texture rather
than a window's swapchain, and that texture is then shared and put onto the
screen via other means that aren't part of the `wgpu` trace. Since there is
no window swapchain involved, RenderDoc can't guess when a frame starts and
ends, so it has to be helped by manually inserting begin/end capture
markers. When built without the `winit` feature, the player replays the
commands without rendering them into a window, and adds these markers at the
beginning and end of the trace; it doesn't when the `winit` feature is
enabled, since it then assumes the trace will eventually render into the
winit window's swapchain.

This applies to Firefox as well as any other software using `wgpu` to render
content into a texture but not to present that texture into a window.
*/

use crate::{Adapter, Device, DeviceDescriptor, InstanceDescriptor, InstanceFlags, Trace};
