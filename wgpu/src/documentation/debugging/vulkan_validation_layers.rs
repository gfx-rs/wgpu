/*!
# Enabling Vulkan Validation Layers

When developing an application it is often useful to enable Vulkan's
validation layers. They can catch errors when you are using the API
incorrectly or exceeding device limits. You can read more about validation
layers here: <https://gpuopen.com/using-the-vulkan-validation-layers/>.

Validation layers are a Vulkan-only feature, so they cannot be used with
other backends. You can restrict which backends `wgpu` uses via
[`InstanceDescriptor`] and [`Backends`].

## Installation

For validation layers to work you need to install the Vulkan SDK:

### Linux

Visit <https://packages.lunarg.com/> to install the Vulkan SDK along with
some validation layers. Click the "Latest Supported Release" button (which
looks like it's just a header, but it is in fact a button) and follow the
instructions.

### Other operating systems

TODO.

## Enabling validation layers

`wgpu` enables validation layers automatically in most cases when the
application is compiled in debug mode (that is, if the `--release` flag is
*not* passed to `cargo`), because [`InstanceFlags::DEBUG`] is set by default
in development builds. You can also enable them explicitly with
[`InstanceFlags::VALIDATION`], or by setting the environment variable
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`.

The validation layers use normal Rust logging functionality, which also
needs to be enabled. The `wgpu` examples use the
[`env_logger`](https://crates.io/crates/env_logger) crate for logging, which
means you can enable validation-layer logging by setting an environment
variable:

```bash
env RUST_LOG=trace cargo run --bin wgpu-examples cube
```

There are several logging levels (see <https://docs.rs/log/latest/log/> for
more info), with `trace` being the highest. If you choose a lower level such
as `info`, then fewer messages will be logged.

If you want to enable validation layers in your own application, make sure
it has logging configured. The
[`env_logger`](https://crates.io/crates/env_logger) crate is one possible
way to do this.
*/

use crate::{Backends, InstanceDescriptor, InstanceFlags};
