/*!
# Debugging Performance Issues

If you're encountering performance problems, here's a checklist:

- Are you building with `--release`? (Don't laugh!) Development-profile
  builds trade overhead for diagnostic ability:
  - [`InstanceFlags::VALIDATION`] is set by default in development builds,
    which enables platform-level validation, like the Vulkan validation layers.
  - The Rust compiler doesn't optimize the generated machine code.
  - Development builds have more assertions enabled.
*/

use crate::InstanceFlags;
