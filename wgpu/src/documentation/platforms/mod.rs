/*!
Platform-Specific Guides

Notes on running `wgpu` on specific platforms and targets: the web, mobile,
ANGLE, and the requirements `wgpu` places on the Vulkan backend.
*/

pub mod android;
pub mod angle;
pub mod emscripten;
pub mod mobile_app_integration;
pub mod vulkan_requirements;
pub mod web;
