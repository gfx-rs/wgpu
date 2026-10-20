#![cfg_attr(not(doc), expect(unused_imports))]

/*!
General documentation and guides for the `wgpu` crate.

Each module here is documentation only (it contains no runtime code) and
collects concept primers, best practices, debugging tips, platform notes,
and explanations of `wgpu`'s internals that don't belong on any single API item.
*/

pub mod best_practices;
pub mod color;
pub mod debugging;
pub mod extensions;
pub mod features;
pub mod getting_started;
pub mod internals;
pub mod platforms;
pub mod shaders;
