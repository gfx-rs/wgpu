use wasm_bindgen_test::wasm_bindgen_test_configure;

// All files containing tests
mod common;

mod buffer;
mod buffer_copy;
mod buffer_map_after_queue_submit;
mod buffer_usages;
mod clear_texture;
mod device;
mod encoder;
mod example_wgsl;
mod instance;
mod poll;
mod queue_transfer;
mod resource_descriptor_accessor;
mod resource_error;
mod shader;
mod shader_primitive_index;
mod texture_bounds;
mod transfer;
mod vertex_indices;
mod write_texture;
mod zero_init_texture_after_discard;

wasm_bindgen_test_configure!(run_in_browser);
