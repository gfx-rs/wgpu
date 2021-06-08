use crate::core_tests::common::init::{initialize_test, TestParameters};

#[test]
fn draw() {
    initialize_test(
        TestParameters::default().features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
        |ctx| {
            let shader_module = ctx
                .device
                .create_shader_module(&wgpu::include_wgsl!("draw.vert.wgsl"));
        },
    )
}
