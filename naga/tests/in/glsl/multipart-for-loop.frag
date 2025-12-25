// issue #6258 https://github.com/gfx-rs/wgpu/issues/6208
# version 460

void main() {
    float a = 1.0;
    float b = 0.25;

    // tests for multiple expressions in the third part!
    for (int i = 0; i < 25; i++, b+=0.01) {
        a -= 0.02;
    }

    // a and b should be both ~0.5!
}