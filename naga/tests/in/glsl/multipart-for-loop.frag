// issue #6208 https://github.com/gfx-rs/wgpu/issues/6208
# version 460

void main() {
    float a = 1.0;
    float b = 0.25;
    float c = 1.5;
    int i = 20;

    // tests for multiple expressions in first part (if it's a expression, not declaration)!
    // also the third part!
    for (i = 0, c-=1.0; i < 25; i++, b+=0.01) {
        a -= 0.02;
    }

    // a, b and c should be all ~0.5!
}