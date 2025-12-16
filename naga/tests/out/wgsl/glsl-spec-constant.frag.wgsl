struct FragmentOutput {
    @location(0) o_color: vec4<f32>,
}

@id(0) override SPEC_CONST_BOOL: bool = true;
@id(1) override SPEC_CONST_INT: i32 = 42i;
@id(2) override SPEC_CONST_UINT: u32 = 10u;
@id(3) override SPEC_CONST_FLOAT: f32 = 3.14f;

var<private> o_color: vec4<f32>;

fn main_1() {
    var result: f32 = 0f;

    if SPEC_CONST_BOOL {
        {
            let _e7 = result;
            result = (_e7 + f32(SPEC_CONST_INT));
        }
    }
    let _e10 = result;
    result = (_e10 + (f32(SPEC_CONST_UINT) * SPEC_CONST_FLOAT));
    let _e14 = result;
    o_color = vec4<f32>(_e14, 0f, 0f, 1f);
    return;
}

@fragment 
fn main() -> FragmentOutput {
    main_1();
    let _e1 = o_color;
    return FragmentOutput(_e1);
}
