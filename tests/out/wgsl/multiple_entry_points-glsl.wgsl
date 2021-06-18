struct VertexOutput {
    [[builtin(position)]] member: vec4<f32>;
};

struct FragmentOutput {
    [[location(0), interpolate(perspective)]] o_color: vec4<f32>;
};

var<private> gl_Position: vec4<f32>;
var<private> o_color: vec4<f32>;
var<private> gl_GlobalInvocationID: vec3<u32>;

fn vert_main1() {
    gl_Position = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return;
}

fn frag_main1() {
    o_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return;
}

fn comp_main1() {
    let _e3: vec3<u32> = gl_GlobalInvocationID;
    if ((_e3.x > u32(1))) {
        {
            return;
        }
    } else {
        return;
    }
}

[[stage(vertex)]]
fn vert_main() -> VertexOutput {
    vert_main1();
    let _e1: vec4<f32> = gl_Position;
    return VertexOutput(_e1);
}

[[stage(fragment)]]
fn frag_main() -> FragmentOutput {
    frag_main1();
    let _e1: vec4<f32> = o_color;
    return FragmentOutput(_e1);
}

[[stage(compute), workgroup_size(1, 1, 1)]]
fn comp_main([[builtin(global_invocation_id)]] param: vec3<u32>) {
    gl_GlobalInvocationID = param;
    comp_main1();
    return;
}
