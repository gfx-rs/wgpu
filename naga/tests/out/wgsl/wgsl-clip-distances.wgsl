enable clip_distances;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @builtin(clip_distances) clip_distances: array<f32, 1>,
}

struct VertexOutputSplit {
    @location(0) color: vec4<f32>,
    @builtin(clip_distances) clip_distances: array<f32, 6>,
    @builtin(position) position: vec4<f32>,
}

@vertex
fn main() -> VertexOutput {
    var out: VertexOutput;

    out.clip_distances[0] = 0.5f;
    let _e4 = out;
    return _e4;
}

@vertex
fn main_split() -> VertexOutputSplit {
    var out_1: VertexOutputSplit;
    var i: i32 = 0i;

    loop {
        let _e3 = i;
        if (_e3 < 6i) {
        } else {
            break;
        }
        {
            let _e7 = i;
            let _e9 = i;
            out_1.clip_distances[_e7] = (f32(_e9) - 2.5f);
        }
        continuing {
            let _e14 = i;
            i = (_e14 + 1i);
        }
    }
    let _e16 = out_1;
    return _e16;
}
