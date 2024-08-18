// meant to be called with 3 vertex indices: 0, 1, 2
// draws one large triangle over the clip space like this:
// (the asterisks represent the clip space bounds)
//-1,1           1,1
// ---------------------------------
// |              *              .
// |              *           .
// |              *        .
// |              *      .
// |              *    .
// |              * .
// |***************
// |            . 1,-1
// |          .
// |       .
// |     .
// |   .
// |.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) ->  @builtin(position) vec4<f32> {
    let x = i32(vertex_index) / 2;
    let y = i32(vertex_index) & 1;
    return vec4<f32>(
        f32(x) * 4.0 - 1.0,
        1.0 - f32(y) * 4.0,
        0.0, 1.0
    );
}


@fragment
fn fs_main(@builtin(position) coord_in: vec4<f32>) -> @location(0) vec4<f32> {
    var x = 0.0;
    // Succeeds on FXC without workaround.
    switch i32(coord_in.x) {
        default {
            x = 1.0;
        }
    }
    var y = 0.0;
    // Fails on FXC without workaround.
    // (even if we adjust switch above to give different x values based on the input coord)
    switch i32(x * 30.0) {
        default {
            y = 1.0;
        }
    }
    var z = 0.0;
    // Multiple cases with a single body also fails on FXC without a workaround.
    switch 0 {
        case 0, 2, default {
            z = 1.0;
        }
    }

    var w = 0.0;
    // Succeeds on FXC without workaround.
    switch 0 {
        case 0 {
            w = 1.0;
        }
        default {
            w = 1.0;
        }
    }

    return vec4<f32>(x, y, z, w);
}
