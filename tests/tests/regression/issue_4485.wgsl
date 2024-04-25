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
fn fs_main() -> @location(0) vec4<f32> {
    var x = 0.0;
    loop {
        if x != 0.0 { break; }
        x = 0.5;
        // Compiled to a do-while in hlsl and glsl,
        // we want to confirm that continue applies to outer loop.
        switch 0 {
            default {
                x = 1.0;
                continue;
            }
        }
        x = 0.0;
    }
    // expect X == 1.0

    var y = 0.0;
    loop {
        if y != 0.0 { break; }
        y = 0.5;
        switch 1 {
            case 0 {
                continue;
            }
            case 1 {}
        }
        // test that loop doesn't continue after the switch when the continue case wasn't executed
        y = 1.0;
        break;
    }
    // expect y == 1.0

    var z = 0.0;
    loop {
        if z != 0.0 { break; }
        switch 0 {
            case 0 {
                z = 0.5;
            }
            case 1 {
                z = 0.5;
            }
        }
        // test that loop doesn't continue after the switch that contains no continue statements
        z = 1.0
    }
    // expect z == 1.0

    var w = 0.0;
    loop {
        if w != 0.0 { break; }
        switch 0 {
            case 0 {
                loop {
                    // continue in loop->switch->loop->switch->switch should affect inner loop
                    switch 1 {
                        case 0 {}
                        case 1 {
                            switch 0 {
                                default { continue; }
                            }
                        }
                    }
                    w = 0.5
                }
            }
            case 1 {
                w = 0.5;
            }
        }
        if w == 0.0 { w = 1.0; }
    }
    // expect w == 1.0

    return vec4<f32>(x, y, z, w);
}
