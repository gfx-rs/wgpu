// Textual backends (MSL, HLSL, GLSL) emit a loop's `continuing` block above
// the loop body, so the temporaries the body defines have to be declared
// outside the loop for the `continuing` block to be able to name them.

fn f(x: i32) -> i32 { return x * 2; }

// The simplest case: `cmp` is a named expression the body defines.
fn named() {
    var a = 0;
    loop {
        let cmp = a < 8;
        if cmp {
            a += 1;
        }
        continuing {
            break if cmp;
        }
    }
}

// `v` is a `CallResult`: it is defined by the `Call` statement rather than by
// an `Emit` range, and it has no inline form at all.
fn call_result() {
    var a = 0;
    loop {
        let v = f(a);
        a += 1;
        continuing {
            break if v > 10;
        }
    }
}

// `p` is named, but pointer-typed, so it never gets a temporary of its own.
// The continuing block re-evaluates it, and so needs the unnamed load of `i`
// that the body captured.
fn through_pointer() {
    var arr = array<i32, 4>(0, 1, 2, 3);
    var i = 0;
    loop {
        let p = &arr[i];
        i += 1;
        continuing {
            break if *p > 2;
        }
    }
}

// Only the innermost loop's body is in scope in its own continuing block, so
// each loop hoists its own.
fn nested() {
    var a = 0;
    loop {
        let outer = a < 8;
        loop {
            let inner = a < 4;
            a += 1;
            continuing {
                break if inner;
            }
        }
        continuing {
            break if outer;
        }
    }
}

// Backends don't do dead code elimination, so they write out a definition for
// `dead` even though nothing reads it, and that definition references `v` and
// `w`. What the continuing block references is therefore wider than what it
// needs the value of: hoisting only what a liveness analysis reports would
// leave behind `v`, which has no inline form at all, and re-evaluate `w`
// against the `a` the body left behind. The definition sits inside an `if` so
// that finding it also means walking into a nested block.
fn dead_use() {
    var a = 0;
    loop {
        let v = f(a);
        let w = a * 2;
        a += 1;
        continuing {
            if a > 5 {
                let dead = v + w;
            }
            break if a > 10;
        }
    }
}

// Under the `Restrict` image load policy, `put_cache_restricted_level` gives
// the load an extra `clamped_lod` temporary, declared next to it in the body.
// Re-evaluating the load in `continuing` named that temporary above its own
// declaration, which isn't even valid MSL. Hoisting the load means
// `continuing` never re-evaluates it.
@group(0) @binding(0) var tex: texture_2d<f32>;

fn restricted_image_load() {
    var i = 0;
    loop {
        let val = textureLoad(tex, vec2i(i, 0), 0).x;
        i += 1;
        continuing {
            break if val > 0.0;
        }
    }
}

// The index of the load is a function call result, which has no inline form,
// so re-evaluating the load in `continuing` reached the `unreachable!` rather
// than merely producing a wrong value.
@group(0) @binding(1) var<storage, read_write> v0: vec2<f32>;

fn one() -> u32 {
    return 1u;
}

fn call_result_as_index() {
    loop {
        let v1 = v0[one()];
        continuing {
            break if v1 > 0.0;
        }
    }
}

// `break_if` is written after the continuing block, and may name the
// expressions it emitted.
var<private> counter: i32 = 0;
fn break_if_uses_continuing() {
    loop {
        continuing {
            let seen = counter;
            counter += 1;
            break if seen > 3;
        }
    }
}

@compute @workgroup_size(1)
fn main() {
    named();
    call_result();
    through_pointer();
    nested();
    dead_use();
    restricted_image_load();
    call_result_as_index();
    break_if_uses_continuing();
}
