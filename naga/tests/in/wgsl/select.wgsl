const_assert select(0xdeadbeef, 42f, false) == 0xdeadbeef;
const_assert select(0xdeadbeefu, 42, false) == 0xdeadbeefu;
const_assert select(0xdeadi, 42, false) == 0xdeadi;

const_assert select(42f, 0xdeadbeef, true) == 0xdeadbeef;
const_assert select(42, 0xdeadbeefu, true) == 0xdeadbeefu;
const_assert select(42, 0xdeadi, true) == 0xdeadi;

const_assert select(42f, 9001, true) == 9001;
const_assert select(42f, 9001, true) == 9001f;
const_assert select(42, 9001i, true) == 9001;
const_assert select(42, 9001u, true) == 9001;

const_assert select(9001, 42f, false) == 9001;
const_assert select(9001, 42f, false) == 9001f;
const_assert select(9001i, 42, false) == 9001;
const_assert select(9001u, 42, false) == 9001;

const_assert !select(false, true, false);
const_assert select(false, true, true);
const_assert select(true, false, false);
const_assert !select(true, false, true);

const_assert all(select(vec2(2f), vec2(), true) == vec2(0));
const_assert all(select(vec2(1), vec2(2f), false) == vec2(1));
const_assert all(select(vec2(1), vec2(2f), false) == vec2(1));
const_assert all(select(vec2(1), vec2(2f), vec2(false, false)) == vec2(1));
const_assert all(select(vec2(1), vec2(2f), vec2(true)) == vec2(2));
const_assert all(select(vec2(1), vec2(2f), vec2(true)) == vec2(2));
const_assert all(select(vec2(1), vec2(2f), vec2(true, false)) == vec2(2, 1));

const_assert all(select(vec3(1), vec3(2f), vec3(true)) == vec3(2));
const_assert all(select(vec4(1), vec4(2f), vec4(true)) == vec4(2));

@compute @workgroup_size(1, 1)
fn main() {
    _ = select(1, 2f, false);

    var x0 = vec2(1, 2);
    var i1: vec2<f32> = select(vec2<f32>(1., 0.), vec2<f32>(0., 1.), (x0.x < x0.y));
}
