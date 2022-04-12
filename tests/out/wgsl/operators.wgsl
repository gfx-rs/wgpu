struct Foo {
    a: vec4<f32>,
    b: i32,
}

let v_f32_one: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0);
let v_f32_zero: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
let v_f32_half: vec4<f32> = vec4<f32>(0.5, 0.5, 0.5, 0.5);
let v_i32_one: vec4<i32> = vec4<i32>(1, 1, 1, 1);
fn builtins() -> vec4<f32> {
    let s1_ = select(0, 1, true);
    let s2_ = select(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), true);
    let s3_ = select(vec4<f32>(1.0, 1.0, 1.0, 1.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<bool>(false, false, false, false));
    let m1_ = mix(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), vec4<f32>(0.5, 0.5, 0.5, 0.5));
    let m2_ = mix(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), 0.10000000149011612);
    let b1_ = bitcast<f32>(vec4<i32>(1, 1, 1, 1).x);
    let b2_ = bitcast<vec4<f32>>(vec4<i32>(1, 1, 1, 1));
    let v_i32_zero = vec4<i32>(vec4<f32>(0.0, 0.0, 0.0, 0.0));
    return (((((vec4<f32>((vec4<i32>(s1_) + v_i32_zero)) + s2_) + m1_) + m2_) + vec4<f32>(b1_)) + b2_);
}

fn splat() -> vec4<f32> {
    let a_1 = (((vec2<f32>(1.0) + vec2<f32>(2.0)) - vec2<f32>(3.0)) / vec2<f32>(4.0));
    let b = (vec4<i32>(5) % vec4<i32>(2));
    return (a_1.xyxy + vec4<f32>(b));
}

fn bool_cast(x: vec3<f32>) -> vec3<f32> {
    let y = vec3<bool>(x);
    return vec3<f32>(y);
}

fn constructors() -> f32 {
    var foo: Foo;
    var unnamed: bool = false;
    var unnamed_1: i32 = 0;
    var unnamed_2: u32 = 0u;
    var unnamed_3: f32 = 0.0;
    var unnamed_4: vec2<u32> = vec2<u32>(0u, 0u);
    var unnamed_5: mat2x2<f32> = mat2x2<f32>(vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0));
    var unnamed_6: array<Foo,3> = array<Foo,3>(Foo(vec4<f32>(0.0, 0.0, 0.0, 0.0), 0), Foo(vec4<f32>(0.0, 0.0, 0.0, 0.0), 0), Foo(vec4<f32>(0.0, 0.0, 0.0, 0.0), 0));
    var unnamed_7: Foo = Foo(vec4<f32>(0.0, 0.0, 0.0, 0.0), 0);
    var unnamed_8: vec2<u32>;
    var unnamed_9: mat2x2<f32>;
    var unnamed_10: array<i32,4u>;

    foo = Foo(vec4<f32>(1.0), 1);
    let mat2comp = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
    let mat4comp = mat4x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    unnamed_8 = vec2<u32>(0u);
    unnamed_9 = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
    unnamed_10 = array<i32,4u>(0, 1, 2, 3);
    let _e70 = foo.a.x;
    return _e70;
}

fn logical() {
    let unnamed_11 = !(true);
    let unnamed_12 = !(vec2<bool>(true));
    let unnamed_13 = (true || false);
    let unnamed_14 = (true && false);
    let unnamed_15 = (true | false);
    let unnamed_16 = (vec3<bool>(true) | vec3<bool>(false));
    let unnamed_17 = (true & false);
    let unnamed_18 = (vec4<bool>(true) & vec4<bool>(false));
}

fn arithmetic() {
    let unnamed_19 = -(vec2<i32>(1));
    let unnamed_20 = -(vec2<f32>(1.0));
    let unnamed_21 = (2 + 1);
    let unnamed_22 = (2u + 1u);
    let unnamed_23 = (2.0 + 1.0);
    let unnamed_24 = (vec2<i32>(2) + vec2<i32>(1));
    let unnamed_25 = (vec3<u32>(2u) + vec3<u32>(1u));
    let unnamed_26 = (vec4<f32>(2.0) + vec4<f32>(1.0));
    let unnamed_27 = (2 - 1);
    let unnamed_28 = (2u - 1u);
    let unnamed_29 = (2.0 - 1.0);
    let unnamed_30 = (vec2<i32>(2) - vec2<i32>(1));
    let unnamed_31 = (vec3<u32>(2u) - vec3<u32>(1u));
    let unnamed_32 = (vec4<f32>(2.0) - vec4<f32>(1.0));
    let unnamed_33 = (2 * 1);
    let unnamed_34 = (2u * 1u);
    let unnamed_35 = (2.0 * 1.0);
    let unnamed_36 = (vec2<i32>(2) * vec2<i32>(1));
    let unnamed_37 = (vec3<u32>(2u) * vec3<u32>(1u));
    let unnamed_38 = (vec4<f32>(2.0) * vec4<f32>(1.0));
    let unnamed_39 = (2 / 1);
    let unnamed_40 = (2u / 1u);
    let unnamed_41 = (2.0 / 1.0);
    let unnamed_42 = (vec2<i32>(2) / vec2<i32>(1));
    let unnamed_43 = (vec3<u32>(2u) / vec3<u32>(1u));
    let unnamed_44 = (vec4<f32>(2.0) / vec4<f32>(1.0));
    let unnamed_45 = (2 % 1);
    let unnamed_46 = (2u % 1u);
    let unnamed_47 = (2.0 % 1.0);
    let unnamed_48 = (vec2<i32>(2) % vec2<i32>(1));
    let unnamed_49 = (vec3<u32>(2u) % vec3<u32>(1u));
    let unnamed_50 = (vec4<f32>(2.0) % vec4<f32>(1.0));
    let unnamed_51 = (vec2<i32>(2) + vec2<i32>(1));
    let unnamed_52 = (vec2<i32>(2) + vec2<i32>(1));
    let unnamed_53 = (vec2<i32>(2) - vec2<i32>(1));
    let unnamed_54 = (vec2<i32>(2) - vec2<i32>(1));
    let unnamed_55 = (vec2<i32>(2) * 1);
    let unnamed_56 = (2 * vec2<i32>(1));
    let unnamed_57 = (vec2<i32>(2) / vec2<i32>(1));
    let unnamed_58 = (vec2<i32>(2) / vec2<i32>(1));
    let unnamed_59 = (vec2<i32>(2) % vec2<i32>(1));
    let unnamed_60 = (vec2<i32>(2) % vec2<i32>(1));
    let unnamed_61 = (mat3x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)) * 1.0);
    let unnamed_62 = (2.0 * mat3x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)));
    let unnamed_63 = (mat4x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)) * vec4<f32>(1.0));
    let unnamed_64 = (vec3<f32>(2.0) * mat4x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)));
    let unnamed_65 = (mat4x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)) * mat3x4<f32>(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0)));
}

fn bit() {
    let unnamed_66 = ~(1);
    let unnamed_67 = ~(1u);
    let unnamed_68 = !(vec2<i32>(1));
    let unnamed_69 = !(vec3<u32>(1u));
    let unnamed_70 = (2 | 1);
    let unnamed_71 = (2u | 1u);
    let unnamed_72 = (vec2<i32>(2) | vec2<i32>(1));
    let unnamed_73 = (vec3<u32>(2u) | vec3<u32>(1u));
    let unnamed_74 = (2 & 1);
    let unnamed_75 = (2u & 1u);
    let unnamed_76 = (vec2<i32>(2) & vec2<i32>(1));
    let unnamed_77 = (vec3<u32>(2u) & vec3<u32>(1u));
    let unnamed_78 = (2 ^ 1);
    let unnamed_79 = (2u ^ 1u);
    let unnamed_80 = (vec2<i32>(2) ^ vec2<i32>(1));
    let unnamed_81 = (vec3<u32>(2u) ^ vec3<u32>(1u));
    let unnamed_82 = (2 << 1u);
    let unnamed_83 = (2u << 1u);
    let unnamed_84 = (vec2<i32>(2) << vec2<u32>(1u));
    let unnamed_85 = (vec3<u32>(2u) << vec3<u32>(1u));
    let unnamed_86 = (2 >> 1u);
    let unnamed_87 = (2u >> 1u);
    let unnamed_88 = (vec2<i32>(2) >> vec2<u32>(1u));
    let unnamed_89 = (vec3<u32>(2u) >> vec3<u32>(1u));
}

fn comparison() {
    let unnamed_90 = (2 == 1);
    let unnamed_91 = (2u == 1u);
    let unnamed_92 = (2.0 == 1.0);
    let unnamed_93 = (vec2<i32>(2) == vec2<i32>(1));
    let unnamed_94 = (vec3<u32>(2u) == vec3<u32>(1u));
    let unnamed_95 = (vec4<f32>(2.0) == vec4<f32>(1.0));
    let unnamed_96 = (2 != 1);
    let unnamed_97 = (2u != 1u);
    let unnamed_98 = (2.0 != 1.0);
    let unnamed_99 = (vec2<i32>(2) != vec2<i32>(1));
    let unnamed_100 = (vec3<u32>(2u) != vec3<u32>(1u));
    let unnamed_101 = (vec4<f32>(2.0) != vec4<f32>(1.0));
    let unnamed_102 = (2 < 1);
    let unnamed_103 = (2u < 1u);
    let unnamed_104 = (2.0 < 1.0);
    let unnamed_105 = (vec2<i32>(2) < vec2<i32>(1));
    let unnamed_106 = (vec3<u32>(2u) < vec3<u32>(1u));
    let unnamed_107 = (vec4<f32>(2.0) < vec4<f32>(1.0));
    let unnamed_108 = (2 <= 1);
    let unnamed_109 = (2u <= 1u);
    let unnamed_110 = (2.0 <= 1.0);
    let unnamed_111 = (vec2<i32>(2) <= vec2<i32>(1));
    let unnamed_112 = (vec3<u32>(2u) <= vec3<u32>(1u));
    let unnamed_113 = (vec4<f32>(2.0) <= vec4<f32>(1.0));
    let unnamed_114 = (2 > 1);
    let unnamed_115 = (2u > 1u);
    let unnamed_116 = (2.0 > 1.0);
    let unnamed_117 = (vec2<i32>(2) > vec2<i32>(1));
    let unnamed_118 = (vec3<u32>(2u) > vec3<u32>(1u));
    let unnamed_119 = (vec4<f32>(2.0) > vec4<f32>(1.0));
    let unnamed_120 = (2 >= 1);
    let unnamed_121 = (2u >= 1u);
    let unnamed_122 = (2.0 >= 1.0);
    let unnamed_123 = (vec2<i32>(2) >= vec2<i32>(1));
    let unnamed_124 = (vec3<u32>(2u) >= vec3<u32>(1u));
    let unnamed_125 = (vec4<f32>(2.0) >= vec4<f32>(1.0));
}

fn assignment() {
    var a: i32 = 1;

    let _e6 = a;
    a = (_e6 + 1);
    let _e9 = a;
    a = (_e9 - 1);
    let _e12 = a;
    let _e13 = a;
    a = (_e12 * _e13);
    let _e15 = a;
    let _e16 = a;
    a = (_e15 / _e16);
    let _e18 = a;
    a = (_e18 % 1);
    let _e21 = a;
    a = (_e21 & 0);
    let _e24 = a;
    a = (_e24 | 0);
    let _e27 = a;
    a = (_e27 ^ 0);
    let _e30 = a;
    a = (_e30 << 2u);
    let _e33 = a;
    a = (_e33 >> 1u);
    let _e36 = a;
    a = (_e36 + 1);
    let _e39 = a;
    a = (_e39 - 1);
    return;
}

@stage(compute) @workgroup_size(1, 1, 1) 
fn main() {
    let _e4 = builtins();
    let _e5 = splat();
    let _e7 = bool_cast(vec4<f32>(1.0, 1.0, 1.0, 1.0).xyz);
    let _e8 = constructors();
    logical();
    arithmetic();
    bit();
    comparison();
    assignment();
    return;
}
