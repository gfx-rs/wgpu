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

    foo = Foo(vec4<f32>(1.0), 1);
    let mat2comp = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
    let mat4comp = mat4x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    let unnamed = vec2<u32>(0u);
    let unnamed_1 = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
    let unnamed_2 = array<i32,4u>(0, 1, 2, 3);
    let _e59 = foo.a.x;
    return _e59;
}

fn logical() {
    let unnamed_3 = !(true);
    let unnamed_4 = !(vec2<bool>(true));
    let unnamed_5 = (true || false);
    let unnamed_6 = (true && false);
    let unnamed_7 = (true | false);
    let unnamed_8 = (vec3<bool>(true) | vec3<bool>(false));
    let unnamed_9 = (true & false);
    let unnamed_10 = (vec4<bool>(true) & vec4<bool>(false));
}

fn arithmetic() {
    let unnamed_11 = -(vec2<i32>(1));
    let unnamed_12 = -(vec2<f32>(1.0));
    let unnamed_13 = (2 + 1);
    let unnamed_14 = (2u + 1u);
    let unnamed_15 = (2.0 + 1.0);
    let unnamed_16 = (vec2<i32>(2) + vec2<i32>(1));
    let unnamed_17 = (vec3<u32>(2u) + vec3<u32>(1u));
    let unnamed_18 = (vec4<f32>(2.0) + vec4<f32>(1.0));
    let unnamed_19 = (2 - 1);
    let unnamed_20 = (2u - 1u);
    let unnamed_21 = (2.0 - 1.0);
    let unnamed_22 = (vec2<i32>(2) - vec2<i32>(1));
    let unnamed_23 = (vec3<u32>(2u) - vec3<u32>(1u));
    let unnamed_24 = (vec4<f32>(2.0) - vec4<f32>(1.0));
    let unnamed_25 = (2 * 1);
    let unnamed_26 = (2u * 1u);
    let unnamed_27 = (2.0 * 1.0);
    let unnamed_28 = (vec2<i32>(2) * vec2<i32>(1));
    let unnamed_29 = (vec3<u32>(2u) * vec3<u32>(1u));
    let unnamed_30 = (vec4<f32>(2.0) * vec4<f32>(1.0));
    let unnamed_31 = (2 / 1);
    let unnamed_32 = (2u / 1u);
    let unnamed_33 = (2.0 / 1.0);
    let unnamed_34 = (vec2<i32>(2) / vec2<i32>(1));
    let unnamed_35 = (vec3<u32>(2u) / vec3<u32>(1u));
    let unnamed_36 = (vec4<f32>(2.0) / vec4<f32>(1.0));
    let unnamed_37 = (2 % 1);
    let unnamed_38 = (2u % 1u);
    let unnamed_39 = (2.0 % 1.0);
    let unnamed_40 = (vec2<i32>(2) % vec2<i32>(1));
    let unnamed_41 = (vec3<u32>(2u) % vec3<u32>(1u));
    let unnamed_42 = (vec4<f32>(2.0) % vec4<f32>(1.0));
    let unnamed_43 = (vec2<i32>(2) + vec2<i32>(1));
    let unnamed_44 = (vec2<i32>(2) + vec2<i32>(1));
    let unnamed_45 = (vec2<u32>(2u) + vec2<u32>(1u));
    let unnamed_46 = (vec2<u32>(2u) + vec2<u32>(1u));
    let unnamed_47 = (vec2<f32>(2.0) + vec2<f32>(1.0));
    let unnamed_48 = (vec2<f32>(2.0) + vec2<f32>(1.0));
    let unnamed_49 = (vec2<i32>(2) - vec2<i32>(1));
    let unnamed_50 = (vec2<i32>(2) - vec2<i32>(1));
    let unnamed_51 = (vec2<u32>(2u) - vec2<u32>(1u));
    let unnamed_52 = (vec2<u32>(2u) - vec2<u32>(1u));
    let unnamed_53 = (vec2<f32>(2.0) - vec2<f32>(1.0));
    let unnamed_54 = (vec2<f32>(2.0) - vec2<f32>(1.0));
    let unnamed_55 = (vec2<i32>(2) * 1);
    let unnamed_56 = (2 * vec2<i32>(1));
    let unnamed_57 = (vec2<u32>(2u) * 1u);
    let unnamed_58 = (2u * vec2<u32>(1u));
    let unnamed_59 = (vec2<f32>(2.0) * 1.0);
    let unnamed_60 = (2.0 * vec2<f32>(1.0));
    let unnamed_61 = (vec2<i32>(2) / vec2<i32>(1));
    let unnamed_62 = (vec2<i32>(2) / vec2<i32>(1));
    let unnamed_63 = (vec2<u32>(2u) / vec2<u32>(1u));
    let unnamed_64 = (vec2<u32>(2u) / vec2<u32>(1u));
    let unnamed_65 = (vec2<f32>(2.0) / vec2<f32>(1.0));
    let unnamed_66 = (vec2<f32>(2.0) / vec2<f32>(1.0));
    let unnamed_67 = (vec2<i32>(2) % vec2<i32>(1));
    let unnamed_68 = (vec2<i32>(2) % vec2<i32>(1));
    let unnamed_69 = (vec2<u32>(2u) % vec2<u32>(1u));
    let unnamed_70 = (vec2<u32>(2u) % vec2<u32>(1u));
    let unnamed_71 = (vec2<f32>(2.0) % vec2<f32>(1.0));
    let unnamed_72 = (vec2<f32>(2.0) % vec2<f32>(1.0));
    let unnamed_73 = (mat3x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)) + mat3x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)));
    let unnamed_74 = (mat3x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)) - mat3x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)));
    let unnamed_75 = (mat3x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)) * 1.0);
    let unnamed_76 = (2.0 * mat3x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)));
    let unnamed_77 = (mat4x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)) * vec4<f32>(1.0));
    let unnamed_78 = (vec3<f32>(2.0) * mat4x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)));
    let unnamed_79 = (mat4x3<f32>(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)) * mat3x4<f32>(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0)));
}

fn bit() {
    let unnamed_80 = ~(1);
    let unnamed_81 = ~(1u);
    let unnamed_82 = !(vec2<i32>(1));
    let unnamed_83 = !(vec3<u32>(1u));
    let unnamed_84 = (2 | 1);
    let unnamed_85 = (2u | 1u);
    let unnamed_86 = (vec2<i32>(2) | vec2<i32>(1));
    let unnamed_87 = (vec3<u32>(2u) | vec3<u32>(1u));
    let unnamed_88 = (2 & 1);
    let unnamed_89 = (2u & 1u);
    let unnamed_90 = (vec2<i32>(2) & vec2<i32>(1));
    let unnamed_91 = (vec3<u32>(2u) & vec3<u32>(1u));
    let unnamed_92 = (2 ^ 1);
    let unnamed_93 = (2u ^ 1u);
    let unnamed_94 = (vec2<i32>(2) ^ vec2<i32>(1));
    let unnamed_95 = (vec3<u32>(2u) ^ vec3<u32>(1u));
    let unnamed_96 = (2 << 1u);
    let unnamed_97 = (2u << 1u);
    let unnamed_98 = (vec2<i32>(2) << vec2<u32>(1u));
    let unnamed_99 = (vec3<u32>(2u) << vec3<u32>(1u));
    let unnamed_100 = (2 >> 1u);
    let unnamed_101 = (2u >> 1u);
    let unnamed_102 = (vec2<i32>(2) >> vec2<u32>(1u));
    let unnamed_103 = (vec3<u32>(2u) >> vec3<u32>(1u));
}

fn comparison() {
    let unnamed_104 = (2 == 1);
    let unnamed_105 = (2u == 1u);
    let unnamed_106 = (2.0 == 1.0);
    let unnamed_107 = (vec2<i32>(2) == vec2<i32>(1));
    let unnamed_108 = (vec3<u32>(2u) == vec3<u32>(1u));
    let unnamed_109 = (vec4<f32>(2.0) == vec4<f32>(1.0));
    let unnamed_110 = (2 != 1);
    let unnamed_111 = (2u != 1u);
    let unnamed_112 = (2.0 != 1.0);
    let unnamed_113 = (vec2<i32>(2) != vec2<i32>(1));
    let unnamed_114 = (vec3<u32>(2u) != vec3<u32>(1u));
    let unnamed_115 = (vec4<f32>(2.0) != vec4<f32>(1.0));
    let unnamed_116 = (2 < 1);
    let unnamed_117 = (2u < 1u);
    let unnamed_118 = (2.0 < 1.0);
    let unnamed_119 = (vec2<i32>(2) < vec2<i32>(1));
    let unnamed_120 = (vec3<u32>(2u) < vec3<u32>(1u));
    let unnamed_121 = (vec4<f32>(2.0) < vec4<f32>(1.0));
    let unnamed_122 = (2 <= 1);
    let unnamed_123 = (2u <= 1u);
    let unnamed_124 = (2.0 <= 1.0);
    let unnamed_125 = (vec2<i32>(2) <= vec2<i32>(1));
    let unnamed_126 = (vec3<u32>(2u) <= vec3<u32>(1u));
    let unnamed_127 = (vec4<f32>(2.0) <= vec4<f32>(1.0));
    let unnamed_128 = (2 > 1);
    let unnamed_129 = (2u > 1u);
    let unnamed_130 = (2.0 > 1.0);
    let unnamed_131 = (vec2<i32>(2) > vec2<i32>(1));
    let unnamed_132 = (vec3<u32>(2u) > vec3<u32>(1u));
    let unnamed_133 = (vec4<f32>(2.0) > vec4<f32>(1.0));
    let unnamed_134 = (2 >= 1);
    let unnamed_135 = (2u >= 1u);
    let unnamed_136 = (2.0 >= 1.0);
    let unnamed_137 = (vec2<i32>(2) >= vec2<i32>(1));
    let unnamed_138 = (vec3<u32>(2u) >= vec3<u32>(1u));
    let unnamed_139 = (vec4<f32>(2.0) >= vec4<f32>(1.0));
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

@compute @workgroup_size(1, 1, 1) 
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
