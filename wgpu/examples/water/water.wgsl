[[block]]
struct Uniforms {
    view: mat4x4<f32>;
    projection: mat4x4<f32>;
    time_size_width: vec4<f32>;
    viewport_height: f32;
};
[[group(0), binding(0)]] var<uniform> uniforms: Uniforms;

const light_point: vec3<f32> = vec3<f32>(150.0, 70.0, 0.0);
const light_colour: vec3<f32> = vec3<f32>(1.0, 0.98, 0.82);
const one: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0);

const Y_SCL: f32 = 0.86602540378443864676372317075294;
const CURVE_BIAS: f32 = -0.1;
const INV_1_CURVE_BIAS: f32 = 1.11111111111; //1.0 / (1.0 + CURVE_BIAS);

//
// The following code to calculate simplex 3D
// is from https://github.com/ashima/webgl-noise
//
//  Simplex 3D Noise
//  by Ian McEwan, Ashima Arts.
//
fn permute(x: vec4<f32>) -> vec4<f32> {
    var temp: vec4<f32> = 289.0 * one;
    return modf(((x*34.0) + one) * x, &temp);
}

fn taylorInvSqrt(r: vec4<f32>) -> vec4<f32> {
    return 1.79284291400159 * one - 0.85373472095314 * r;
}

fn snoise(v: vec3<f32>) -> f32 {
    const C = vec2<f32>(1.0/6.0, 1.0/3.0);
    const D = vec4<f32>(0.0, 0.5, 1.0, 2.0);

    // First corner
    //TODO: use the splat operations when available
    const vCy = dot(v, C.yyy);
    var i: vec3<f32> = floor(v + vec3<f32>(vCy, vCy, vCy));
    const iCx = dot(i, C.xxx);
    const x0 = v - i + vec3<f32>(iCx, iCx, iCx);

    // Other corners
    const g = step(x0.yzx, x0.xyz);
    const l = (vec3<f32>(1.0, 1.0, 1.0) - g).zxy;
    const i1 = min(g, l);
    const i2 = max(g, l);

    //   x0 = x0 - 0.0 + 0.0 * C.xxx;
    //   x1 = x0 - i1  + 1.0 * C.xxx;
    //   x2 = x0 - i2  + 2.0 * C.xxx;
    //   x3 = x0 - 1.0 + 3.0 * C.xxx;
    const x1 = x0 - i1 + C.xxx;
    const x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    const x3 = x0 - D.yyy; // -1.0+3.0*C.x = -0.5 = -D.y

    // Permutations
    var temp: vec3<f32> = 289.0 * one.xyz;
    i = modf(i, &temp);
    const p = permute(
        permute(
            permute(i.zzzz + vec4<f32>(0.0, i1.z, i2.z, 1.0))
            + i.yyyy + vec4<f32>(0.0, i1.y, i2.y, 1.0))
        + i.xxxx + vec4<f32>(0.0, i1.x, i2.x, 1.0));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    const n_ = 0.142857142857;// 1.0/7.0
    const ns = n_ * D.wyz - D.xzx;

    const j = p - 49.0 * floor(p * ns.z * ns.z);//  mod(p,7*7)

    const x_ = floor(j * ns.z);
    const y_ = floor(j - 7.0 * x_);// mod(j,N)

    var x: vec4<f32> = x_ *ns.x + ns.yyyy;
    var y: vec4<f32> = y_ *ns.x + ns.yyyy;
    const h = one - abs(x) - abs(y);

    const b0 = vec4<f32>(x.xy, y.xy);
    const b1 = vec4<f32>(x.zw, y.zw);

    //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - one;
    //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - one;
    const s0 = floor(b0)*2.0 + one;
    const s1 = floor(b1)*2.0 + one;
    const sh = -step(h, 0.0 * one);

    const a0 = b0.xzyw + s0.xzyw*sh.xxyy;
    const a1 = b1.xzyw + s1.xzyw*sh.zzww;

    var p0: vec3<f32> = vec3<f32>(a0.xy, h.x);
    var p1: vec3<f32> = vec3<f32>(a0.zw, h.y);
    var p2: vec3<f32> = vec3<f32>(a1.xy, h.z);
    var p3: vec3<f32> = vec3<f32>(a1.zw, h.w);

    //Normalise gradients
    const norm = taylorInvSqrt(vec4<f32>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 = p0 * norm.x;
    p1 = p1 * norm.y;
    p2 = p2 * norm.z;
    p3 = p3 * norm.w;

    // Mix final noise value
    var m: vec4<f32> = max(0.6 * one - vec4<f32>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0 * one);
    m = m * m;
    return 9.0 * dot(m*m, vec4<f32>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

// End of 3D simplex code.

fn apply_distortion(pos: vec3<f32>) -> vec3<f32> {
    var perlin_pos: vec3<f32> = pos;

    //Do noise transformation to permit for smooth,
    //continuous movement.

    //TODO: we should be able to name them `sin` and `cos`.
    const sn = uniforms.time_size_width.x;
    const cs = uniforms.time_size_width.y;
    const size = uniforms.time_size_width.z;

    // Rotate 90 Z, Move Left Size / 2
    perlin_pos = vec3<f32>(perlin_pos.y - perlin_pos.x - size, perlin_pos.x, perlin_pos.z);

    const xcos = perlin_pos.x * cs;
    const xsin = perlin_pos.x * sn;
    const ycos = perlin_pos.y * cs;
    const ysin = perlin_pos.y * sn;
    const zcos = perlin_pos.z * cs;
    const zsin = perlin_pos.z * sn;

    // Rotate Time Y
    const perlin_pos_y = vec3<f32>(xcos + zsin, perlin_pos.y, -xsin + xcos);

    // Rotate Time Z
    const perlin_pos_z = vec3<f32>(xcos - ysin, xsin + ycos, perlin_pos.x);

    // Rotate 90 Y
    perlin_pos = vec3<f32>(perlin_pos.z - perlin_pos.x, perlin_pos.y, perlin_pos.x);

    // Rotate Time X
    const perlin_pos_x = vec3<f32>(perlin_pos.x, ycos - zsin, ysin + zcos);

    // Sample at different places for x/y/z to get random-looking water.
    return vec3<f32>(
        //TODO: use splats
        pos.x + snoise(perlin_pos_x + 2.0*one.xxx) * 0.4,
        pos.y + snoise(perlin_pos_y - 2.0*one.xxx) * 1.8,
        pos.z + snoise(perlin_pos_z) * 0.4
    );
}

// Multiply the input by the scale values.
fn make_position(original: vec2<f32>) -> vec4<f32> {
    const interpreted = vec3<f32>(original.x * 0.5, 0.0, original.y * Y_SCL);
    return vec4<f32>(apply_distortion(interpreted), 1.0);
}

// Create the normal, and apply the curve. Change the Curve Bias above.
fn make_normal(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    const norm = normalize(cross(b - c, a - c));
    const center = (a + b + c) * (1.0 / 3.0); //TODO: use splat
    return (normalize(a - center) * CURVE_BIAS + norm) * INV_1_CURVE_BIAS;
}

// Calculate the fresnel effect.
fn calc_fresnel(view: vec3<f32>, normal: vec3<f32>) -> f32 {
    var refractive: f32 = abs(dot(view, normal));
    refractive = pow(refractive, 1.33333333333);
    return refractive;
}

// Calculate the specular lighting.
fn calc_specular(eye: vec3<f32>, normal: vec3<f32>, light: vec3<f32>) -> f32 {
    const light_reflected = reflect(light, normal);
    var specular: f32 = max(dot(eye, light_reflected), 0.0);
    specular = pow(specular, 10.0);
    return specular;
}

struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(0)]] f_WaterScreenPos: vec2<f32>;
    [[location(1)]] f_Fresnel: f32;
    [[location(2)]] f_Light: vec3<f32>;
};

[[stage(vertex)]]
fn vs_main(
    [[location(0)]] position: vec2<i32>,
    [[location(1)]] offsets: vec4<i32>,
) -> VertexOutput {
    const p_pos = vec2<f32>(position);
    const b_pos = make_position(p_pos + vec2<f32>(offsets.xy));
    const c_pos = make_position(p_pos + vec2<f32>(offsets.zw));
    const a_pos = make_position(p_pos);
    const original_pos = vec4<f32>(p_pos.x * 0.5, 0.0, p_pos.y * Y_SCL, 1.0);

    const vm = uniforms.view;
    const transformed_pos = vm * a_pos;
    //TODO: use vector splats for division
    const water_pos = transformed_pos.xyz * (1.0 / transformed_pos.w);
    const normal = make_normal((vm * a_pos).xyz, (vm * b_pos).xyz, (vm * c_pos).xyz);
    const eye = normalize(-water_pos);
    const transformed_light = vm * vec4<f32>(light_point, 1.0);

    var out: VertexOutput;
    out.f_Light = light_colour * calc_specular(eye, normal, normalize(water_pos.xyz - (transformed_light.xyz * (1.0 / transformed_light.w))));
    out.f_Fresnel = calc_fresnel(eye, normal);

    const gridpos = uniforms.projection * vm * original_pos;
    out.f_WaterScreenPos = (0.5 * gridpos.xy * (1.0 / gridpos.w)) + vec2<f32>(0.5, 0.5);

    out.position = uniforms.projection * transformed_pos;
    return out;
}


const water_colour: vec3<f32> = vec3<f32>(0.0, 0.46, 0.95);
const zNear: f32 = 10.0;
const zFar: f32 = 400.0;

[[group(0), binding(1)]] var reflection: texture_2d<f32>;
[[group(0), binding(2)]] var terrain_depth_tex: texture_2d<f32>;
[[group(0), binding(3)]] var colour_sampler: sampler;

fn to_linear_depth(depth: f32) -> f32 {
    const z_n: f32 = 2.0 * depth - 1.0;
    const z_e: f32 = 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));
    return z_e;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    const reflection_colour = textureSample(reflection, colour_sampler, in.f_WaterScreenPos.xy).xyz;

    const pixel_depth = to_linear_depth(in.position.z);
    const normalized_coords = in.position.xy / vec2<f32>(uniforms.time_size_width.w, uniforms.viewport_height);
    const terrain_depth = to_linear_depth(textureSample(terrain_depth_tex, colour_sampler, normalized_coords).r);

    const dist = terrain_depth - pixel_depth;
    const clamped = pow(smoothStep(0.0, 1.5, dist), 4.8);

    const final_colour = in.f_Light + reflection_colour;
    const t = smoothStep(1.0, 5.0, dist) * 0.2; //TODO: splat for mix()?
    const depth_colour = mix(final_colour, water_colour, vec3<f32>(t, t, t));

    return vec4<f32>(depth_colour, clamped * (1.0 - in.f_Fresnel));
}
