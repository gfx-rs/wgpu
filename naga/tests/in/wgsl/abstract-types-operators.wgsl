const plus_fafaf: f32 = 1.0 + 2.0;
const plus_fafai: f32 = 1.0 + 2;
const plus_faf_f: f32 = 1.0 + 2f;
const plus_faiaf: f32 = 1 + 2.0;
const plus_faiai: f32 = 1 + 2;
const plus_fai_f: f32 = 1 + 2f;
const plus_f_faf: f32 = 1f + 2.0;
const plus_f_fai: f32 = 1f + 2;
const plus_f_f_f: f32 = 1f + 2f;

const plus_iaiai: i32 = 1 + 2;
const plus_iai_i: i32 = 1 + 2i;
const plus_i_iai: i32 = 1i + 2;
const plus_i_i_i: i32 = 1i + 2i;

const plus_uaiai: u32 = 1 + 2;
const plus_uai_u: u32 = 1 + 2u;
const plus_u_uai: u32 = 1u + 2;
const plus_u_u_u: u32 = 1u + 2u;

const bitflip_u_u: u32 = ~0xffffffffu;
const bitflip_uai: u32 = ~0xffffffff & (0x100000000 - 1);

const least_i32: i32 = -2147483648;
const least_f32: f32 = -3.40282347e+38;

const shl_iaiai: i32 = 1 << 2;
const shl_iai_u: i32 = 1 << 2u;
const shl_uaiai: u32 = 1 << 2;
const shl_uai_u: u32 = 1 << 2u;

const shr_iaiai: i32 = 1 >> 2;
const shr_iai_u: i32 = 1 >> 2u;
const shr_uaiai: u32 = 1 >> 2;
const shr_uai_u: u32 = 1 >> 2u;

fn runtime_values() {
  var f: f32 = 42;
  var i: i32 = 43;
  var u: u32 = 44;

  var plus_fafaf: f32 = 1.0 + 2.0;
  var plus_fafai: f32 = 1.0 + 2;
  var plus_faf_f: f32 = 1.0 + f;
  var plus_faiaf: f32 = 1 + 2.0;
  var plus_faiai: f32 = 1 + 2;
  var plus_fai_f: f32 = 1 + f;
  var plus_f_faf: f32 = f + 2.0;
  var plus_f_fai: f32 = f + 2;
  var plus_f_f_f: f32 = f + f;

  var plus_iaiai: i32 = 1 + 2;
  var plus_iai_i: i32 = 1 + i;
  var plus_i_iai: i32 = i + 2;
  var plus_i_i_i: i32 = i + i;

  var plus_uaiai: u32 = 1 + 2;
  var plus_uai_u: u32 = 1 + u;
  var plus_u_uai: u32 = u + 2;
  var plus_u_u_u: u32 = u + u;
}

fn wgpu_4445() {
  // This ok:
  let a = (3.0*2.0-(1.0)) * 1.0;
  let b = (3.0*2.0+1.0) * 1.0;
  // This fails:
  let c = (3.0*2.0-1.0) * 1.0;
}

const wgpu_4492 = i32(-0x80000000);
const wgpu_4492_2 = -2147483648;

var<workgroup> a: array<u32, 64>;

fn wgpu_4435() {
    let x = 1;
    let y = a[x-1];
}
