// This shader has gone through several stages of derived work:
//
// 1. Originally authored by a user named `dynamite`, distributed as a [Shadertoy page] under the
//    [CC BY-NC-SA 3.0 license].
//
//    [Shadertoy page]: https://www.shadertoy.com/view/XdlSDs
//    [CC BY-NC-SA 3.0 license]: https://creativecommons.org/licenses/by-nc-sa/3.0/deed.en
//
// 2. Ported to Slang for the Slang Playground project as [the `circle.slang` demo].
//
//    [the `circle.slang` demo]: https://github.com/shader-slang/slang-playground/blob/60f0ca29d9952d3cb598936511288f00451ced34/public/demos/circle.slang
//
// 3. Compiled to WGSL via Slang 2026.4 on the Slang playground.
//
//    For convenience, both the ported shader and the WGSL output can be found at:
//    <https://shader-slang.org/slang-playground/?target=WGSL&code=eJx9VNtu2kAQfcZfMcrTmsDa4JBKpqlEIJWQkjYqUW9RFC14jVeyvdZ6zaVR_r2za-5Ji7AFc86cOTsztufBAEqRFSmHMmERV1BIpXkEsZIZJFoXZeh5y-WS1rCWazqTmbcQfOn9jNLJqKSOhyqVTqQKIVrnLBOam9itmPG85CEMh3D9q_1l2J4MIKD-TnamONNiwVEwk3lJpZp7aZ1UetN1O5-1S-ZhhhdxHlGeOw5aRX9QpGw9V7LKo_42pHiO_kQ-7zvO4x4Pw4fx3c2TU-UiliqDOJVMgxYZP-V9_ja4u3kej065sWKWrNcFZ6lgJaDjLlzVcLd_DARbIDgBLrbAhSlcd5Oc4dGLSvMz98l5zKtMJ9iSqCSdyxbYy8QPTQ4Ht7dhOBn_vnn--pmcyUpj-gNf6UpZkYUUEYiMzfkdEzmpRK67EImyYHqWPFj18SiEyffn0UnQdV4cwE-k2PJerHhKTtNaQKxciXPj-UT84S5cfYIXp9GwHSnwhG-S6GoNTehSP4b2QSaGXfCA2I64B_F1H-U2Q2LVtmd0zvW9IG4ttacwJDDN8i4p6KpV0LW7xxRiKc_nOiGF2_Tph15_67RaIGZ-EYYmsE4L1EGmNmXN1BG89GmMA2s0PA89gE44zGQq1Y68GsrUHLxa0BUeUXuBcRnULjdgnMmIsGlJzH-3ZdDY3bgJAB8cDMtK1aYC4tNurwX7u6XiV8RgFeAjdKjvYsT0vrHLpwrOrwyEPgyvf4zODbqNv-LF05LDgWj3QNSG2lbtPZl_Fpm-V-RUtPtGdPpfUXUsaqdhF7WeBUw5yzBox0pQHAdQLVzU6hwui2H9EJFODAv3Ac7BlPFpD_kzWdYjbELHtwpm_ZrY_46FU5YVZlulIj2Ez7csk6dd1wzKb9mg65omNppgBm6OhGse-BtTuKFHg79GT9ux7_xZhuL4TOf2zUEI2VDNau3agkXNGiD5FW-vfwGjSbCy>

@binding(1) @group(0) var outputTexture_0 : texture_storage_2d<rgba8unorm, write>;

struct GlobalParams_std140_0
{
    @align(16) time_0 : f32,
    @align(4) frame_0 : f32,
};

@binding(0) @group(0) var<uniform> globalParams_0 : GlobalParams_std140_0;
struct imageMain_slang_Lambda_imageMain_1_0
{
     dispatchThreadID_0 : vec2<u32>,
};

fn imageMain_slang_Lambda_imageMain_1_x24init_0( dispatchThreadID_1 : vec2<u32>) -> imageMain_slang_Lambda_imageMain_1_0
{
    var _S1 : imageMain_slang_Lambda_imageMain_1_0;
    _S1.dispatchThreadID_0 = dispatchThreadID_1;
    return _S1;
}

fn float_getPi_0() -> f32
{
    return 3.14159274101257324f;
}

fn imageMain_slang_Lambda_imageMain_1_x28x29_0( this_0 : imageMain_slang_Lambda_imageMain_1_0,  screenSize_0 : vec2<i32>) -> vec4<f32>
{
    var _S2 : vec2<f32> = vec2<f32>(2.0f);
    var p_0 : vec2<f32> = (vec2<f32>(this_0.dispatchThreadID_0.xy) * _S2 - vec2<f32>(screenSize_0.xy)) / vec2<f32>(f32(screenSize_0.y));
    var tau_0 : f32 = float_getPi_0() * 2.0f;
    var _S3 : f32 = atan2(p_0.x, p_0.y) / tau_0;
    var uv_0 : vec2<f32> = vec2<f32>(_S3, length(p_0) * 0.75f);
    var t_0 : f32 = globalParams_0.frame_0 / 60.0f;
    var xCol_0 : f32 = ((((abs((_S3 - t_0 / 3.0f) * 3.0f))) % ((3.0f))));
    var horColour_0 : vec3<f32> = vec3<f32>(0.25f, 0.25f, 0.25f);
    if(xCol_0 < 1.0f)
    {
        horColour_0[i32(0)] = horColour_0[i32(0)] + (1.0f - xCol_0);
        horColour_0[i32(1)] = horColour_0[i32(1)] + xCol_0;
    }
    else
    {
        if(xCol_0 < 2.0f)
        {
            var xCol_1 : f32 = xCol_0 - 1.0f;
            horColour_0[i32(1)] = horColour_0[i32(1)] + (1.0f - xCol_1);
            horColour_0[i32(2)] = horColour_0[i32(2)] + xCol_1;
        }
        else
        {
            var xCol_2 : f32 = xCol_0 - 2.0f;
            horColour_0[i32(2)] = horColour_0[i32(2)] + (1.0f - xCol_2);
            horColour_0[i32(0)] = horColour_0[i32(0)] + xCol_2;
        }
    }
    var uv_1 : vec2<f32> = _S2 * uv_0 - vec2<f32>(1.0f);
    return vec4<f32>(vec3<f32>(((0.69999998807907104f + 0.5f * cos(uv_1.x * 10.0f * tau_0 * 0.15000000596046448f * clamp(floor(5.0f + 10.0f * cos(t_0)), 0.0f, 10.0f))) * abs(1.0f / (30.0f * uv_1.y)))) * horColour_0, 1.0f);
}

fn imageMain_slang_Lambda_imageMain_1_x24_syn_x28x29_0( this_1 : imageMain_slang_Lambda_imageMain_1_0,  _S4 : vec2<i32>) -> vec4<f32>
{
    return imageMain_slang_Lambda_imageMain_1_x28x29_0(this_1, _S4);
}

fn drawPixel_0( location_0 : vec2<u32>,  _S5 : imageMain_slang_Lambda_imageMain_1_0)
{
    var width_0 : u32 = u32(0);
    var height_0 : u32 = u32(0);
    {var dim = textureDimensions((outputTexture_0));((width_0)) = dim.x;((height_0)) = dim.y;};
    var color_0 : vec4<f32> = imageMain_slang_Lambda_imageMain_1_x24_syn_x28x29_0(_S5, vec2<i32>(i32(width_0), i32(height_0)));
    var _S6 : bool;
    if((location_0.x) >= width_0)
    {
        _S6 = true;
    }
    else
    {
        _S6 = (location_0.y) >= height_0;
    }
    if(_S6)
    {
        return;
    }
    textureStore((outputTexture_0), (location_0), (color_0));
    return;
}

@compute
@workgroup_size(16, 16, 1)
fn imageMain(@builtin(global_invocation_id) dispatchThreadID_2 : vec3<u32>)
{
    var dispatchThreadID_3 : vec2<u32> = dispatchThreadID_2.xy;
    drawPixel_0(dispatchThreadID_3, imageMain_slang_Lambda_imageMain_1_x24init_0(dispatchThreadID_3));
    return;
}
