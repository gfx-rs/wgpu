@fragment
fn fs_main() -> @location(0) vec4f {
    // Make sure case-insensitive keywords are escaped in HLSL.
    var Pass = vec4(1.0,1.0,1.0,1.0);
    return Pass;
}