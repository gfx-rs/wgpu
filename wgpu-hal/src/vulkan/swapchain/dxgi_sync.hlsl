// No-op draw issued by the DXGI interop swapchain's sync command lists (see `dxgi.rs`).
//
// The single triangle is collapsed to a zero-area degenerate at the origin, so the rasterizer
// produces no fragments and the back buffer's contents are left untouched. The draw exists only so
// the interop queue carries real render-target work referencing the back buffer.
//
// The compiled DXIL is checked in next to this file as `dxgi_sync_vs.cso` / `dxgi_sync_ps.cso` and
// embedded via `include_bytes!`. The blobs are only byte-reproducible with the exact DXC that CI's
// `install-dxc` action pins (currently release v1.8.2505.1); a unit test in `dxgi.rs` recompiles
// this file with that compiler and byte-compares against the checked-in blobs. Regenerate both
// after editing:
//
//   dxc -T vs_6_0 -E vs_main -HV 2018 -Qstrip_debug -Qstrip_reflect -Fo dxgi_sync_vs.cso dxgi_sync.hlsl
//   dxc -T ps_6_0 -E ps_main -HV 2018 -Qstrip_debug -Qstrip_reflect -Fo dxgi_sync_ps.cso dxgi_sync.hlsl

#define NOOP_ROOT_SIGNATURE "RootFlags(0)"

// The vertex-shader container carries the (empty) root signature above; the swapchain creates its
// D3D12 root signature straight from this blob.
[RootSignature(NOOP_ROOT_SIGNATURE)]
float4 vs_main() : SV_Position {
    // All three vertices collapse to the origin: a degenerate, zero-area triangle.
    return float4(0.0, 0.0, 0.0, 1.0);
}

// Never reached (the triangle covers no pixels); included so the pipeline has both a vertex and
// a pixel shader.
float4 ps_main() : SV_Target {
    return float4(0.0, 0.0, 0.0, 0.0);
}
