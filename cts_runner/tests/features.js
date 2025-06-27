const adapter = await navigator.gpu.requestAdapter();

if (adapter.features.has("mappable-primary-buffers")) {
    throw new TypeError("Adapter should not report support for wgpu native-only features");
}
