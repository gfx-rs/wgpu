# Naga physical pointer GPU test

Run from the repository root with `spirv-val` on PATH and the Khronos Vulkan
validation layer installed:

```powershell
$env:SPIRV_VAL = "path/to/spirv-val.exe"
# If the validation layer is not installed system-wide:
$env:VK_ADD_LAYER_PATH = "path/to/VulkanSDK/Bin"
cargo test -p naga-physical-storage-gpu -- --nocapture
# Also execute wider atomics and cooperative matrix multiply-add:
cargo test -p naga-physical-storage-gpu --features extended-native-tests -- --nocapture
```

Requires a native Vulkan GPU with buffer-device-address, shaderInt64 and scalarBlockLayout support, and host-visible
coherent buffer memory. Software adapters are rejected unless explicitly enabled
with `NAGA_ALLOW_SOFTWARE_ADAPTER=1`, as in the Mesa software execution CI job.
Software execution does not establish hardware support. Missing capabilities,
missing validation layers, and Vulkan validation errors fail the test.

Run in the debug profile: validation error capture requires debug assertions.
This package is excluded from default workspace members so normal Naga tests do
not require Vulkan or link wgpu. Its wgpu dependency enables only the Vulkan
backend, plus Naga IR input to check rejection by the safe shader API.

The test builds Naga IR without parsing shader text, validates it with an
explicit, minimal capability set per case, and validates the generated SPIR-V
with `spirv-val`. Every case runs the direct binary through wgpu passthrough and
checks results for unchecked, zero/skip, and restricted indexing. One
representative per case family (each bounds policy, the checked atomic module,
and each span mode) also runs a binary parsed back into Naga IR with spv-in and
re-emitted as SPIR-V. Out-of-range reads are made observable in readback; guard
values detect writes outside the allocation's payload. The safe shader API's
rejection of physical pointer IR is checked once per module family on each
device, for both direct and imported modules.

Extended cases cover helper pointer ABI, mutable pointer locals and selects,
address casts, signed offsets, aliasing, bounded spans, and scalar layouts.
Pointer operations are constructed directly in Naga IR.

The 26 base cases cover physical u32 atomics (including successful/failed
compare-exchange, acquire/release ordering, and checked out-of-range updates),
contention, standalone matrix pointers, packed/padded row-major and column-major
matrix layout conversion, runtime-sized arrays, null/default pointer values,
and explicit alignment. With 12 imported representatives they give 38
executions, which check 4,864 payload values and 152 guards. The
`extended-native-tests` feature adds 14 cases and 5 imported representatives,
giving 57 executions in total:

- Contended u64, i64, and f32 atomic adds verify the final counter and every
  returned old value. Integer counters use values outside the 32-bit range;
  signed counters include negative values.
- u64/i64 load/store, min/max, bitwise operations, subtraction, exchange, and
  successful/failed compare-exchange verify old values, exchange flags, and guards,
  with independent Relaxed/Acquire failure ordering.
- f16-input/f32-accumulator cooperative matrix multiply-add checks every output
  against an independent CPU calculation, plus unchanged inputs, padding, and
  guards. It covers row/column layouts and padded strides, using a
  configuration and subgroup size queried from the device. Padded cases use
  coherent matrix loads/stores.
- Coherent neighbor exchange across a workgroup barrier exercises Device,
  QueueFamily, and Workgroup scopes.
- Packed i8/u8 records and arrays exercise one-byte alignment, signed/unsigned
  extension, wrapping arithmetic, narrowing stores, and untouched neighboring
  bytes.

These cases additionally require `SHADER_INT64_ATOMIC_ALL_OPS`,
`SHADER_FLOAT32_ATOMIC`, `SHADER_F16`, and `EXPERIMENTAL_COOPERATIVE_MATRIX`, with
an advertised f16/f32 configuration whose dimensions Naga can represent (8 or
16). They also query and require Vulkan `shaderInt8`, `storageBuffer8BitAccess`,
and `uniformAndStorageBuffer8BitAccess`; the HAL callback enables byte storage.
Enabling the test feature makes these requirements mandatory; unsupported
hardware fails explicitly. The software CI job executes the 38 base executions
and lints the extended cases without running them. All 57 executions passed on
RTX 2080 Ti, driver 596.49, with no Vulkan validation errors, including
teardown. This does not establish float64 atomic support.

Compaction is not executed on the GPU. The compiler suite compacts the shared
pointer fixtures and their imports. The extended modules are built only here, so
each one asserts that compaction leaves its emitted SPIR-V unchanged; a
compacted execution would repeat an identical dispatch.

The compiler suite separately validates immutable argument contracts, matrix
arrays, stronger alignment, all coherence scopes, CAS ordering operands, byte
constant/override bounds, and rejection of byte atomics, immediate data, stage
I/O, and non-SPIR-V output.

The compiler suite also imports and re-emits the pointer fixtures before and
after compaction. It compares memory operand semantics, validates re-emitted
SPIR-V, and rejects malformed or unrepresentable physical memory operations.
