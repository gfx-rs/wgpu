# Global instructions

## Code Style

- Limit the amount of comments you put in the code to a strict minimum. You should almost never add comments, except sometimes on non-trivial code, function definitions if the arguments aren't self-explanatory, and class definitions and their members.
- Do not use emoji.

## Workflow

- You can test that the code builds by running `cargo build`. Don't pass `--release` unless there's a specific need to test a release build, it's much slower.
- After making code changes, ensure the code is formatted by using `cargo fmt`, linted by using `cargo clippy --tests`. If there are no errors, you can use `cargo xtask test` and `cargo xtask cts --backend <backend>` to run tests (to fully validate a change, run both). On MacOS, the backend is `metal`. On Windows, the backend is `dx12`. On Linux, the backend is `vulkan`.
- Do not perform commits yourself, ever.
- Use the WebGPU and WGSL specifications as a reference to determine the correct behavior. Do not assume that a behavior is correct just because the CTS expects it.

## Changelog

We maintain a changelog in CHANGELOG.md. Changes should be noted in the
changelog if they are user-visible (changes to documented public APIs,
significant bug fixes, or new functionality). Changelog descriptions should be
concise. If you are not sure whether something should be in the CHANGELOG or
you are not sure how to describe the change, ask the user for guidance.

## More information about tests

There is some more detailed information about tests in @docs/testing.md.

# CTS

## Listing and Running Individual Tests

Each line is a selector that will run multiple subtests. You can run a test with:

```bash
cargo xtask cts 'webgpu:selector/path:test:*'
```

For fixing problems, it may be useful to run specific cases. You can get a list of the subtests with:

```bash
cargo xtask cts -- --list 'webgpu:selector,path:test:*'
```

Then you can pass individual subtests on the command line as the selector.

## Running Suites

To run a suite of CTS tests and keep just the summary at the end:

```bash
cargo xtask cts 'webgpu:selector,path:test:*' 2>&1 | grep -E "(Summary|Passed|Failed)" | tail -5
```

To run a suite of tests and keep just the failing tests:

```bash
cargo xtask cts 'webgpu:selector,path:test:*' 2>&1 | grep "\[fail\]"
```

## Keeping Track of Results

We maintain three files listing CTS test selectors that are expected to pass,
not pass, or to be entirely skipped. These files are respectively
cts_runner/test.lst, cts_runner/fail.lst, and cts_runner/skip.lst.

If you fix a CTS test, add the selector to test.lst. Be aware that the file is
not an exhaustive list of passing tests.

CI expects every test in test.lst to pass, so only add suites that have zero
failures to test.lst. Do not add any suites that are not 100% pass/skip, even
if they are ≥99% passing.

List suites with failures in fail.lst. You can put a comment `// xx%`
noting the pass rate, especially in cases where the pass rate is high.

List suites that are ≥90% skips in skip.lst

When adding to a lst file, use wildcards to identify the tests with as few lines
as possible. In some cases when adding tests it may be possible to combine with
existing lines to use a higher level wildcard. Only use a higher level wildcard
when all the tests it will match go in the same file. Do not use a higher level
wildcard if it would match tests that belong in a different file.

## Source for the CTS Tests

The TypeScript sources for the CTS are under cts/src. There is also generated JavaScript in cts/out.
Do not assume that a behavior is correct just because a CTS test expects it. Verify the correct
behavior in the WebGPU or WGSL specification.

# Overview of the Code

See `docs/big-picture.png` for an overview of the system.

The major components are:

- `wgpu-hal` implements a backend for each supported graphics API (Vulkan, DX12, Metal, GLES).

- `wgpu-core` implements the WebGPU API, including resource management and validation.
  It calls the platform graphics APIs via `wgpu-hal`, and uses `naga` for shader translation.

- `wgpu` is the native Rust API. In addition to providing bindings to `wgpu-core`, the `wgpu`
  crate can also be compiled to WASM and built against the "WebGPU" backend, where it uses
  whatever WebGPU implementation is provided by the WASM environment. `wgpu` is not used
  by Deno and Firefox.

- `naga` is the shader translator. It reads shaders in WGSL, GLSL, or SPIR-V,
  translates to Naga IR, and then writes shaders in GLSL, HLSL, MSL (Metal Shading Language), SPIR-V, or WGSL.
  It is responsible for validating that WGSL shaders are valid according to the WGSL language specification.

- `wgpu-types` contains some type definitions that are applicable both to `wgpu-core` and to
  the `wgpu` "WebGPU" backend.

- `deno_webgpu` contains WebGPU bindings for the Deno Javascript runtime.
  We also use Deno as a test environment for running the WebGPU CTS.
  Only make a change in the Deno bindings if you are sure that the issue
  doesn't apply to other clients (Firefox or `wgpu` Rust API). If it does
  apply to other clients, the issue should probably be fixed in `wgpu-core`.

For a more detailed discussion of the `wgpu` architecture, refer to
<https://github.com/gfx-rs/wgpu/wiki/Architecture>.

## Naga

### Constant Evaluator (`naga/src/proc/constant_evaluator.rs`)

The constant evaluator is responsible for evaluating constant expressions at compile time in WGSL/GLSL shaders.

**Key Components:**

1. **Expression Handling** (~line 1228): The `try_eval_and_append_impl` method handles different expression types and evaluates them if possible.

2. **Type Conversions**:
   - `cast()` (~line 2256): Handles type conversions with `Expression::As` where `convert` is `Some(width)`
   - `bitcast()` (~line 2449): Handles bit reinterpretation with `Expression::As` where `convert` is `None`
   - Key insight: For bitcast, the target width equals the source width (bit-preserving operation)

3. **Helper Macros**:
   - `gen_component_wise_extractor!`: Generates functions for component-wise operations on scalars/vectors
   - `component_wise_scalar!`, `component_wise_float!`: Convenience macros for applying operations to each component

4. **Borrow Checker Patterns**:
   - When implementing methods that need to call themselves recursively or make multiple mutable borrows, extract all needed data from `self` fields before the recursive calls
   - Clone vectors before iterating if you need to mutably borrow `self` during iteration

## Expression Types and Type Inference

- `Expression::As` with `convert: None` represents bitcast operations
- The validator (`naga/src/valid/expression.rs` ~line 1132) determines result types by:
  - Getting source type's scalar
  - Updating the `kind` to target kind
  - Keeping the same `width` for bitcast (no conversion)

# Other

This is stuff that Claude wrote for itself. It can probably be improved.

## Naga Tests

**Test Structure:**

1. **Unit Tests**: In the same file as the code being tested
   - Example: `naga/src/proc/constant_evaluator.rs` has tests in a `#[cfg(test)] mod tests` block
   - Pattern: Create arenas, build expressions, evaluate, assert results

2. **Snapshot Tests**:
   - Input files: `naga/tests/in/wgsl/*.wgsl`
   - Output files: `naga/tests/out/wgsl/*.wgsl`
   - Run with: `cargo test -p naga --test naga snapshots::convert_snapshots_wgsl`
   - The output shows how constant expressions are evaluated at compile time

3. **Adding Integration Tests**:
   - Add test WGSL code to `naga/tests/in/wgsl/const-exprs.wgsl` for const expression tests
   - The test automatically parses, validates, and generates output in `naga/tests/out/wgsl/wgsl-const-exprs.wgsl`
   - Verify constant evaluation by checking the output file

**Running Tests:**

```bash
# Unit tests only
cargo test -p naga --lib

# Specific test
cargo test -p naga --lib bitcast

# All naga tests (including integration)
cargo test -p naga

# WGSL snapshot tests
cargo test -p naga --test naga snapshots::convert_snapshots_wgsl
```

## WGSL Specification Implementation

When implementing WGSL built-in functions:

1. **Read the spec carefully**: WGSL spec section numbers are usually commented in code (e.g., "17.2.1. bitcast")
2. **Check parameterization**: The spec lists exactly which type combinations are valid
3. **Don't assume types**: For bitcast, only `i32`, `u32`, `f32` are specified - not 64-bit types
4. **AbstractInt special cases**: Some operations have special handling for abstract integers
