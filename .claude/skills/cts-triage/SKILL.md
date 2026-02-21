---
name: cts-triage
description: Run CTS test suites and investigate failures
---

# Triage Process

When working on a category of CTS tests, follow this systematic process to identify issues, prioritize fixes, and document findings.

## Step 0: Divide Into Manageable Chunks

List all the tests matching a selector:

```bash
cargo xtask cts -- --list 'webgpu:api,validation,*' 2>&1 | wc -l
```

If there is a reasonable number of tests matching the selector (less than a
couple hundred or so, but this isn't a hard cutoff), then you can proceed with
triage. Otherwise, make a list of more detailed wildcards that match fewer
tests, verify that each wildcard matches a reasonable number of tests, and
triage each wildcard separately.

## Step 1: Get Overall Statistics

Run the full test suite for the category to understand the scope:

```bash
cargo xtask cts 'webgpu:api,validation,category:*' 2>&1 | grep -E "(Summary|Passed|Failed)" | tail -5
```

This gives you the pass rate and number of failures. Document this as your baseline.

## Step 2: Identify Test Subcategories

Review the output from the running the CTS (with or without `--list`) to
identify any subcategories that may exist within the suite being analyzed.
Subcategories typically have an additional `:`- or `,`-delimited word in the
test name. Running tests by subcategory may be more manageable than running
with the entire suite at once or running individual tests.

## Step 3: Run Each Subcategory

Test each subcategory individually to identify which ones are passing vs failing:

```bash
cargo xtask cts 'webgpu:api,validation,category:subcategory:*' 2>&1 | grep -E "(Passed|Failed|Skipped)" | tail -3
```

Track the pass rate for each subcategory. This helps you identify:
- What's already working (don't break it!)
- Where the failures are concentrated
- Which issues affect multiple categories

## Step 4: Analyze Failure Patterns

For failing subcategories, look at what tests are failing:

```bash
cargo xtask cts 'webgpu:api,validation,category:subcategory:*' 2>&1 | grep "\[fail\]" | head -20
```

Look for patterns:
- **Format-specific failures**: May indicate missing format capability checks
- **Parameter-specific failures**: Validation missing for specific parameter combinations

## Step 5: Examine Specific Failures

Pick a representative failing test and run it individually to see the error:

```bash
cargo xtask cts 'webgpu:api,validation,category:subcategory:specific_test' 2>&1 | tail -30
```

Look for:
- **"EXPECTATION FAILED: DID NOT REJECT"**: wgpu is accepting invalid input (validation gap)
- **"Validation succeeded unexpectedly"**: Similar to above
- **"Unexpected validation error occurred"**: wgpu is rejecting valid input
- **Error message content**: Tells you what validation is triggering or missing

## Step 6: Check CTS Test Source

To understand what the test expects, read the TypeScript source:

```bash
grep -A 40 "test_name" cts/src/webgpu/api/validation/path/file.spec.ts
```

The test source shows:
- What configurations are being tested
- What the expected behavior is (pass/fail)
- The validation rules from the WebGPU spec
- Comments explaining the rationale

## Step 7: Categorize Issues

Group failures into categories:

**High Priority - Validation Gaps:**
- wgpu accepts invalid configurations that should fail
- Security or correctness implications
- Example: Accepting wrong texture formats, missing aspect checks

**Medium Priority - Spec Compliance:**
- Edge cases not handled correctly
- Optional field validation issues
- Example: depthCompare optional field handling

**Low Priority - Minor Gaps:**
- Less common scenarios
- Limited real-world impact
- Example: Depth bias with non-triangle topologies

**Known Issues - Skip:**
- Known failure patterns (documented in AGENTS.md)
- Track count but don't try to fix

## Step 8: Identify Root Causes

For validation gaps, find where validation should happen:

1. **Search for existing validation:**
   ```bash
   grep -n "relevant_keyword" wgpu-core/src/device/resource.rs
   ```

2. **Look for render/compute pipeline creation:**
   - Render pipeline: `wgpu-core/src/device/resource.rs` around `create_render_pipeline`
   - Compute pipeline: Similar location
   - Look for existing validation patterns you can follow

3. **Check for helper functions:**
   ```bash
   grep "fn is_" wgpu-types/src/texture/format.rs
   ```

4. **Find error enums:**
   ```bash
   grep "pub enum.*Error" wgpu-core/src/pipeline.rs
   ```

## Step 9: Implement Fixes

When implementing fixes:

1. **Add error variants if needed** (in `wgpu-core/src/pipeline.rs`)
2. **Add helper methods** (in `wgpu-types` if checking properties)
3. **Add validation checks** (in `wgpu-core/src/device/resource.rs`)
4. **Test the fix** with specific failing tests
5. **Run full subcategory** to verify all related tests pass
6. **Check you didn't break passing tests**

## Step 10: Document Findings

Create or update a triage document (e.g., `category_triage.md`).

Do not write information about changes you have made to the triage document. Only capture the state of the tests and any investigation into open issues.

````markdown
# Category CTS Tests - Triage Report

**Overall Status:** XP/YF/ZS (%/%/%)

## Passing Sub-suites ✅
[List sub-suites that have no failures (all pass or skip)]

## Remaining Issues ⚠️
[List sub-suites that have failures and if it can be stated concisely, a summary of the issue]

## Issue Detail
[List detail of any investigation into failures. Do not go into detail about passed suites, just list the failures.]

### 1. title, e.g. a distinguishing word from the test selector
**Test selector:** `webgpu:api,validation,render_pipeline,depth_stencil_state:format:*`
**What it tests:** [Description]
**Example failure:**
[a selector for a single failing test, e.g.:]
```
webgpu:api,validation,render_pipeline,depth_stencil_state:depthCompare_optional:isAsync=false;format="stencil8"
```

**Error:**
[error message from the failing tests, e.g.:]
```
Unexpected validation error occurred: Depth/stencil state is invalid:
Format Stencil8 does not have a depth aspect, but depth test/write is enabled
```

**Root cause:**
[Your analysis of the root cause. Do not speculate. Only include the results of specific investigation you have done.]
The validation is triggering incorrectly. When `depthCompare` is undefined/missing in JavaScript, it's getting translated to a default value that makes `is_depth_enabled()` return true, even for stencil-only formats.

**Fix needed:**
[Your proposed fix. Again, do not speculate. Only state the fix if it is obvious from the root cause analysis, or if you have done specific investigation into how to fix it.]

### 2. title
[repeat as needed for additional issues]
````

## Step 11: Update test.lst

For fixed tests that are now passing, add them to `cts_runner/test.lst`:

1. **Use wildcards** to minimize lines:
   ```
   webgpu:api,validation,category:subcategory:isAsync=false;*
   ```

2. **Group related tests** when possible:
   - If multiple subcategories all pass, use a higher-level wildcard
   - Example: `webgpu:api,validation,category:*` if all subcategories pass

3. **Maintain alphabetical order** roughly in the file

## Step 12: Verify and Build

Before finishing:

```bash
# Format code
cargo fmt

# Check for errors
cargo clippy --tests

# Build to ensure no compilation errors
cargo build

# Run the tests you added to test.lst
cargo xtask cts 'webgpu:api,validation,category:subcategory:isAsync=false;*'
```

# Common Patterns

If the user asked to investigate a failure, and the cause of the failure is
noted here with "do not attempt to fix", then stop and ask the user before
attempting to fix.

**Pattern: Format-specific failures**
- Check if format validation is missing
- Look for `is_depth_stencil_format()`, `is_color_format()` etc.
- May need to add format capability checks

**Pattern: Aspect-related failures**
- Check if code validates format aspects (DEPTH, STENCIL, COLOR)
- Use `hal::FormatAspects::from(format)` to check
- Validate operations match available aspects

**Pattern: Optional field failures**
- May be WebGPU optional field semantics issue
- Check if undefined in JS becomes a default value in Rust
- May need to distinguish "not set" from "set to default"

**Pattern: Atomics accepted incorrectly**
- Naga allows referencing an atomic directly in an expression
- Should only allow accessing via `atomicLoad`, `atomicStore`, etc.
- Only investigate as necessary to confirm this is the issue. Do not attempt to fix. Refer user to https://github.com/gfx-rs/wgpu/issues/5474.

**Pattern: Error reporting for destroyed resources**
- Tests that check for validation errors when a destroyed resource is used. `wgpu` often reports these errors later than WebGPU requires, causing the tests to fail.
- `wgpu` may report these errors earlier than it should, causing the test to fail with an unexpected validation error.
- Look for:
  - Tests with `state="destroyed"` parameter
  - Tests checking that operations on destroyed buffers or textures should fail
- Example failing tests:
  - `webgpu:api,validation,encoding,cmds,compute_pass:indirect_dispatch_buffer_state:` with `state="destroyed"` subcases
- Only investigate as necessary to confirm this is the issue. Do not attempt to fix. Refer user to https://github.com/gfx-rs/wgpu/issues/7881.

# Tips

- **Start with high-impact fixes**: Validation gaps with security implications
- **Look for existing patterns**: Other validation code shows the style
- **Test incrementally**: Fix one category at a time, verify it works
- **Document as you go**: Don't wait until the end to write the triage
- **Ask for clarification**: If test expectations are unclear, check the WebGPU spec or test source
- **Track your progress**: Update pass rates as you fix issues
