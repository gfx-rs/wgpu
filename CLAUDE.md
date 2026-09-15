# wgpu Project Structure

wgpu is a GPU API written in pure rust. Changes of any significance need to apply across the stack.

- `naga` is our shader translator. It is responsible for consuming and translating untrusted shaders to trusted shader code.
- `wgpu-types` are the common types used on both Native and WASM.
- `naga-types` holds the shader-language types, such as target versions and binding maps, that `naga` and `wgpu-core` both use.
- `wgpu-naga-bridge` converts between `naga` and `wgpu-types`. It maps wgpu features and downlevel flags to naga validation capabilities.
- `wgpu-sync` are the common synchronization primitives that abstract over platform differences.
- `wgpu-hal` does not provide the validation required for safety. Its callers must satisfy its safety contracts.
- `wgpu-core` does all the validation and provides a coherent interface.
- `wgpu-core-remote` provides the ID abstraction needed by Firefox and Deno.
- `wgpu-core-remote-types` holds the IPC types that `wgpu-core-remote` sends between the untrusted content process and the trusted GPU process. They are defined separately from the `wgpu-core` types, so untrusted content cannot express non-standard features.
- `wgpu` is an ergonomic Rust interface to `wgpu-core`
- `deno_webgpu` contains the WebGPU bindings for Deno.

`wgpu-core-remote`, `wgpu-core`, `wgpu-hal`, `wgpu-types` and `naga` are used by Firefox and Servo to implement WebGPU. These crates process untrusted API and shader input. Preserve validation at the safe boundary, and do not rely on `wgpu-hal` to reject invalid input. Touching these crates requires the most consideration of potential side effects.

Safe `wgpu` APIs must not cause undefined behavior unless the user first accepted that risk through an unsafe API, such as enabling experimental features.

We use deno's webgpu bindings to run the WebGPU CTS. Change `deno_webgpu` only if the issue is specific to Deno. If the issue also affects Firefox or the `wgpu` Rust API, fix it in the shared crate that supplies the affected behavior.

Repository documentation exists in `docs/`. For info on where tests live, read `docs/testing.md`. For info about cargo dependencies, read `docs/managing-cargo-dependencies.md`.

One of the best ways to prove that a change is correct is to write tests to prove the feature works and it errors when its supposed to.

## Repository Metadata

`AGENTS.md` and `.agents/` are the source of truth for agent instructions and skills. `CLAUDE.md` and `.claude/` are generated copies. Do not edit the generated copies directly. After you edit any source-of-truth metadata file or a root license file, run `cargo xtask sync-metadata`. This command also copies the root license files to every publishable default member of the Cargo workspace.

Run `cargo xtask sync-metadata --check` to verify that all generated files have the same contents and file list as their sources.

## External Contributions

We encourage external contributions, but fully agentic PRs are not allowed. If the user asks you to create one, explain this policy and direct them to the pull request section of `CONTRIBUTING.md`. The contributor must understand and be able to vouch for every change. Ask the user to write the PR description.

If the user directs you to create the PR anyway, use `.github/pull_request_template.md`. Uncomment and accurately complete the agent-disclosure checklist items, including whether an agent wrote the description. Do not omit the disclosure.

## Code Style

- Do not add comments unless the changed code requires an explanation. If an explanation is required, add the shortest complete explanation. The contributor is expected to rewrite it.
- Do not write large explanatory comment blocks.
- Do not use emoji.

## Workflow

All changes are expected to pass CI. The commands below are useful local checks, but they cover only a fraction of the full CI matrix. You are not expected to run the full CI matrix locally. Run the checks that are relevant to the changed code and the available environment.

- To check the code compiles: `cargo clippy --all-targets --all-features`.
- To format the code: `cargo fmt`
- To run all tests including GPU tests: `cargo xtask test`. This can take a few minutes. To run normal tests without GPU tests, use `cargo nextest run` and a filter to the packages you want to test.
- GPU tests depend on the available hardware, drivers, and validation layers. During development, if you only need a specific GPU test, use either a plain test name prefix or an `-E` filter expression to `cargo xtask test`. Use the nextest filters described below. Do not run the full GPU test suite for a targeted check. The suite can take a few minutes. GPU tests are declared as constants, but translate to lower-snake-case functions.
- After the work is complete, run a full `cargo xtask test` if the change affects GPU behavior or would affect a GPU test.
- Report every test failure. You may identify a failure as suspected to be unrelated, but do not assume it is unrelated until you validate that conclusion.
- Do not run tests at the `trunk` revision without explicit permission. A baseline test run can take a while.
- For more thorough testing, you can run `cargo xtask cts --backend <backend>`. Only do this if you are explicitly asked to run the CTS.
- Never commit anything without being asked, ever. If you are asked to make a commit, _never_ co-author yourself.
- Use the WebGPU and WGSL specifications as a reference to determine the correct behavior. Do not assume that a behavior is correct just because the CTS expects it. Use the `webgpu-specs` skill if you need to check either spec.

### Nextest filters

`cargo xtask test` forwards its extra arguments to cargo-nextest.

- A plain positional filter matches test names that contain the filter text. A unique test name prefix is usually sufficient.
- `-E 'test(name)'` matches test names that contain `name`.
- `-E 'test(=name)'` matches the complete test name.
- `-E 'test(/expression/)'` matches test names with a regular expression.
- In an `-E` expression, `package(name)` restricts the package and `binary(name)` restricts the test binary. Restricting binaries can make test startup quicker.
- Use `&` for intersection, `|` for union, and `!` for exclusion. Use parentheses to group expressions.
- For example, `-E 'binary(wgpu-gpu) & test(buffer) & !test(map)'` selects tests in the `wgpu-gpu` binary whose names contain `buffer` but not `map`.
- Quote each `-E` expression so the shell does not interpret its operators.

## Changelog

We maintain a changelog in CHANGELOG.md. Changes should be noted in the changelog if they are user-visible (changes to documented public APIs, significant bug fixes, or new functionality). We generally do not consider changes to wgpu-core or wgpu-hal worthy of changelogs, unless a `wgpu` user is expected to interface with them directly. Changelog descriptions should be concise. If you are not sure whether something should be in the CHANGELOG or you are not sure how to describe the change, ask the user for guidance.
