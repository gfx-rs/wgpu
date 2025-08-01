This is an overview of how to run wgpu releases.

## Structure

We do a major breaking release every 12 weeks. This happens no matter the status of various in-flight projects.

We do a patch releases as needed in the weeks between major releases. Once a new major release is cut, we stop doing patch releases for the previous major release unless there is a critical bug or a compilation issue.

## People

Anyone in the @gfx-rs/wgpu team can perform these steps.

## Major Release Process

Approx 1 Week Before:
- Determine if `glow` (@groves), `metal-rs` (@gfx-rs/wgpu) or any other dependant crates will need a release. If so, coordinate with their maintainers.
- Go through the changelog:
  - Re-categorize miscategorized items.
  - Edit major changes so a user can easily understand what they need to do.
  - Add missing major changes that users need to know about.
  - Copy-edit the changelog for clarity.

Day of Release:
- Update the version number in the root `Cargo.toml` to the new version, this will update all crates to the new version.
- Bump the wgpu dependency numbers in the following places:
  - `Cargo.toml`
  - `examples/standalone/*`
- Grep for the previous version to ensure various documentation links are updated.
  - For example, if the previous version was v24.0.0, grep for `v24` and `24.0`
- Ensure `glow` and `metal` are updated to the latest version if needed.
- Add a new header for the changelog with the release version and date.
- Create a PR with all of the version changes and changelog updates.
- Once the PR is CI clean, (force) merge it.
- Checkout `trunk` with the merged PR.
- Publish! These commands can be pasted directly into your terminal in a single command, and they will publish everything.
  ```bash
    cargo publish -p naga
    cargo publish -p naga-cli
    cargo publish -p wgpu-types
    cargo publish -p wgpu-hal --all-features
    cargo publish -p wgpu-core-deps-apple
    cargo publish -p wgpu-core-deps-emscripten
    cargo publish -p wgpu-core-deps-wasm
    cargo publish -p wgpu-core-deps-windows-linux-android
    cargo publish -p wgpu-core --all-features
    cargo publish -p wgpu
    cargo publish -p wgpu-info
  ```
- If there were any newly published crates, ensure `github:gfx-rs/wgpu` is added as an owner of that crate.
- Create a new tag called `vX.Y.Z` and push it to the repo.
- Create a new release on the `wgpu` repo with the changelog from this version, targeting that tag
- Create a branch with the with the new version `vX` and push it to the repo.
  - On this branch, remove the [!NOTE] at the top of [wgpu/examples/README.md].
- Complete the release's milestone on GitHub.
- Create a new milestone for the next release, in 12 weeks time.
- Update the release checklist with any needed changes.
- Publish the link to the github release in the following places.
  - [r/rust](https://www.reddit.com/r/rust/).
    - Add an AMA comment.
  - Crosspost to [r/rust_gamedev](https://www.reddit.com/r/rust_gamedev/).
    - Add an AMA comment.
  - Include the r/rust post shortlink in the following posts as well:
  - [wgpu matrix](https://matrix.to/#/#wgpu:matrix.org)
  - [Rust Gamedev Discord](https://discord.gg/yNtPTb2) in the #crates channel
  - [Bevy Discord](https://discord.com/invite/bevy) in the #rendering-dev channel
  - [Graphics Programming Discord](https://discord.gg/6mgNGk7) in the #webgpu channel
  - [Rust Community Discord](https://discord.gg/rust-lang-community) in the #games-and-graphics channel

## Patch Release Process
- Enumerate all PRs that haven't been backported yet. These use the `needs-backport` label. [GH Link](https://github.com/gfx-rs/wgpu/pulls?q=sort%3Aupdated-desc+is%3Apr+label%3A%22PR%3A+needs+back-porting%22)
- On _your own branch_ based on the latest release branch. Cherry-pick the PRs that need to be backported. When modifying the commits, use --append to retain their original authorship.
- Remove the `needs-backport` label from the PRs.
- Fix the changelogs items and add a new header for the patch release with the release version and date.
- Once all the PRs are cherry-picked, look at the diff between HEAD and the previous patch release. See what crates changed.
- Bump all the versions of the crates that changed.
- Create a PR with all of the version changes and changelog updates into the release branch.
- Once the PR is CI clean, (force) rebase merge it.
- Checkout the release branch with the merged PR.
- Publish all relevant crates (see list above).
- Create a new release on the `wgpu` repo with the changelog and a tag called `vX.Y.Z` on the release branch.
  - The release should start with a header saying the following (for example)
    ```markdown
    This release includes `crate1`, `crate2` and `crate3` version `X.Y.Z`. All other crates remain at their previous versions.
    ```
- Backport the changelog and version bumps to the `trunk` branch.
  - Ensure that any items in the newly-released changelog don't appear in the "unreleased" section of the trunk changelog.
- Update the release checklist with any needed changes.
