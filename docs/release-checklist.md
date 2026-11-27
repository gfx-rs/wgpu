This is an overview of how to run wgpu releases.

## Structure

We do a major breaking release every 12 weeks. This happens no matter the status of various in-flight projects.

We do a patch releases as needed in the weeks between major releases. Once a new major release is cut, we stop doing patch releases for the previous major release unless there is a critical bug or a compilation issue.

## People

Anyone in the @gfx-rs/wgpu team can perform these steps.

## Changelog workflow

The [fragment workflow](../changelog.d/README.md) starts immediately after v31.0.0, when the Unreleased section is empty.
After publication, cherry-pick the cutover onto the v31 release branch. If the branch does not have the fragment tooling, cherry-pick that change first.
Older release branches keep their existing changelog workflow.

Before a release, update your local trunk history. The release tool uses that history to find the original PRs, including for backports.
If your local trunk ref has a different name, pass `--trunk <ref>` to `preview` and `release`.
The release command needs `gh` and network access for attribution lookup.

## Major Release Process

Approx 1 Week Before:

- Determine if `glow` (@groves), `rspirv` (@gfx-rs/wgpu) or any other dependant crates will need a release. If so, coordinate with their maintainers.
- Run `cargo xtask changelog preview` and review the fragments:
  - Re-categorize miscategorized items.
  - Edit major changes so a user can easily understand what they need to do.
  - Merge fragments for any missing major changes into trunk.
  - Copy-edit the fragments for clarity.

Day of Release:

- Update the version number in the root `Cargo.toml` to the new version, this will update all crates to the new version.
- Bump the wgpu dependency numbers in the following places:
  - `Cargo.toml`
  - `examples/standalone/*`
- Grep for the previous version to ensure various documentation links are updated.
  - For example, if the previous version was v24.0.0, grep for `v24` and `24.0`
- Ensure `glow` and `rspirv` are updated to the latest version if needed.
- Commit any fragment edits, then run `cargo xtask changelog release vX.Y.Z` to assemble the release notes and delete the fragments.
- Copy-edit the assembled notes in `CHANGELOG.md`.
- Create a PR with all of the version changes and changelog updates.
  - Add the `changelog: released entry changed` label.
- While waiting on the PR, do a dry run of publishing.
  ```bash
    cargo publish --dry-run --workspace --all-features --exclude deno_webgpu
  ```
- Once the PR is CI clean and publish worked, (force) merge it.
- Checkout `trunk` with the merged PR.
- Publish! These commands can be pasted directly into your terminal in a single command, and they will publish everything.
  ```bash
    cargo publish --workspace --all-features --exclude deno_webgpu
  ```
- If there were any newly published crates, ensure `github:gfx-rs:wgpu` is added as an owner of that crate.
- Create a new signed tag called `vX.Y.Z` and push it to the repo.
  - For each crate being released (viz., every `publish`-able crate that is not `deno*`), create a new tag of the form `{crate_name}-vX.Y.Z`.
  - Push at most 25 tags in one push. A repository setting rejects a push that updates more.
- Create a new release on the `wgpu` repo with the changelog from this version, targeting that tag
- Create a branch with the new version `vX` and push it to the repo.
  - On this branch, remove the [!NOTE] at the top of [examples/README.md].
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
  - [Rust Gamedev Discord](https://discord.gg/X3MYBNXUMJ) in the #crates and #wgpu channel
  - [Bevy Discord](https://discord.com/invite/bevy) in the #rendering-dev channel
  - [Graphics Programming Discord](https://discord.gg/6mgNGk7) in the #webgpu channel
  - [Rust Community Discord](https://discord.gg/rust-lang-community) in the #games-and-graphics channel

## Patch Release Process

- Enumerate all PRs that haven't been backported yet. These use the `PR: needs back-porting` label. [GH Link](https://github.com/gfx-rs/wgpu/pulls?q=sort%3Aupdated-desc+is%3Apr+label%3A%22PR%3A+needs+back-porting%22)
- Review open PRs that target the release branch. If an approved PR has a fragment, merge it into trunk first so the release tool can find its original PR.
- On _your own branch_ based on the latest release branch. Cherry-pick the PRs that need to be backported, including their fragments. Keep the original author of each commit.
- Remove the `PR: needs back-porting` label from the PRs.
- Run `cargo xtask changelog release vX.Y.Z` to assemble the release notes and delete the fragments.
  - Copy-edit only the assembled notes in `CHANGELOG.md`. Do not edit fragments on the release branch; those edits can cause conflicts when the release notes return to trunk.
  - Commit the release notes and fragment deletions in their own commit, separate from version changes.
- Update the version number in the root `Cargo.toml` to the new version. Do not change the versions of the workspace dependencies, so that users can roll back a single crate.
- Create a PR with all of the version changes and changelog updates into the release branch.
  - Add the `changelog: released entry changed` label.
  - If CI on the release branch fails for causes that are not related to the release, such as retired runner images, fix it in this PR.
- While waiting on the PR, do a dry run of publishing.
  ```bash
    cargo publish --dry-run --workspace --all-features --exclude deno_webgpu
  ```
- Once the PR is CI clean and publish worked, fast-forward the release branch to it manually.
- Checkout the version branch with the merged PR.
- Publish! These commands can be pasted directly into your terminal in a single command, and they will publish everything.
  ```bash
    cargo publish --workspace --all-features --exclude deno_webgpu
  ```
- Create a new release on the `wgpu` repo with the relevant changelog included, based on a new tag called `vX.Y.Z` in the release branch.
  - For each crate released, also create a tag `{crate_name}-vX.Y.Z`.
  - Push at most 25 tags in one push. A repository setting rejects a push that updates more.
- Close the PRs into the release branch that the release PR replaced.
- After publication, cherry-pick the release notes commit to `trunk`. It adds the release section and deletes the same fragments from trunk.
- Forward port any release checklist changes to `trunk`.
- Update the release checklist with any needed changes.
