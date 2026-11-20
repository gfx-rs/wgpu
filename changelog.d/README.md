# Changelog fragments

Add an entry for a user-visible change as `changelog.d/<slug>.md`. Do not add unreleased entries to `CHANGELOG.md`.

## Quick path

1. Run `cargo xtask changelog new`. It prompts for a slug, category, and area. Press Enter for the `general` area.
   You can also supply the arguments: `cargo xtask changelog new vertex-format added metal`.
2. Write the Markdown body in the new file.
3. Run `cargo xtask changelog preview vertex-format` to inspect the entry, then `cargo xtask changelog check` to validate the fragments.
4. Include the fragment in your PR. You do not need to add the PR number after you open the PR.

Use a unique slug without slashes or colons. `README` is reserved.

## Format

Start each fragment with TOML between `+++` lines:

```markdown
+++
category = "added"
area = "metal"
+++

Added the `snorm10-10-10-2` vertex format on Metal.
```

| Key        | Required           | Value                                                                                                                                    |
| ---------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `category` | Yes                | `major`, `added`, `changes`, `bug-fixes`, `performance`, `documentation`, `dependency-updates`, `deno-webgpu`, `examples`, or `testing`. |
| `area`     | Except for `major` | `general`, `naga`, `validation`, `dx12`, `vulkan`, `metal`, `gles`, `webgpu`, `emscripten`, or `hal`. Use `gles` for WebGL.              |
| `prs`      | No                 | An array of PR numbers, such as `[123, 456]`. Defaults to the PR that added the fragment.                                                |
| `authors`  | No                 | An array of GitHub user names without `@`, such as `["contributor"]`. Defaults to the authors of the PRs.                                |

Omit `area` for `major`. Start the body of a `major` fragment with its own `####` heading.
For other categories, write the body without an outer list bullet. The tool adds the bullet, indentation, and attribution.
Unknown keys or values are errors.

At release time, the tool finds the original trunk commit that added each fragment and asks GitHub for its PR and author.
Set `prs` and `authors` when an entry describes different PRs or needs more than one author.
Within each category and area, entries follow their order in trunk history.

## Commands

- `cargo xtask changelog preview <slug>` previews one fragment offline.
- `cargo xtask changelog preview` previews all fragments offline. It uses local trunk history for order and puts drafts after merged entries.
  Previews show only attribution supplied in the fragment; they do not look up PRs or authors.
- `cargo xtask changelog check` validates fragments and checks for changes to `CHANGELOG.md` since the merge base with `trunk`.
  CI does not require every PR to have a fragment. Changes to `CHANGELOG.md` require the `changelog: released entry changed` label.
- `cargo xtask changelog release vX.Y.Z` writes the release section to `CHANGELOG.md` and removes the fragments.
  Commit the fragments first. The command needs current trunk history and `gh` with network access for attribution lookup.
  Use `--trunk <ref>` with `release` or the full preview if your local trunk ref has a different name.

## Releases and backports

Follow the [release checklist](../docs/release-checklist.md).
Cherry-pick fragments with their changes to release branches. The release tool uses original trunk history to link to the original PRs.
On a patch release branch, copy-edit only the assembled notes in `CHANGELOG.md`; leave the fragments unchanged.
Commit the assembled notes and fragment deletions in their own commit. After publication, cherry-pick that commit to trunk to remove the same unreleased entries.

The fragment workflow starts immediately after v31.0.0. At cutover, move any unreleased entries in your open PRs into fragments.
