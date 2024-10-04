**Connections**
_Link to the issues addressed by this PR, or dependent PRs in other repositories_

**Description**
_Describe what problem this is solving, and how it's solved._

**Testing**
_Explain how this change is tested._

<!--
Thanks for filing! The codeowners file will automatically request reviews from the appropriate teams.

After you get a review and have addressed any comments, please explicitly re-request a review from the
person(s) who reviewed your changes. This will make sure it gets re-added to their review queue - you're no bothering us!
-->

**Checklist**

- [ ] Run `cargo fmt`.
- [ ] Run `taplo format`.
- [ ] Run `cargo clippy`. If applicable, add:
  - [ ] `--target wasm32-unknown-unknown`
  - [ ] `--target wasm32-unknown-emscripten`
- [ ] Run `cargo xtask test` to run tests.
- [ ] Add change to `CHANGELOG.md`. See simple instructions inside file.
