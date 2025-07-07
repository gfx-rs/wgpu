**Connections**
_Link to the issues addressed by this PR, or dependent PRs in other repositories_

_When one pull request builds on another, please put "Depends on
#NNNN" towards the top of its description. This helps maintainers
notice that they shouldn't merge it until its ancestor has been
approved. Don't use draft PR status to indicate this._

**Description**
_Describe what problem this is solving, and how it's solved._

**Testing**
_Explain how this change is tested._

**Squash or Rebase?**

_If your pull request contains multiple commits, please indicate whether
they need to be squashed into a single commit before they're merged,
or if they're ready to rebase onto `trunk` as they stand. In the
latter case, please ensure that each commit passes all CI tests, so
that we can continue to bisect along `trunk` to isolate bugs._

<!--
Thanks for filing! The codeowners file will automatically request reviews from the appropriate teams.

After you get a review and have addressed any comments, please explicitly re-request a review from the
person(s) who reviewed your changes. This will make sure it gets re-added to their review queue - you're not bothering us!
-->

**Checklist**

- [ ] Run `cargo fmt`.
- [ ] Run `taplo format`.
- [ ] Run `cargo clippy --tests`. If applicable, add:
  - [ ] `--target wasm32-unknown-unknown`
- [ ] Run `cargo xtask test` to run tests.
- [ ] If this contains user-facing changes, add a `CHANGELOG.md` entry. <!-- See instructions at the top of `CHANGELOG.md`. -->
