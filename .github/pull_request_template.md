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
Thanks for filing! Reviewers are assigned for non-draft PRs in the weekly wgpu maintainers meetings.

After you get a review and have addressed any comments, please explicitly re-request a review from the
person(s) who reviewed your changes. This will make sure it gets re-added to their review queue - you're not bothering us!
-->

**Checklist**

<!-- Note that checking all the boxes is not necessary to open a PR. -->

- [ ] I self-reviewed and fully understand this PR.
- [ ] WebGPU implementations built with `wgpu` may be affected behaviorally.
- [ ] Validation and feature gates are in place to confine behavioral changes.
- [ ] Tests demonstrate the validation and altered logic works. <!-- See `docs/testing.md` -->
- [ ] `CHANGELOG.md` entries for the user-facing effects of this change are present. <!-- See instructions at the top of `CHANGELOG.md`. -->
- [ ] The PR is minimal, and doesn't make sense to land as multiple PRs.
- [ ] Commits are logically scoped and individually reviewable.
- [ ] The PR description has enough context to understand the motivation and solution implemented.
