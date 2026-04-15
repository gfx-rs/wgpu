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

- [ ] I self-reviewed this PR and fully understand it.
- [ ] This PR potentially affects behavior on WebGPU.
- [ ] I added all necessary validation to ensure behavior changes are confined to where they should be.
  - [ ] I added validation tests demonstrating that any behavior changes are in fact confined with proper error messages.
- [ ] I added all necessary `CHANGELOG.md` entries for this change. <!-- See instructions at the top of `CHANGELOG.md`. -->
- [ ] I think this PR is a minimal change that doesn't make sense to land as multiple separate PRs.
- [ ] I think the commit history is logical and decently reviewable.
- [ ] The PR description contains enough context for a reviewer to understand the motivation for this PR, and the solution it implements.
