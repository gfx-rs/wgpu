This document is a guide for contributions to the WGPU project.

## Welcome!

First of all, welcome to the WGPU community! 👋 We're glad you want to
contribute. If you are unfamiliar with the WGPU project, we recommend you read
[`GOVERNANCE.md`] for an overview of its goals, and how it's governed.

[`GOVERNANCE.md`]: ./GOVERNANCE.md

## Talking to other humans in the WGPU project

The WGPU project has multiple official platforms for community engagement:

- The Matrix channel [`wgpu:matrix.org`](https://matrix.to/#/#wgpu:matrix.org)
  is dedicated to informal chat about contributions the project. It is
  particularly useful for:

  - …saying hello, and introducing yourself.
  - …validating contributions (i.e., determining if they'll be accepted,
    ensuring your approach is correct, making sure you aren't wasting effort,
    etc.).
  - …setting expectations for contributions.

  Notification in Matrix can sometimes be unreliable. Feel free to explicitly
  tag people from whom you would like attention, esp. to follow-up after a day
  or so if you do not get a response to your contributions.

- [GitHub issues] are used to discuss open development questions and track work
  the community intends to complete; this might include:

  - Work that needs resolution via pull requests (see below)
    - Bug reports
    - Feature requests
    - Creating new releases of crates
  - Recording project decisions formally.
    - Architectural discussion
    - ???
  - Compiling sets of other issues needed for a specific feature or use case
    (AKA `[meta]` issues).

- [GitHub pull requests]: Modifications to the contents of this repository are
  done through pull requests.
- [GitHub discussions]: TODO: Experimentally used by some enthusiastic members
  of our community. Not supported officially.

[GitHub discussions]: https://github.com/gfx-rs/wgpu/discussions
[GitHub issues]: https://github.com/gfx-rs/wgpu/issues
[GitHub pull requests]: https://github.com/gfx-rs/wgpu/pulls

## Contributing to WGPU

Community response to contributions are, in general, prioritized based on their
relevance to WGPU's mission and decision-making groups' interest (see
[`GOVERNANCE.md`]).

### "What can I work on?" as a new contributor

TODO

We discourage new contributors from submitting large changes or opinionated
refactors unless they have been specifically validated by WGPU maintainership.
These are likely to be rejected on basis of needing discussion before a formal
review.

### Setting up a WGPU development environment

We use the following components in a WGPU development environment:

- [The version of the Rust toolchain with the `cargo` command][install-rust],
  pointed to by `rust-toolchain.toml` at the root of the repository, to compile
  WGPU's code.
- [Taplo](https://taplo.tamasfe.dev/) to keep TOML files formatted.

Once these are done, you should be ready to hack on WGPU! Drop into your
favorite editor, make some changes to the repository's code, and test that WGPU
has been changed the way you expect. We recommend
[using a `path` dependency][path-deps] in Cargo for local testing of changes,
and a [`git` dependency][git-deps] pointing to your own fork to share changes
with other contributors.

Once you are ready to request a review of your changes so they become part of
WGPU public history, create a pull request with your changes committed to a
branch in your own fork of WGPU in GitHub. See documentation for that
[here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

[install-rust]: https://www.rust-lang.org/tools/install
[path-deps]: https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#specifying-path-dependencies
[git-deps]: https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#specifying-dependencies-from-git-repositories

### What to expect when you file an issue

TODO

- Describe the filing process
  - Link to new issue page
  - Describe how to socialize the issue effectively
  - Feel free to ping us if it's a blocker!
  - Suggesting tags is helpful.
  - Describe how the project will handle the issue
    - Our ability to respond to an issue depends entirely on whether it is
      _actionable_ (viz., that there is a course of action that is reasonable
      for a volunteer to take the time to do). If it's not actionable, we
      reserve the right to close it.
      - Being responsive to requests for further information is important.
      - Understanding what point in the repository's history an issue began is
        also important. Maybe link to `git bisect` or something similar?
      - In particular, expecting others to fix something hardware- or
        driver-specific that current maintainership (1) can't mentor you
        into fixing and (2) otherwise isn't being prioritized are likely to
        be closed.

### What to expect when you submit a PR

TODO: It is strongly recommended that you validate your contributions before
you make significant efforts…

The "Assigned" field on a PR indicates who has taken responsibility
for reviewing the PR, not who is responsible for the content of the
PR.
