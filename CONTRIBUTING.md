This document is a guide for contributions to the WGPU project.

## Welcome!

First of all, welcome to the WGPU community! 👋 We're glad you want to
contribute. If you are unfamiliar with the WGPU project, we recommend you read
[`GOVERNANCE.md`] for an overview of its goals, and how it's governed.

## Documentation Overview:

- [`GOVERNANCE.md`]: An overview of the WGPU project's goals and governance.
- [`CODE_OF_CONDUCT.md`]: The code of conduct for the WGPU project.
- [`docs/release-checklist.md`]: Checklist for creating a new release of WGPU.
- [`docs/review-checklist.md`]: Checklist for reviewing a pull request in WGPU.
- [`docs/testing.md`]: Information on the test suites in WGPU and Naga.

[`GOVERNANCE.md`]: ./GOVERNANCE.md
[`CODE_OF_CONDUCT.md`]: ./CODE_OF_CONDUCT.md
[`docs/release-checklist.md`]: ./docs/release-checklist.md
[`docs/review-checklist.md`]: ./docs/review-checklist.md
[`docs/testing.md`]: ./docs/testing.md

## Talking to other humans in the WGPU project

The WGPU project has multiple official platforms for community engagement:

- The Matrix channel [`wgpu:matrix.org`](https://matrix.to/#/#wgpu:matrix.org)
  is dedicated to informal chat about contributions the project. It is
  particularly useful for:

  - Saying hello, and introducing yourself.
  - Validating contributions (i.e., determining if they'll be accepted,
    ensuring your approach is correct, making sure you aren't wasting effort,
    etc.).
  - Setting expectations for contributions.

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
- `wgpu` Maintainership Meetings: Every week, the maintainership of the wgpu
  project meets to discuss the project's direction and review ongoing work.
  These meetings are open to the public, and you are welcome to attend. They
  happen on Google Meet and happen on Wednesday at 11:00 US Eastern Standard
  Time and last approximately an hour. Remember to obey the
  [`CODE_OF_CONDUCT.md`] in the meeting.

  - [Meeting Notes]
  - [Meeting Link]
- [GitHub discussions]: TODO: Experimentally used by some enthusiastic members
  of our community. Not supported officially.
  

[GitHub discussions]: https://github.com/gfx-rs/wgpu/discussions
[GitHub issues]: https://github.com/gfx-rs/wgpu/issues
[GitHub pull requests]: https://github.com/gfx-rs/wgpu/pulls
[Meeting Notes]: https://docs.google.com/document/d/1Z3qjy3m7eAYaTsh2n-iKxLV4Hjc6wZxgukzdQOgVH1c/edit?usp=sharing
[Meeting Link]: https://meet.google.com/ubo-ztcw-gwf
[`CODE_OF_CONDUCT.md`]: ./CODE_OF_CONDUCT.md

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

- [A Rust toolchain][install-rust] matching the version specified in
  [`rust-toolchain.toml`](./rust-toolchain.toml), to compile WGPU's code. If you
  use `rustup`, this will be automatically installed when you first run a
  `cargo` command in the repository.
- [Taplo](https://taplo.tamasfe.dev/) to keep TOML files formatted.
- [Vulkan SDK](https://vulkan.lunarg.com/) to provide Vulkan validation layers
  and other Vulkan/SPIR-V tools for testing.

Once these are done, you should be ready to hack on WGPU! Drop into your
favorite editor, make some changes to the repository's code, and test that WGPU
has been changed the way you expect. Take a look at [`docs/testing.md`] for more
info on testing.

When testing your own code against your patch, we recommend
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

### Pull requests

You can see some common things that PR reviewers are going to look for in
[`docs/review-checklist.md`].

A draft pull request is taken to be not yet ready for review. Marking
drafts as such helps the maintainers triage review work.

The `Assigned` field on a pull request indicates who has taken
responsibility for shepherding it through the review process, not who
is responsible for authoring it. The assignee is usually the reviewer,
but they can also delegate the review to someone else. The intent of
assignment is simply to ensure that pull requests don't get neglected.

#### Designing new features

As an open source project, WGPU wants to serve a broad audience. This
helps us cast a wide net for contributors, and widens the impact of
their work. However, WGPU does not promise to incorporate every
proposed feature.

Large efforts that are ultimately rejected tend to burn contributors
out on both sides of a review. To avoid this, we strongly encourage
you to validate time-consuming contributions by engaging
maintainership before you invest yourself too heavily. Try to build a
consensus on the approach, including API changes, shader language
extensions, implementation architecture, error handling, testing
plans, benchmarking, and so on.

#### Large pull requests are risky

Contributors should anticipate that the larger and more complex a pull
request is, the less likely it is that reviewers will accept it,
regardless of its merits.

The WGPU project has had poor experiences with large, complex pull
requests:

- Complex pull requests are difficult to review effectively. It is
  common for us to debug a problem in WGPU and find that it was
  introduced by some massive pull request that we had reviewed and
  accepted, showing that we obviously hadn't understood it as well as
  we'd thought.

- A large, complex pull request obviously represents a significant
  effort on the part of the author. At a personal level, it is quite
  stressful to question its design decisions, knowing that changing
  them will require the author to essentially reimplement the project
  from scratch. Such pull requests make it hard for maintainers to
  uphold their responsibility to keep WGPU maintainable. Incremental
  changes are easier to discuss and revise without drama.

These problems are serious enough that maintainers may choose to
reject large, complex pull requests, regardless of the value of the
feature or the technical merit of the code.

The problem isn't really the *size* of the pull request: a simple
rename, with no changes to functionality, might touch hundreds of
files, but be easy to review. Or, a change to Naga might affect dozens
of snapshot test output files, without being hard to understand.

Rather, the problem is the *complexity* of the pull request: how many
moving pieces does the reviewer need to assess at once? In our
experience, almost every large change can be pared down by separating
out:

- Preparatory refactors that are at least harmless in isolation, and
  perhaps beneficial.

- Helpers and utilities that can be used elsewhere in the code base,
  even if they don't show their full value until the whole thing is
  merged.
  
- Renames and code motion with no semantic effect, like changes to
  types or behavior. When putting these in a separate pull request
  would be awkward, they should at least be segregated into their own
  commits within a pull request.

Brevity for brevity's sake is not the goal. Rather, the goal is to
help the reviewer anticipate the changes' consequences. When a pull
request addresses only a single issue, even if it is textually large,
a trustworthy review becomes more achievable.
