# Managing Cargo dependencies

In general, `wgpu` tries to stay up-to-date with the current Rust ecosystem; we are interested in keeping the version of dependency crates we use up-to-date. However, there are some considerations to keep in mind when upgrading or adding dependencies to `wgpu`.

## Adding or upgrading dependencies may cause dependency duplication downstream

Some crates in the Rust ecosystem are useful for both `wgpu` and its downstream consumers. Popular crates—like those in the `serde` ecosystem, for example—are often used in multiple places in a Rust crate's dependency tree. Normally, Cargo can unify dependencies behind a single SemVer-compatible version. When one or more, but not all, of these places consume a [SemVer-incompatible upgrade][semver-summary], Cargo uses both in the new dependency tree (see The Cargo Book's section titled [`Version-incompatibility hazards`](https://doc.rust-lang.org/cargo/reference/resolver.html#version-incompatibility-hazards)). These are often referred to as "duplicate dependencies"[^1].

[semver-summary]: https://semver.org/spec/v2.0.0.html#summary

[^1]: One can detect such duplicates with [`cargo tree --duplicates`](https://doc.rust-lang.org/cargo/commands/cargo-tree.html#option-cargo-tree---duplicates).

There exist many Rust projects that, for some reason or another, are sensitive to duplicate dependencies. The most common reason for this is to minimize the number of built dependencies, but there are other reasons for doing so, like projects that prefer to [vendor in their dependencies' source code](https://doc.rust-lang.org/cargo/commands/cargo-vendor.html) and keep diffs easy to inspect. **When a dependency of `wgpu` releases a SemVer-incompatible change and `wgpu` consumes it (even if it's new _to `wgpu`_), it may cause conflicts for projects that avoid duplicate dependencies.**

> [!TIP]
> Downstreams like Firefox can and do work around Cargo's (well-motivated) refusal to unify SemVer-incompatible versions of crates, but this happens only on a case-by-case basis. In general, this requires source compatibility for the APIs actually consumed between both versions.

## Firefox needs to audit every dependency

In addition to the above, Firefox has unusual sensitivities to updates because of required auditing via [`cargo-vet`](https://github.com/mozilla/cargo-vet). **Firefox contributors _cannot_ bring in a Rust dependency—including new versions of dependencies already in use—without an audit indicating that the dependency meets [`safe-to-deploy`] criteria for _all_ of its feature combinations.** One can ask the following questions to determine whether an upgrade is feasible for Firefox; "yes" answers are significantly easier cases to deal with:

1. If the dependency had to be audited for [`safe-to-deploy`], would it take less than an hour to review?

   <sup>One can use <https://diff.rs/> to quickly gauge the answer to this question for upgrades. Example: [`serde` 1.0.218 → 1.0.219](https://diff.rs/serde/1.0.218/1.0.219/src%2Flib.rs)</sup>

1. Does the same code work between the two dependency's versions without needing to be changed (viz., are they "source-compatible")?

   <sup>NOTE: This is somewhat different from [SemVer compatibility][semver-summary], because it deals with code already written, rather than SemVer's dealing with code that _might_ be written.</sup>

1. Does the dependency already have an audit in Mozilla's [`supply-chain/audits.toml`](https://searchfox.org/mozilla-central/source/supply-chain/audits.toml) or in the other audit databases it `import`s in [`supply-chain/config.toml`](https://searchfox.org/mozilla-central/source/supply-chain/config.toml)?

[`safe-to-deploy`]: https://mozilla.github.io/cargo-vet/built-in-criteria.html#safe-to-deploy

> [!TIP]
> [Searchfox](https://searchfox.org) can be used to answer question (3) and other simple questions with just a web browser. For example:
>
> - "Does Firefox use the `itertools` crate?" can be answered by a case-sensitive search scoped to files with the `rs` extension: https://searchfox.org/mozilla-central/search?q=itertools&path=*.rs&case=true&regexp=false
> - "Is `serde` 1.0.143 already audited?" can be answered by examining `supply-chain/audits.toml` ([permalink](https://searchfox.org/mozilla-central/rev/f3c8c63a097b61bb1f01e13629b9514e09395947/supply-chain/audits.toml#4730-4733)).

> [!TIP]
> For more complicated questions like, "What features does Mozilla use in the `hashbrown` crate?" the best known way to satisfy the question is using Cargo's CLI, i.e., `cargo tree` or `cargo metadata`, in tandem with editing Cargo manifests directly. A Git mirror of Firefox's source code can be found at [`mozilla/gecko-dev`](https://github.com/mozilla/gecko-dev).
>
> A full checkout of `gecko-dev` fills multiple gigabytes, and is not recommended to make one locally unless you plan on doing work directly on Firefox. However, we can optimize to somewhat less than 1 GiB of storage, and speed up the checkout process by using `git sparse-checkout`:
>
> ```sh
> git clone --sparse --depth=1 'https://github.com/mozilla/gecko-dev.git'
> cd gecko-dev
> git sparse-checkout set --no-cone
> git sparse-checkout add \
>   '**/Cargo.toml' \
>   '**/lib.rs' \
>   '**/main.rs' \
>   '**/src/bin/*.rs' \
>   ;
> ```
>
> From here, you can invoke Cargo with the `--locked` flag, i.e:
>
> ```sh
> cargo --locked tree -i itertools
> ```
>
> ```sh
> cargo --locked metadata --format-version=1
> ```
