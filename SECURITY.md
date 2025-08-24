# WGPU Security Policy

This document describes what is considered a security vulnerability in WGPU and
how vulnerabilities should be reported.


## Vulnerability Definition

WebGPU introduces a different threat model than is sometimes applied to
GPU-related software. Unlike typical gaming or high-performance computing
applications, where the software accessing GPU APIs is proprietary or
obtained from a trusted developer, WebGPU makes GPU APIs available to
arbitrary web applications. In the threat model of the web, malicious
content should not be able to use the GPU APIs to access data or interfaces
outside the intended scope for interaction with web content. Therefore, `wgpu`
seeks to prevent undefined behavior and data leaks even when its API is
misused, and failures to do so may be considered vulnerabilities. (This is
also in accordance with the Rust principle of safe vs. unsafe code, since the
`wgpu` library exposes a safe API.)

The WGPU maintainers have discretion in assigning a severity to individual
vulnerabilities. It is generally considered a high-severity vulnerability in
WGPU if JavaScript or WebAssembly code, running with privileges of ordinary web
content in a browser that is using WGPU to provide the WebGPU API to that
content, is able to:

- Access data associated with native applications other than the user agent,
  or associated with other web origins.
- Escape the applicable sandbox and run arbitrary code or call arbitrary system
  APIs on the user agent host.
- Consume system resources to the point that it is difficult to recover
  (e.g. by closing the web page).

The WGPU Rust API offers some functionality, both supported and experimental,
that is not part of the WebGPU standard and is not made available in JavaScript
environments using WGPU. Associated vulnerabilities may be assigned lower
severity than vulnerabilities that apply to a WGPU-based WebGPU implementation
exposed to JavaScript.


## Supported Versions

The WGPU project maintains security support for serious vulnerabilities in the
[most recent major release](https://github.com/gfx-rs/wgpu/releases). Fixes for
security vulnerabilities found shortly after the initial release of a major
version may also be provided for the previous major release.

Mozilla provides security support for versions of WGPU used in [current
versions of Firefox](https://whattrainisitnow.com/).

The version of WGPU that is active can be found in the Firefox repositories:

- [release](https://github.com/mozilla-firefox/firefox/blob/release/gfx/wgpu_bindings/Cargo.toml),
- [beta](https://github.com/mozilla-firefox/firefox/blob/beta/gfx/wgpu_bindings/Cargo.toml), and
- [nightly](https://github.com/mozilla-firefox/firefox/blob/main/gfx/wgpu_bindings/Cargo.toml),

We welcome reports of security vulnerabilities in any of these released
versions or in the latest code on the `trunk` branch.


## Reporting a Vulnerability

Although not all vulnerabilities in WGPU will affect Firefox, Mozilla accepts
all vulnerability reports for WGPU and directs them appropriately. Additionally,
Mozilla serves as the CVE numbering authority for the WGPU project.

To report a security problem with WGPU, create a bug in Mozilla's Bugzilla
instance in the
[Core :: Graphics :: WebGPU](https://bugzilla.mozilla.org/enter_bug.cgi?product=Core&component=Graphics%3A+WebGPU&groups=core-security&groups=gfx-core-security)
component.

**IMPORTANT: For security issues, please make sure that you check the box
labelled "Many users could be harmed by this security problem".** We advise
that you check this option for anything that is potentially
security-relevant, including memory safety, crashes, race conditions, and
handling of confidential information.

Review Mozilla's [guides on bug
reporting](https://bugzilla.mozilla.org/page.cgi?id=bug-writing.html) before
you open a bug.

Mozilla operates a [bug bounty
program](https://www.mozilla.org/en-US/security/bug-bounty/). Some
vulnerabilities in this project may be eligible.
