---
name: webgpu-specs
description: Download WebGPU and WGSL specifications for use as a reference
allowed-tools: "Bash(curl -fsSL https://raw.githubusercontent.com/gpuweb/gpuweb/main/spec/index.bs -o .claude/skills/webgpu-specs/webgpu-spec.bs), Bash(curl -fsSL https://raw.githubusercontent.com/gpuweb/gpuweb/main/wgsl/index.bs -o .claude/skills/webgpu-specs/wgsl-spec.bs)"
---

Download the WebGPU specification by running:
```
curl -fsSL https://raw.githubusercontent.com/gpuweb/gpuweb/main/spec/index.bs -o .claude/skills/webgpu-specs/webgpu-spec.bs
```
Search in the webgpu-spec.bs file for relevant sections of the specification.

Download the WGSL specification by running:
```
curl -fsSL https://raw.githubusercontent.com/gpuweb/gpuweb/main/wgsl/index.bs -o .claude/skills/webgpu-specs/wgsl-spec.bs
```
Search in the wgsl-spec.bs file for relevant sections of the specification.

When referencing the specifications, prefer to use named anchors rather than
line numbers. For example, to reference the "Object Descriptors" section, which has the
following header:

```
### Object Descriptors ### {#object-descriptors}
```

Use the URL <https://gpuweb.github.io/gpuweb/#object-descriptors> so the user
can click to navigate directly to that section.

For the WGSL specification, the base URL is <https://gpuweb.github.io/gpuweb/wgsl/>.

If necessary, read additional content from the file to find the header preceding
the text you want to reference. You may provide line numbers as additional
context, but always make every effort to provide the user with a clickable link.
