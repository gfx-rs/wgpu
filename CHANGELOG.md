# Change Log

## v0.4 (TBD)
  - development release for wgpu-0.8
  - API:
    - expressions are explicitly emitted with `Statement::Emit`
    - entry points have inputs in arguments and outputs in the result type
    - `input`/`output` storage classes are gone, but `push_constant` is added
    - `Interpolation` is moved into `Binding::Location` variant
    - real pointer semantics with required `Expression::Load`
    - `TypeInner::ValuePointer` is added
    - image query expressions are added
    - new `Statement::ImageStore`
    - all function calls are `Statement::Call`
    - `GlobalUse` is moved out into processing
    - field layout is controlled by `size` and `alignment` overrides, based on a default layout
    - `Header` is removed
    - entry points are an array instead of a map
  - Infrastructure:
    - control flow uniformity analysis
    - texture-sampler combination gathering
    - `CallGraph` processor is moved out into `glsl` backend
    - `Interface` is removed
    - statement tree and constants are validated
    - code linting is more strict for matches
  - new GraphViz `dot` backend for pretty visualization of the IR
  - `convert` is default a binary target, published with the crate

### v0.3.2 (2021-02-15)
  - fix logical expression types
  - fix _FragDepth_ semantics
  - spv-in:
    - derive block status of structures
  - spv-out:
    - add lots of missing math functions
    - implement discard

### v0.3.1 (2021-01-31)
  - wgsl:
    - support constant array sizes
  - spv-out:
    - fix block decorations on nested structures
    - fix fixed-size arrays
    - fix matrix decorations inside structures
    - implement read-only decorations

## v0.3 (2021-01-30)
  - development release for wgpu-0.7
  - API:
    - math functions
    - type casts
    - updated storage classes
    - updated image sub-types
    - image sampling/loading options
    - storage images
    - interpolation qualifiers
    - early and conservative depth
  - Processors:
    - name manager
    - automatic layout
    - termination analysis
    - validation of types, constants, variables, and entry points

## v0.2 (2020-08-17)
  - development release for wgpu-0.6

## v0.1 (2020-02-26)
  - initial release
