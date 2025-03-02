# Review Checklist

This is a collection of notes on things to watch out for when
reviewing pull requests submitted to wgpu and Naga.

Ideally, we want to keep items off this list entirely:

- Using Rust effectively can turn some mistakes into compile-time
  errors. For example, in Naga, using exhaustive matching ensures that
  changes to the IR will cause compile-time errors in any code that
  hasn't been updated.

- Refactoring can gather together all the code responsible for
  enforcing some invariant in one place, making it clear whether a
  change preserves it or not. For example, Naga localizes all handle
  validation to `naga::valid::Validator::validate_module_handles`,
  allowing the rest of the validator to assume that all handles are
  valid.

- Offering custom abstractions can help contributors avoid
  implementing a weaker abstraction by themselves. For example,
  because `HandleSet` and `HandleVec` are used throughout Naga,
  contributors are less likely to write code that uses a `BitSet` or
  `Vec` on handle indices, which would invite bugs by erasing the
  handle types.

This checklist gathers up the concerns that we haven't found a
satisfying way to address in a more robust way.

## Naga

### General

- [ ] If your change iterates over a collection, did you ensure the
      order of iteration was deterministic? Using `HashMap` and
      `HashSet` is fine, as long as you don't iterate over it.
- [ ] If you insert elements into a set or map that you expect are not
      already present, did you make an assertion about `insert`'s
      return value?

### WGSL Extensions

- [ ] If you added a new feature to WGSL that is not covered by the
      WebGPU specification:
  - [ ] Did you add a `Capability` flag for it?
  - [ ] Did you document the feature fully in that flag's doc comment?
  - [ ] Did you ensure the validator rejects programs that use the
        feature unless its capability is enabled?

### IR changes

If your change adds or removes `Handle`s from the IR:
- [ ] Did you update handle validation in `valid::handles`?
- [ ] Did you update the compactor in `compact`?
- [ ] Did you update `back::pipeline_constants::adjust_expr`?

If your change adds a new operation:
- [ ] Did you update the typifier in `proc::typifier`?
- [ ] Did you update the validator in `valid::expression`?
- [ ] If the operation can be used in constant expressions, did you
      update the constant evaluator in `proc::constant_evaluator`?

### Backend changes

- [ ] If your change introduces any new identifiers to generated code,
      how did you ensure they won't conflict with the users'
      identifiers? (This is usually not relevant to the SPIR-V
      backend.)
  - [ ] Did you use the `Namer` to generate a fresh identifier?
  - [ ] Did you register the identifier as a reserved word with the the `Namer`?
  - [ ] Did you use a reserved prefix registered with the `Namer`?
