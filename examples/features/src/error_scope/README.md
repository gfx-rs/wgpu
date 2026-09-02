# error_scope

This example demonstrates wgpu's error scopes. It pushes a scope, compiles a
valid shader and then an invalid one, pops the scope to read the first error,
and checks that nested scopes pop in LIFO order.

Nothing is rendered; the results are logged to the console.

## To Run

```
cargo run --bin wgpu-examples error_scope
```
