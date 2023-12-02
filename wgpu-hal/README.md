*wgpu-hal* is an explicit low-level GPU abstraction powering *wgpu-core*.
It's a spiritual successor to [gfx-hal](https://github.com/gfx-rs/gfx),
but with reduced scope, and oriented towards WebGPU implementation goals.

It has no overhead for validation or tracking, and the API translation overhead is kept to the bare minimum by the design of WebGPU.
This API can be used for resource-demanding applications and engines.

# Usage notes

All of the API is `unsafe`. Documenting the exact safety requirements for the
state and function arguments is desired, but will likely be incomplete while the library is in early development.

The returned errors are only for cases that the user can't anticipate,
such as running out-of-memory, or losing the device.
For the counter-example, there is no error for mapping a buffer that's not mappable.
As the buffer creator, the user should already know if they can map it.

The API accepts iterators in order to avoid forcing the user to store data in particular containers. The implementation doesn't guarantee that any of the iterators are drained, unless stated otherwise by the function documentation.
For this reason, we recommend that iterators don't do any mutating work.

# Debugging

Most of the information in https://github.com/gfx-rs/wgpu/wiki/Debugging-wgpu-Applications still applies to this API, with an exception of API tracing/replay functionality, which is only available in *wgpu-core*.
