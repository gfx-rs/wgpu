# wgpu-info

This is a command line utility that does two different functions. 

#### Listing Adapters

When called with no arguments, wgpu-info will list all adapters visible to wgpu and all the information about them we have.

For OpenGL on platforms other than Linux add the `angle` feature, for Vulkan on macOS add the `vulkan-portability` feature.

```
cargo run --bin wgpu-info
```

#### Running Test on many Adapters

When called with any amount of arguments, it will interpret all of the arguments as a command to run. It will run this command N different times, one for every combination of adapter and backend on the system.

For every command invocation, it will set `WGPU_ADAPTER_NAME` to the name of the adapter name and `WGPU_BACKEND` to the name of the backend. This is used as the primary means of testing across many adapters.
