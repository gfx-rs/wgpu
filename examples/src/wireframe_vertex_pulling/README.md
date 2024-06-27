# Wireframe Vertex Pulling

This example renders a cube in one of:
 - Solid
 - Points
 - Wireframe
 - Wireframe with thick lines

This example is ported from [WebGPU native examples](https://github.com/samdauwe/webgpu-native-examples/tree/master): [Wireframe Vertex Pulling](https://github.com/samdauwe/webgpu-native-examples/blob/master/src/examples/wireframe_vertex_pulling.c#L598)

## To Run

### Solid
```
cargo run --bin wgpu-examples wireframe_vertex_pulling solid
```

### Points
```
cargo run --bin wgpu-examples wireframe_vertex_pulling points
```

### Wireframe
```
cargo run --bin wgpu-examples wireframe_vertex_pulling wireframe
```

### Wireframe Thick
```
cargo run --bin wgpu-examples wireframe_vertex_pulling wireframe-thick
```

## Screenshots

### Solid
![Solid example](./solid.png)

### Points
![Points example](./points.png)

### Wireframe
![Wireframe example](./wireframe.png)

### Wireframe with thick lines
![Wireframe Thick example](./wireframe_thick.png)
