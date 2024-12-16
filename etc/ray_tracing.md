## Ray Tracing Extensions
`wgpu` supports an experimental version of ray tracing which is subject to change. The extensions allow for acceleration structures to be created and built (with 
`Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE` enabled) and interacted with in shaders. Currently `naga` only supports ray queries
(accessible with `Features::EXPERIMENTAL_RAY_QUERY` enabled in wgpu).

**Note**: The features documented here may have major bugs in them and are expected to be subject
to breaking changes, suggestions for the API exposed by this should be posted on [the ray-tracing issue](https://github.com/gfx-rs/wgpu/issues/1040).
Large changes may mean that this documentation may be out of date.

***This is not*** an introduction to raytracing, and assumes basic prior knowledge, to look at the fundamentals look at 
an [introduction](https://developer.nvidia.com/blog/introduction-nvidia-rtx-directx-ray-tracing/).


### `wgpu`'s raytracing API:
The documentation and specific details of the functions and structures provided
can be found with their definitions.
A `Blas` can be created with `device.create_blas`.
A `Tlas` can be created with `device.create_tlas`.

Unless one is planning on using the unsafe (not recommended for beginners) building API a `Tlas` should be put inside
a `TlasPackage`. After that a reference to the `Tlas` can be retrieved by calling `TlasPackage::tlas`,
this reference can be placed in a bind group to be used in a shader. A reference to a `Blas` can
be used to create `TlasInstance` alongside a transformation matrix, a custom index
(this can be any data that should be given to the shader on a hit) which only the first 24
bits may be set, and a mask to filter hits in the shader.

A `Blas` must be built in either the same build as any `Tlas` it is used to build or an earlier build call.
Before a `Tlas` is used in a shader it must
- have been built
- have all `Blas`es that it was last built with to have last been built in either the same build as
  this `Tlas` or an earlier build call.

### `naga`'s raytracing API:
`naga` supports ray queries (also known as inline raytracing) only. Ray tracing pipelines are currently unsupported.
Naming is mostly taken from vulkan.

Function definitions:

`rayQueryInitialize(rq: ptr<function, ray_query>, acceleration_structure: acceleration_structure, ray_desc: RayDesc)`
- Initializes the `ray_query` to check where (if anywhere) the ray defined by `ray_desc` hits in `acceleration_structure`

`rayQueryProceed(rq: ptr<function, ray_query>) -> bool`
- Traces the ray in the initialized ray_query (partially) through the scene.
- Returns true if a triangle that was hit by the ray was in a `Blas` that is not marked as opaque.
- Returns false if all triangles that were hit by the ray were in `Blas`es that were marked as opaque.
- The hit is considered `Candidate` if this function returns true, and the hit is considered `Committed` if
  this function returns false.
- A `Candidate` intersection interrupts the ray traversal.
- A `Candidate` intersection may happen anywhere along the ray, it should not be relied on to give the closest hit. A 
`Candidate` intersection is to allow the user themselves to decide if that intersection is valid*. If one wants to get
the closest hit a `Committed` intersection should be used.
- Calling this function multiple times will cause the ray traversal to continue if it was interrupted by a `Candidate`
intersection.

`rayQueryGetCommittedIntersection(rq: ptr<function, ray_query>) -> RayIntersection`
- Returns intersection details about a hit considered `Committed`.

`rayQueryGetCandidateIntersection(rq: ptr<function, ray_query>) -> RayIntersection`
- Returns intersection details about a hit considered `Candidate`.

*The API to commit a candidate intersection is not yet implemented but would be possible to be user implemented.

Undefined behavior*:
- Calling `rayQueryGetCommittedIntersection` or `rayQueryGetCandidateIntersection` when `rayQueryProceed` has not been
called on this ray query since it was initialized (or if the ray query has not been previously initialized).
- Calling `rayQueryGetCommittedIntersection` when `rayQueryProceed`'s latest return on this ray query is considered
  `Candidate`. 
- Calling `rayQueryGetCandidateIntersection` when `rayQueryProceed`'s latest return on this ray query is considered
  `Committed`.
- Calling `rayQueryProceed` when `rayQueryInitialize` has not previously been called on this ray query

*this is only known undefined behaviour.

Structure definitions:
````wgsl
struct RayDesc {
    flags: u32,
    cull_mask: u32,
    t_min: f32,
    t_max: f32,
    origin: vec3<f32>,
    dir: vec3<f32>,
}
````
- `flags`: contains flags to use for this ray (e.g. consider all `Blas`es opaque)
- `cull_mask`: if the bitwise and of this and any `TlasInstance`'s `mask` is not zero then the object inside
  the `Blas` contained within that `TlasInstance` may be hit.
- `t_min`: only points on the ray whose t is greater than this may be hit.
- `t_max`: only points on the ray whose t is less than this may be hit.
- `origin`: the origin of the ray.
- `dir`: the direction of the ray, t is calculated as the length down the ray divided by the length of `dir`.

````wgsl
struct RayIntersection {
    kind: u32,
    t: f32,
    instance_custom_index: u32,
    instance_id: u32,
    sbt_record_offset: u32,
    geometry_index: u32,
    primitive_index: u32,
    barycentrics: vec2<f32>,
    front_face: bool,
    object_to_world: mat4x3<f32>,
    world_to_object: mat4x3<f32>,
}
````
- `kind`: the kind of the hit, no other member of this structure is useful if this is equal
  to constant `RAY_QUERY_INTERSECTION_NONE`.
- `t`: see `RayDesc::dir` for calculations.
- `instance_custom_index`: corresponds to `instance.custom_index` where `instance` is the `TlasInstance`
  that the intersected object was contained in.
- `instance_id`: the index into the `TlasPackage` to get the `TlasInstance` that the hit object is in
- `sbt_record_offset`: the offset into the shader binding table. Currently, this value is always 0.
- `geometry_index`: the index into the `Blas`'s build descriptor (e.g. if `BlasBuildEntry::geometry` is
  `BlasGeometries::TriangleGeometries` then it is the index into that contained vector).
- `primitive_index`: the object hit's index into the provided buffer (e.g. if the object is a triangle
  then this is the triangle index)
- `barycentrics`: two of the barycentric coordinates, the third can be calculated (only useful if this is a triangle).
- `front_face`: whether the hit face is the front (only useful if this is a triangle).
- `object_to_world`: matrix for converting from object-space to world-space*
- `world_to_object`: matrix for converting from world-space to object-space*

*These matrices need to be transposed currently otherwise they will not work properly.

Constant definitions:

`const FORCE_OPAQUE = 0x1;`
- When `RayDesc::flags` contains this flag all `Blas`es as being marked as opaque.

`const FORCE_NO_OPAQUE = 0x2;`
- When `RayDesc::flags` contains this flag all `Blas`es as being not marked as opaque.

`const TERMINATE_ON_FIRST_HIT = 0x4;`
- When `RayDesc::flags` contains this flag instead of searching for the closest hit return the first hit.

`const SKIP_CLOSEST_HIT_SHADER = 0x8;`
- Unused: implemented for raytracing pipelines.

`const CULL_BACK_FACING = 0x10;`
- When `RayDesc::flags` contains this flag if `RayIntersection::front_face` is false do not return a hit.

`const CULL_FRONT_FACING = 0x20;`
- When `RayDesc::flags` contains this flag if `RayIntersection::front_face` is true do not return a hit.

`const CULL_OPAQUE = 0x40;`
- When `RayDesc::flags` contains this flag if the `Blas` a intersection is checking is marked as opaque do not return a
  hit.

`const CULL_NO_OPAQUE = 0x80;`
- When `RayDesc::flags` contains this flag if the `Blas` a intersection is checking is not marked as opaque do not return
  a hit.

`const SKIP_TRIANGLES = 0x100;`
- When `RayDesc::flags` contains this flag if the `Blas` a intersection is checking is a triangle containing blas do not
  return a hit.

`const SKIP_AABBS = 0x200;`
- When `RayDesc::flags` contains this flag if the `Blas` a intersection is checking is a AABB containing blas do not
  return a hit.

`const RAY_QUERY_INTERSECTION_NONE = 0;`
- If `RayIntersection::kind` is equal to this the ray hit nothing.

`const RAY_QUERY_INTERSECTION_TRIANGLE = 1;`
- If `RayIntersection::kind` is equal to this the ray hit a triangle.

`const RAY_QUERY_INTERSECTION_GENERATED = 2;`
- If `RayIntersection::kind` is equal to this the ray hit a custom object, this will only happen in a committed
  intersection if a ray which intersected a bounding box for a custom object which was then committed.

`const RAY_QUERY_INTERSECTION_AABB = 3;`
- If `RayIntersection::kind` is equal to this the ray hit a AABB, this will only happen in a candidate intersection if
  the ray intersects the bounding box for a custom object.