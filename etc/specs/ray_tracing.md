# Ray Tracing Extensions

🧪Experimental🧪

`wgpu` supports an experimental version of ray tracing which is subject to change. The extensions allow for acceleration structures to be created and built (with 
`Features::EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE` enabled) and interacted with in shaders. Currently `naga` only supports ray queries
(accessible with `Features::EXPERIMENTAL_RAY_QUERY` enabled in wgpu).

**Note**: The features documented here may have major bugs in them and are expected to be subject
to breaking changes, suggestions for the API exposed by this should be posted on [the ray-tracing issue](https://github.com/gfx-rs/wgpu/issues/1040).
Large changes may mean that this documentation may be out of date.

***This is not*** an introduction to raytracing, and assumes basic prior knowledge, to look at the fundamentals look at 
an [introduction](https://developer.nvidia.com/blog/introduction-nvidia-rtx-directx-ray-tracing/).

## `wgpu`'s raytracing API:

The documentation and specific details of the functions and structures provided
can be found with their definitions.  

A [`Blas`] can be created with [`Device::create_blas`].  
A [`Tlas`] can be created with [`Device::create_tlas`].

Unless one is planning on using the unsafe building API (not recommended for beginners) a [`Tlas`] should be put inside
a [`TlasPackage`]. After that a reference to the [`Tlas`] can be retrieved by calling [`TlasPackage::tlas`].
This reference can be placed in a bind group to be used in a shader. A reference to a [`Blas`] can
be used to create [`TlasInstance`] alongside a transformation matrix, a custom index
(this can be any data that should be given to the shader on a hit) which only the first 24
bits may be set, and a mask to filter hits in the shader.

A [`Blas`] must be built in either the same build as any [`Tlas`] it is used to build or an earlier build call.
Before a [`Tlas`] is used in a shader it must
- have been built
- have all [`Blas`]es that it was last built with to have last been built in either the same build as
  this [`Tlas`] or an earlier build call.

[`Device::create_blas`]: https://wgpu.rs/doc/wgpu/struct.Device.html#method.create_blas
[`Device::create_tlas`]: https://wgpu.rs/doc/wgpu/struct.Device.html#method.create_tlas
[`Tlas`]: https://wgpu.rs/doc/wgpu/struct.Tlas.html
[`Blas`]: https://wgpu.rs/doc/wgpu/struct.Blas.html
[`TlasInstance`]: https://wgpu.rs/doc/wgpu/struct.TlasInstance.html
[`TlasPackage`]: https://wgpu.rs/doc/wgpu/struct.TlasPackage.html
[`TlasPackage::tlas`]: https://wgpu.rs/doc/wgpu/struct.TlasPackage.html#method.tlas

## `naga`'s raytracing API:

`naga` supports ray queries (also known as inline raytracing) only. Ray tracing pipelines are currently unsupported.
Naming is mostly taken from vulkan.

```wgsl
// - Initializes the `ray_query` to check where (if anywhere) the ray defined by `ray_desc` hits in `acceleration_structure
rayQueryInitialize(rq: ptr<function, ray_query>, acceleration_structure: acceleration_structure, ray_desc: RayDesc)

// - Traces the ray in the initialized ray_query (partially) through the scene.
// - Returns true if a triangle that was hit by the ray was in a `Blas` that is not marked as opaque.
// - Returns false if all triangles that were hit by the ray were in `Blas`es that were marked as opaque.
// - The hit is considered `Candidate` if this function returns true, and the hit is considered `Committed` if
//   this function returns false.
// - A `Candidate` intersection interrupts the ray traversal.
// - A `Candidate` intersection may happen anywhere along the ray, it should not be relied on to give the closest hit. A 
//   `Candidate` intersection is to allow the user themselves to decide if that intersection is valid*. If one wants to get
//   the closest hit a `Committed` intersection should be used.
// - Calling this function multiple times will cause the ray traversal to continue if it was interrupted by a `Candidate`
//   intersection.
rayQueryProceed(rq: ptr<function, ray_query>) -> bool`

// - Returns intersection details about a hit considered `Committed`.
rayQueryGetCommittedIntersection(rq: ptr<function, ray_query>) -> RayIntersection

// - Returns intersection details about a hit considered `Candidate`.
rayQueryGetCandidateIntersection(rq: ptr<function, ray_query>) -> RayIntersection
```

*The API to commit a candidate intersection is not yet implemented but would be possible to be user implemented.

> [!CAUTION]
> 
> ### ⚠️Undefined behavior ⚠️:
> - Calling `rayQueryGetCommittedIntersection` or `rayQueryGetCandidateIntersection` when `rayQueryProceed` has not been
> called on this ray query since it was initialized (or if the ray query has not been previously initialized).
> - Calling `rayQueryGetCommittedIntersection` when `rayQueryProceed`'s latest return on this ray query is considered
>   `Candidate`. 
> - Calling `rayQueryGetCandidateIntersection` when `rayQueryProceed`'s latest return on this ray query is considered
>   `Committed`.
> - Calling `rayQueryProceed` when `rayQueryInitialize` has not previously been called on this ray query
> 
> *this is only known undefined behaviour, and will be worked around in the future.

```wgsl
struct RayDesc {
    // Contains flags to use for this ray (e.g. consider all `Blas`es opaque)
    flags: u32,
    // If the bitwise and of this and any `TlasInstance`'s `mask` is not zero then the object inside
    // the `Blas` contained within that `TlasInstance` may be hit.
    cull_mask: u32,
    // Only points on the ray whose t is greater than this may be hit.
    t_min: f32,
    // Only points on the ray whose t is less than this may be hit.
    t_max: f32,
    // The origin of the ray.
    origin: vec3<f32>,
    // The direction of the ray, t is calculated as the length down the ray divided by the length of `dir`.
    dir: vec3<f32>,
}

struct RayIntersection {
    // the kind of the hit, no other member of this structure is useful if this is equal
    // to constant `RAY_QUERY_INTERSECTION_NONE`.
    kind: u32,
    // Distance from starting point, measured in units of `RayDesc::dir`.
    t: f32,
    // Corresponds to `instance.custom_index` where `instance` is the `TlasInstance`
    // that the intersected object was contained in.
    instance_custom_index: u32,
    // The index into the `TlasPackage` to get the `TlasInstance` that the hit object is in
    instance_id: u32,
    // The offset into the shader binding table. Currently, this value is always 0.
    sbt_record_offset: u32,
    // The index into the `Blas`'s build descriptor (e.g. if `BlasBuildEntry::geometry` is
    // `BlasGeometries::TriangleGeometries` then it is the index into that contained vector).
    geometry_index: u32,
    // The object hit's index into the provided buffer (e.g. if the object is a triangle
    // then this is the triangle index)
    primitive_index: u32,
    // Two of the barycentric coordinates, the third can be calculated (only useful if this is a triangle).
    barycentrics: vec2<f32>,
    // Whether the hit face is the front (only useful if this is a triangle).
    front_face: bool,
    // Matrix for converting from object-space to world-space.
    //
    // This matrix needs to be on the left side of the multiplication. Using it the other way round will not work.
    // Use it this way: `let transformed_vector = intersecion.object_to_world * vec4<f32>(x, y, z, transform_multiplier);
    object_to_world: mat4x3<f32>,
    // Matrix for converting from world-space to object-space
    //
    // This matrix needs to be on the left side of the multiplication. Using it the other way round will not work.
    // Use it this way: `let transformed_vector = intersecion.world_to_object * vec4<f32>(x, y, z, transform_multiplier);
    world_to_object: mat4x3<f32>,
}

/// -- Flags for `RayDesc::flags` --

// All `Blas`es are marked as opaque.
const FORCE_OPAQUE = 0x1;

// All `Blas`es are marked as non-opaque.
const FORCE_NO_OPAQUE = 0x2;

// Instead of searching for the closest hit return the first hit.
const TERMINATE_ON_FIRST_HIT = 0x4;

// Unused: implemented for raytracing pipelines.
const SKIP_CLOSEST_HIT_SHADER = 0x8;

// If `RayIntersection::front_face` is false do not return a hit.
const CULL_BACK_FACING = 0x10;

// If `RayIntersection::front_face` is true do not return a hit.
const CULL_FRONT_FACING = 0x20;

// If the `Blas` a intersection is checking is marked as opaque do not return a hit.
const CULL_OPAQUE = 0x40;

// If the `Blas` a intersection is checking is not marked as opaque do not return a hit.
const CULL_NO_OPAQUE = 0x80;

// If the `Blas` a intersection is checking contains triangles do not return a hit.
const SKIP_TRIANGLES = 0x100;

// If the `Blas` a intersection is checking contains AABBs do not return a hit.
const SKIP_AABBS = 0x200;

/// -- Constants for `RayIntersection::kind` --

// The ray hit nothing.
const RAY_QUERY_INTERSECTION_NONE = 0;

// The ray hit a triangle.
const RAY_QUERY_INTERSECTION_TRIANGLE = 1;

// The ray hit a custom object, this will only happen in a committed intersection
// if a ray which intersected a bounding box for a custom object which was then committed.
const RAY_QUERY_INTERSECTION_GENERATED = 2;

// The ray hit a AABB, this will only happen in a candidate intersection
// if the ray intersects the bounding box for a custom object.
const RAY_QUERY_INTERSECTION_AABB = 3;
```
