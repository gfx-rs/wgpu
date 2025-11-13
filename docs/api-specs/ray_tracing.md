# Ray Tracing Extensions

🧪Experimental🧪

`wgpu` supports an experimental version of ray tracing which is subject to change. The extensions allow for acceleration structures to be created and built (with
`Features::EXPERIMENTAL_RAY_QUERY` enabled) and interacted with in shaders. Currently `naga` only supports ray queries
(accessible with `Features::EXPERIMENTAL_RAY_QUERY` enabled in wgpu).

**Note**: The features documented here may have major bugs in them and are expected to be subject
to breaking changes, suggestions for the API exposed by this should be posted on [the ray-tracing issue](https://github.com/gfx-rs/wgpu/issues/1040).
Large changes may mean that this documentation may be out of date.

***This is not*** an introduction to raytracing, and assumes basic prior knowledge, to look at the fundamentals look at
an [introduction](https://developer.nvidia.com/blog/introduction-nvidia-rtx-directx-ray-tracing/).

## `wgpu`'s raytracing API:

The documentation and specific details of the functions and structures provided
can be found with their definitions.

Acceleration structures do not have a separate feature, instead they are enabled by `Features::EXPERIMENTAL_RAY_QUERY`, unlike vulkan.
When ray tracing pipelines are added, that feature will also enable acceleration structures.

A [`Blas`] can be created with [`Device::create_blas`].
A [`Tlas`] can be created with [`Device::create_tlas`].

The [`Tlas`] reference can be placed in a bind group to be used in a shader. A reference to a [`Blas`] can
be used to create [`TlasInstance`] alongside a transformation matrix, a custom index
(this can be any data that should be given to the shader on a hit) which only the first 24
bits may be set, and a mask to filter hits in the shader.

A [`Blas`] must be built in either the same build as any [`Tlas`] it is used to build or an earlier build call.
Before a [`Tlas`] is used in a shader it must
- have been built
- have all [`Blas`]es that it was last built with to have last been built in either the same build as
  this [`Tlas`] or an earlier build call.

### [`Blas`] compaction

Once a [`Blas`] has been built, it can be compacted. Acceleration structures are allocated conservatively, without
knowing the exact data that is inside them. Once a [`Blas`] has been built, the driver can make data specific
optimisations to make the [`BLAS`] smaller. To begin compaction call [`Blas::prepare_compaction_async`] on it. This
method waits until all builds operating on the [`Blas`] are finished, prepares the [`Blas`] to be compacted, and runs
the given callback. To check whether the [`Blas`] is ready, you can also call [`Blas::ready_for_compaction`] instead of
waiting for the callback (useful if you are asynchronously compacting a large number of [`Blas`]es). Submitting a
rebuild of a [`Blas`] terminates any [`Blas::prepare_compaction_async`], preventing the callback from being called, and
making the [`Blas`] no longer ready to compact. Once a [`Blas`] is ready for compaction, it can be compacted using
[`Queue::compact_blas`] this returns the new compacted [`Blas`], which is independent of the [`Blas`] passed in. The
other [`Blas`] can be used for other things, including being rebuilt without affecting the new [`Blas`]. The returned
[`Blas`] behaves largely like the [`Blas`] it was created from, except that it can be neither rebuilt, nor compacted
again.

An example of compaction being run when [`Blas`]es are ready, this would be in a situation when memory was not a major
problem, otherwise (e.g. if you get an out of memory error) you should compact immediately (and switching all
non-compacted [`Blas`]es to compacted ones).
```rust
use std::iter;
use wgpu::Blas;

struct BlasToBeCompacted {
    blas: Blas,
    /// The index into the TlasInstance this BLAS is used in.
    tlas_index: usize,
}

fn render(/*whatever args you need to render*/) {
  /* additional code to prepare the renderer */ 
  //An iterator of whatever BLASes you have called `prepare_compaction_async` on.
  let blas_s_pending_compaction: impl Iterator<Item = BlasToBeCompacted> = iter::empty();
  for blas_to_be_compacted in blas_s_pending_compaction {
    if blas_to_be_compacted.blas.ready_for_compaction() {
        let compacted_blas = queue.compact_blas(&blas_to_be_compacted.blas);
        tlas_instance[blas_to_be_compacted.tlas_index].set_blas(&compacted_blas);
    }
  }
  let mut encoder =
    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
  /* do other preparations on th TlasInstance.*/
  encoder.build_acceleration_structures(iter::empty(), iter::once(&tlas_package));
  /* more render code */
  queue.submit(Some(encoder.finish()));
}
```

[`Device::create_blas`]: https://wgpu.rs/doc/wgpu/struct.Device.html#method.create_blas
[`Device::create_tlas`]: https://wgpu.rs/doc/wgpu/struct.Device.html#method.create_tlas
[`Tlas`]: https://wgpu.rs/doc/wgpu/struct.Tlas.html
[`Blas`]: https://wgpu.rs/doc/wgpu/struct.Blas.html
[`TlasInstance`]: https://wgpu.rs/doc/wgpu/struct.TlasInstance.html
[`Blas::prepare_compaction_async`]: https://wgpu.rs/doc/wgpu/struct.Blas.html#method.prepare_compaction_async
[`Blas::ready_for_compaction`]: https://wgpu.rs/doc/wgpu/struct.Blas.html#method.ready_for_compaction
[`Queue::compact_blas`]: https://wgpu.rs/doc/wgpu/struct.Queue.html#method.compact_blas

## `naga`'s raytracing API:

`naga` supports ray queries (also known as inline raytracing) only. Ray tracing pipelines are currently unsupported.
Naming is mostly taken from vulkan.

```wgsl
// - Initializes the `ray_query` to check where (if anywhere) the ray defined by `ray_desc` hits in `acceleration_structure
rayQueryInitialize(rq: ptr<function, ray_query>, acceleration_structure: acceleration_structure, ray_desc: RayDesc)
// Overload.
rayQueryInitialize(rq: ptr<function, ray_query<vertex_return>>, acceleration_structure: acceleration_structure<vertex_return>, ray_desc: RayDesc)

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
rayQueryProceed(rq: ptr<function, ray_query>) -> bool
// Overload.
rayQueryProceed(rq: ptr<function, ray_query<vertex_return>>) -> bool

// - Generates a hit from procedural geometry at a particular distance.
rayQueryGenerateIntersection(hit_t: f32)

// - Commits a hit from triangular non-opaque geometry.
rayQueryConfirmIntersection()

// - Aborts the query.
rayQueryTerminate()

// - Returns intersection details about a hit considered `Committed`.
rayQueryGetCommittedIntersection(rq: ptr<function, ray_query>) -> RayIntersection
// Overload.
rayQueryGetCommittedIntersection(rq: ptr<function, ray_query<vertex_return>>) -> RayIntersection

// - Returns intersection details about a hit considered `Candidate`.
rayQueryGetCandidateIntersection(rq: ptr<function, ray_query>) -> RayIntersection
// Overload.
rayQueryGetCandidateIntersection(rq: ptr<function, ray_query<vertex_return>>) -> RayIntersection

// - Returns the vertices of the hit triangle considered `Committed`.
getCommittedHitVertexPositions(rq: ptr<function, ray_query<vertex_return>>) -> array<vec3<f32>, 3>

// - Returns the vertices of the hit triangle considered `Candidate`.
getCandidateHitVertexPositions(rq: ptr<function, ray_query<vertex_return>>) -> array<vec3<f32>, 3>
```

> [!CAUTION]
>
> ### ⚠️Undefined behavior ⚠️:
> - Calling `rayQueryGetCommittedIntersection` or `rayQueryGetCandidateIntersection` when `rayQueryProceed` has not been
> called on this ray query since it was initialized (or if the ray query has not been previously initialized).
> - Calling `rayQueryGetCommittedIntersection` when `rayQueryProceed`'s latest return on this ray query is considered
>   `Candidate`.
> - Calling `rayQueryGetCandidateIntersection` when `rayQueryProceed`'s latest return on this ray query is considered
>   `Committed`.
> - Calling `getCommittedHitVertexPositions` when `rayQueryProceed`'s latest return on this ray query is considered
>   `Candidate`.
> - Calling `getCandidateHitVertexPositions` when `rayQueryProceed`'s latest return on this ray query is considered
>   `Committed`.
> - Calling `get*HitVertexPositions` when the last `rayQueryProceed` did not hit a triangle
> - Calling `rayQueryProceed` when `rayQueryInitialize` has not previously been called on this ray query
> - Calling `rayQueryGenerateIntersection` on a query with last intersection kind not being
>   `RAY_QUERY_INTERSECTION_AABB`,
> - Calling `rayQueryGenerateIntersection` with `hit_t` outside of `RayDesc::t_min .. RayDesc::t_max` range.
>   or when `rayQueryProceed`'s latest return on this ray query is not considered `Candidate`.
> - Calling `rayQueryConfirmIntersection` on a query with last intersection kind not being
>   `RAY_QUERY_INTERSECTION_TRIANGLE`,
>   or when `rayQueryProceed`'s latest return on this ray query is not considered `Candidate`.
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
    // Corresponds to `instance.custom_data` where `instance` is the `TlasInstance`
    // that the intersected object was contained in.
    instance_custom_data: u32,
    // The index into the `TlasPackage` to get the `TlasInstance` that the hit object is in
    instance_index: u32,
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
