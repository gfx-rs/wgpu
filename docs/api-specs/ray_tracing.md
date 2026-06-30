# Ray Tracing Extensions

🧪Experimental🧪

`wgpu` supports an experimental version of ray tracing which is subject to change. The extensions allow for acceleration structures to be created and built (with
`Features::EXPERIMENTAL_RAY_QUERY` enabled) and interacted with in shaders. Currently `naga` only supports ray queries
(accessible with `Features::EXPERIMENTAL_RAY_QUERY` enabled in wgpu).

**Note**: The features documented here may have major bugs in them and are expected to be subject
to breaking changes, suggestions for the API exposed by this should be posted on [the ray-tracing issue](https://github.com/gfx-rs/wgpu/issues/1040).
Large changes may mean that this documentation may be out of date.

**_This is not_** an introduction to raytracing, and assumes basic prior knowledge, to look at the fundamentals look at
an [introduction](https://developer.nvidia.com/blog/introduction-nvidia-rtx-directx-ray-tracing/).

## `wgpu`'s raytracing API:

The documentation and specific details of the functions and structures provided
can be found with their definitions.

Acceleration structures do not have a separate feature, instead they are enabled by `Features::EXPERIMENTAL_RAY_QUERY`, unlike vulkan.
When ray tracing pipelines are added, that feature will also enable acceleration structures.

A [`Blas`] can be created with [`Device::create_blas`].
A [`Tlas`] can be created with [`Device::create_tlas`].

The [`Tlas`] reference can be placed in a bind group to be used in a shader. A reference to a [`Blas`] can
be used to create [`TlasInstance`] alongside a transformation matrix, custom data
(this can be any data that should be given to the shader on a hit) which only the first 24
bits may be set, and a mask to filter hits in the shader.

A [`Blas`] must be built in either the same build as any [`Tlas`] it is used to build or an earlier build call.
Before a [`Tlas`] is used in a shader it must

- have been built
- have all [`Blas`]es that it was last built with to have last been built in either the same build as
  this [`Tlas`] or an earlier build call.

### [`Blas`]es and [`Tlas`]es

Both acceleration structures are opaque objects. These objects typically (but are not guaranteed to)
contain a tree-like structure of objects that are quick to be intersected with rays (as of 2026, these
are usually axis-aligned bounding boxes) and contain two or more "branches" (these objects) and "leaves".
For a [`Tlas`] these "leaves" are references to the [`Blas`] (not copies, hence a [`Tlas`] must be rebuild after
any [`Blas`] it is storing is modified). For a [`Blas`] with triangle geometry the "leaves" are triangles. For
a [`Blas`] with AABBs, the "leaves" might not exist or might contain only a small amount of information. This is
because there is no requirement to store the AABBs as this might require a third intersection type. However,
the "branches" will probably be fairly closely approximating the AABBs to maintain fairly good trace performance.

#### Building

Building an acceleration structure is a slow operation as it is likely to require building a tree-like structure.
This happens on the GPU and can be encoded using [`CommandEncoder::build_acceleration_structures`].

For [`Blas`]es with triangle geometry, this will copy the triangles out of the vertex buffer (using indexing if
provided), multiply them by the geometry transform matrix (if provided), and store a representation of this
somewhere in the structure to be traced against (and computes surrounding objects for sets of triangles, repeating
the process for objects of the surrounding objects etc. if the implementation uses a tree-like structure).

For [`Blas`]es with AABB geometry, this will construct object(s) in such a way that all possible rays will generate
a candidate intersection that would have generated a candidate intersection with the AABBs. Note that this may be
(though is very unlikely to be) a volume containing all space. (These may then have objects constructed around them
if the implementation has a tree-like structure.)

For [`Tlas`]es, these will store references to the [`Blas`]es within the [`TlasInstance`]s as well as transforms,
masks, and any other information required. (These [`Blas`]es then may have objects constructed around their transformed
positions if the implementation uses a tree-like structure.) `wgpu` will store references to the [`Blas`]es to keep
them within device memory and alive while the [`Tlas`] uses them. However, if you use functions to build it using
the underlying functions (e.g. by using `as_hal` functions), you are responsible for keeping the [`Blas`]es alive.

Some memory is allocated when building to be "scratch" data (a temporary buffer used by the GPU to store data during
the build) and instance staging memory (to copy the instances to the GPU). The some of this can be reused for between
the [`Blas`] and [`Tlas`] builds so in general it is advisable to try and integrate the builds together. Because
building is slow, it should be done as few times as possible. For moving geometry (players, particles, etc.), use
[`PREFER_FAST_BUILD`](https://wgpu.rs/doc/wgpu_hal/struct.AccelerationStructureBuildFlags.html#associatedconstant.PREFER_FAST_BUILD)
to speed up builds. Updating (performing a partial rebuild) is currently unsupported, but may be implemented in the
future. You should compact any geometry which will not change (e.g. static level geometry) once it has been built.

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
  /* do other preparations on the TlasInstance.*/
  encoder.build_acceleration_structures(iter::empty(), iter::once(&tlas_package));
  /* more render code */
  queue.submit([encoder.finish()]);
}
```

[`Device::create_blas`]: https://wgpu.rs/doc/wgpu/struct.Device.html#method.create_blas
[`CommandEncoder::build_acceleration_structures`]: https://wgpu.rs/doc/wgpu/struct.CommandEncoder.html#method.build_acceleration_structures
[`Device::create_tlas`]: https://wgpu.rs/doc/wgpu/struct.Device.html#method.create_tlas
[`Tlas`]: https://wgpu.rs/doc/wgpu/struct.Tlas.html
[`Blas`]: https://wgpu.rs/doc/wgpu/struct.Blas.html
[`TlasInstance`]: https://wgpu.rs/doc/wgpu/struct.TlasInstance.html
[`Blas::prepare_compaction_async`]: https://wgpu.rs/doc/wgpu/struct.Blas.html#method.prepare_compaction_async
[`Blas::ready_for_compaction`]: https://wgpu.rs/doc/wgpu/struct.Blas.html#method.ready_for_compaction
[`Queue::compact_blas`]: https://wgpu.rs/doc/wgpu/struct.Queue.html#method.compact_blas

## `naga`'s raytracing API:

`naga` supports ray queries (also known as inline raytracing). To enable basic ray query functions you must add
`enable wgpu_ray_query` to the shader, ray queries and acceleration structures also support tags which require extra
`enable` extensions (see Acceleration structure tags for more info). Ray tracing pipelines are currently in
development. Naming is mostly taken from vulkan.

### Ray Queries

```wgsl
// - Initializes the `ray_query` to check where (if anywhere) the ray defined by `ray_desc` hits in `acceleration_structure`
// - If `ray_desc` is "invalid" (see definition on struct) then either:
//    1. the call is discarded (see behaviour of other calls if the ray query is uninitialized)
//    2. the call behaves *as if* the minimum number of fields were changed to make `ray_desc` valid
// - A ray query is "initialized" if this function has been called *and* was not discarded. If the call was discarded,
//   the ray query may either be treated as if it were uninitialized or left in its previous state.
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
// - If `rq` was not previously initialized, the call is discarded.
// - If the previous proceed returned false (and no initialize was performed in between), the call is discarded.
rayQueryProceed(rq: ptr<function, ray_query>) -> bool
// Overload.
rayQueryProceed(rq: ptr<function, ray_query<vertex_return>>) -> bool

// - Generates a hit from procedural geometry at a particular distance.
// - If `rq` was not previously initialized, the call is discarded.
// - If the previous proceed returned false (and no initialize was performed in between), the call is discarded.
// - If `hit_t` is not between the `ray_desc.tmin` provided to the previous initialize and the `t` value of the
//   latest committed hit (or the last `ray_desc.tmax` if there are no committed hits), the call is discarded.
rayQueryGenerateIntersection(rq: ptr<function, ray_query>, hit_t: f32)

// - Commits a hit from triangular non-opaque geometry.
// - If `rq` was not previously initialized, the call is discarded.
// - If the previous proceed returned false (and no initialize was performed in between), the call is discarded.
rayQueryConfirmIntersection(rq: ptr<function, ray_query>)

// Aborts the query which is in progress, that is, the next `rayQueryProceed` is guaranteed to return `false`
// and any call to `rayQueryGetCommittedIntersection` after the `rayQueryProceed` will return the closest
// committed result so far. (`rayQueryProceed` must be called for `rayQueryGetCommittedIntersection` to return
// a non zeroed value).
//
// - If `rq` was not previously initialized, the call is discarded.
// - If the previous proceed returned false (and no initialize was performed in between), the call is discarded.
rayQueryTerminate(rq: ptr<function, ray_query>)

// - Returns intersection details about a hit considered `Committed`.
//
// Depending on what type is hit, different fields will be populated. `RayIntersection::kind` is always populated
// with the kind of hit.
// - The following fields are populated if the closest hit was any object.
//    - t
//    - instance_custom_data
//    - instance_index
//    - sbt_record_offset
//    - geometry_index
//    - primitive_index
//    - object_to_world
//    - world_to_object
// - The following fields are populated if the closest hit is was a triangle.
//    - barycentrics
//    - front_face
// If `rq` was not previously initialized, the call is discarded.
// If the previous proceed returned true or there were no previous proceed calls since the last initialize, a
//  zero-initialized `RayIntersection` is returned.
rayQueryGetCommittedIntersection(rq: ptr<function, ray_query>) -> RayIntersection
// Overload.
rayQueryGetCommittedIntersection(rq: ptr<function, ray_query<vertex_return>>) -> RayIntersection

// - Returns intersection details about a hit considered `Candidate`.
//
// Depending on what type is hit, different fields will be populated. `RayIntersection::kind` is always populated
// with the kind of hit.
// - The following fields are populated if the closest hit was any object.
//    - instance_custom_data
//    - instance_index
//    - sbt_record_offset
//    - geometry_index
//    - primitive_index
//    - object_to_world
//    - world_to_object
// - The following fields are populated if the closest hit is was a triangle.
//    - t
//    - barycentrics
//    - front_face
//
// Note that `t` is *only* returned for a candidate triangle intersection. This is because
// `RAY_QUERY_INTERSECTION_AABB` is an AABB which has a volume.
// If `rq` was not previously initialized, the call is discarded.
// If the previous proceed returned false or there were no previous proceed calls since the last initialize, a
// zero-initialized `RayIntersection` is returned.
rayQueryGetCandidateIntersection(rq: ptr<function, ray_query>) -> RayIntersection
// Overload.
rayQueryGetCandidateIntersection(rq: ptr<function, ray_query<vertex_return>>) -> RayIntersection

// - Returns the vertices of the hit triangle considered `Committed`.
// - If `rq` was not previously initialized, the call is discarded.
// - If the previous proceed returned true or there were no previous proceed calls since the last initialize, a
//   zero-initialized `array` is returned.
// - If the last hit was not a triangle, a zero-initialized `array` is returned.
getCommittedHitVertexPositions(rq: ptr<function, ray_query<vertex_return>>) -> array<vec3<f32>, 3>

// - Returns the vertices of the hit triangle considered `Candidate`.
// - If `rq` was not previously initialized, the call is discarded.
// - If the previous proceed returned true or there were no previous proceed calls since the last initialize, a
//   zero-initialized `array` is returned.
// - If the last hit was not a triangle, a zero-initialized `array` is returned.
getCandidateHitVertexPositions(rq: ptr<function, ray_query<vertex_return>>) -> array<vec3<f32>, 3>

// A `RayDesc` is invalid if:
// - `ray_desc.flags` contains more than one of `SKIP_TRIANGLES` and `SKIP_AABBS`
// - `ray_desc.flags` contains more than one of `SKIP_TRIANGLES`, `CULL_BACK_FACING` and `CULL_FRONT_FACING`
// - `ray_desc.flags` contains more than one of `FORCE_OPAQUE`, `FORCE_NO_OPAQUE`, `CULL_OPAQUE`, `CULL_NO_OPAQUE`
// - `ray_desc.t_min` is less than `0.0`, or is not finite (`NaN` or `Inf`).
// - `ray_desc.t_max` is less than `0.0`, less than `ray_desc.t_min` (the first rule is implied by the second).
// - Any component of `ray_desc.origin` or `ray_desc.dir` is not finite (`NaN` or `Inf`)
// - `ray_desc.dir`'s length is zero (this implies `ray_desc.dir`)
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

### Ray Tracing Pipelines

Functions

```wgsl
// Begins to check where (if anywhere) the ray defined by `ray_desc` hits in `acceleration_structure` calling through the `any_hit` shaders and `closest_hit` shader if something was hit or the `miss` shader if no hit was found
traceRay<T>(acceleration_structure: acceleration_structure, ray_desc: RayDesc, payload: ptr<ray_payload, T>)
```

> [!CAUTION]
>
> #### ⚠️Undefined behavior ⚠️:
>
> Calling `traceRay` inside another `traceRay` more than `max_recursion_depth` times
>
> \*this is only known undefined behaviour, and will be worked around in the future.

New shader stages

```wgsl
// First stage to be called, allowed to call `traceRay`
@ray_generation
fn rg() {}

// Stage called on any hit that is not opaque, not allowed to call `traceRay`
@any_hit
fn ah() {}

// Stage called on the closest hit, allowed to call `traceRay`
@closest_hit
fn ch() {}

// Stage call if there was never a hit, allowed to call `traceRay`
@miss
fn miss() {}
```

### Acceleration structure tags

These are tags that can be added to a acceleration structure (`acceleration_structure` ->
`acceleration_structure<... insert tags here! ...>`) and to a ray query (`ray_query` ->
`ray_query<... insert tags here! ...>`). These require more features.

| Tag             | Requirements                          | Description                                                            |
| --------------- | ------------------------------------- | ---------------------------------------------------------------------- |
| `vertex_return` | `enable wgpu_ray_query_vertex_return` | Allows getting the vertices of the hit triangle when using ray queries |
