// Ray tracing api proposal for wgpu (underlining Vulkan, Metal and DX12 implementations)

// The general design goal is to come up with an simpler Api, which allows for validation.
// Since this validation would have high overhead in some cases, 
// I decided to provide more limited functions and unsafe functions for the same action, to evade this tradeoff.  

// Error handling and traits like Debug are omitted. 

// Core structures with no public members
pub struct Blas {}
pub struct Tlas {}
pub struct BlasRequirements {}
pub struct TlasInstances{}

// Size descriptors used to describe the size requirements of blas geometries.
// Also used internally for strict validation
pub struct BlasTriangleGeometrySizeDescriptor{
    pub vertex_format: wgt::VertexFormat,
    pub vertex_count: u32,
    pub index_format: Option<wgt::IndexFormat>,
    pub index_count: Option<u32>,
    pub flags: AccelerationStructureGeometryFlags,
}

pub struct BlasProceduralGeometrySizeDescriptor{
    pub count: u32,
    pub flags: AccelerationStructureGeometryFlags,
} 

// Procedural geometry contains AABBs
pub struct BlasProceduralGeometry{
    pub size: BlasTriangleGeometrySize,
    pub bounding_box_buffer: Buffer,
    pub bounding_box_buffer_offset: wgt::BufferAddress,
    pub bounding_box_stride: wgt::BufferAddress,
}

// Triangle Geometry contains vertices, optionally indices and transforms 
pub struct BlasTriangleGeometry{
    pub size: BlasTriangleGeometrySize,
    pub vertex_buffer: Buffer
    pub first_vertex: u32,
    pub vertex_stride: wgt::BufferAddress,
    pub index_buffer: Option<Buffer>,
    pub index_buffer_offset: Option<wgt::BufferAddress>,
    pub transform_buffer: Option<Buffer>,
    pub transform_buffer_offset: Option<wgt::BufferAddress>,
}

// Build flags 
pub struct AccelerationStructureFlags{
    // build_speed, small_size, ...
}

// Geometry flags
pub struct AccelerationStructureGeometryFlags{
    // opaque, no_duplicate_any_hit, ...
}

// Descriptors used to determine the memory requirements and validation of a acceleration structure 
pub enum BlasGeometrySizeDescriptors{
    Triangles{desc: Vec<BlasTriangleGeometrySizeDescriptor>},
    Procedural(desc: Vec<BlasProceduralGeometrySize>) 
}

// With prefer update, we decide if an update is possible, else we rebuild.
// Maybe a force update option could be useful
pub enum UpdateMode{
    Build,
    // Update,
    PreferUpdate,
}

// General descriptor for the size requirements, 
// since the required size depends on the contents and build flags 
pub struct GetBlasRequirementsDescriptor{
    pub flags: AccelerationStructureFlags,
}

// Creation descriptors, we provide flags, and update_mode.
// We store it in the structure, so we don't need to pass it every build.
pub struct CreateBlasDescriptor<'a>{
    pub flags: AccelerationStructureFlags,
    pub update_mode: UpdateMode,
}

pub struct CreateTlasDescriptor{
    pub max_instances: u32,
    pub flags: AccelerationStructureFlags,
    pub update_mode: UpdateMode,
}

// Secure instance entry for tlas
struct TlasInstance{
    transform: [f32; 12],
    custom_index: u32,
    mask: u8,
    shader_binding_table_record_offset: u32,
    flags: u8 //bitmap
    blas: Blas
}

impl Device {
    // Creates a new bottom level accelerations structures and sets internal states for validation and builds (e.g update mode)
    pub fn create_blas(&self, desc: &CreateBlasDescriptor, entries: BlasGeometrySizeDescriptors) -> Blas;

    // Creates a new top level accelerations structures and sets internal states for builds (e.g update mode)
    pub fn create_tlas(&self, desc: &CreateTlasDescriptor) -> Tlas;
}

// Enum for the different types of geometries inside a single blas build
// [Should we used nested iterators with dynamic dispatch instead] 
enum BlasGeometries<'a>{
    TriangleGeometries(&'a [BlasTriangleGeometry])
    ProceduralGeometries(&'a [BlasProceduralGeometry])
}

impl CommandEncoder {
    // Build acceleration structures.
    // Elements of blas may be used in a tlas (internal synchronization).
    // This function will allocate a single big scratch buffer, that is shared between internal builds.
    // If there are to many acceleration structures for a single build (size constraint),
    // we will automatically distribute them between multiple internal builds. (reducing the required size of the scratch buffer).
    // This version will be implemented in wgpu::util not wgpu-core.
    pub fn build_acceleration_structures<'a>(&self,
        blas: impl IntoIterator<Item=(&'a Blas,BlasGeometries<'a>)>,
        tlas: impl IntoIterator<Item=(&'a Tlas,TlasInstances<'a>)>,
    );

    // unsafe version without validation for tlas, directly using an instance buffer.
    // u32 for the number of instances to build
    pub fn build_acceleration_structures_unsafe_tlas<'a>(&self,
        blas: impl IntoIterator<Item=(&'a Blas,BlasGeometries<'a>)>,
        tlas: impl IntoIterator<Item=(&'a Tlas,&'a Buffer, u32)>,
    );

    // Creates a new blas and copies (in a compacting way) the contents of the provided blas
    // into the new one (compaction flag must be set). 
    pub fn compact_blas(&self, blas: &Blas) -> Blas;
}

// Safe Tlas Instance
impl TlasInstances{
    pub fn new(max_instances: u32) -> Self;

    // gets instances to read from
    pub fn get(&self) -> &[TlasInstance];
    // gets instances to modify, we keep track of the range to determine what we need to validate and copy
    pub fn get_mut_range(&mut self, range: Range<u32>) -> &mut [TlasInstance];
    // get the number of instances which will be build
    pub fn active(&self) -> u32;
    // set the number of instances which will be build
    pub fn set_active(&mut self, active: u32);
}