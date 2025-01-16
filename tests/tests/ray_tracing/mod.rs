use std::mem;
use wgpu::util::BufferInitDescriptor;
use wgpu::{
    util::DeviceExt, Blas, BlasBuildEntry, BlasGeometries, BlasGeometrySizeDescriptors,
    BlasTriangleGeometry, BlasTriangleGeometrySizeDescriptor, Buffer, CreateBlasDescriptor,
    CreateTlasDescriptor, TlasInstance, TlasPackage,
};
use wgpu_test::TestingContext;
use wgt::{
    AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, BufferAddress, BufferUsages, VertexFormat,
};

mod as_build;
mod as_create;
mod as_use_after_free;
mod scene;
mod shader;

pub struct AsBuildContext {
    vertices: Buffer,
    blas_size: BlasTriangleGeometrySizeDescriptor,
    blas: Blas,
    // Putting this last, forces the BLAS to die before the TLAS.
    tlas_package: TlasPackage,
}

impl AsBuildContext {
    pub fn new(ctx: &TestingContext) -> Self {
        let vertices = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &[0; mem::size_of::<[[f32; 3]; 3]>()],
            usage: BufferUsages::BLAS_INPUT,
        });

        let blas_size = BlasTriangleGeometrySizeDescriptor {
            vertex_format: VertexFormat::Float32x3,
            vertex_count: 3,
            index_format: None,
            index_count: None,
            flags: AccelerationStructureGeometryFlags::empty(),
        };

        let blas = ctx.device.create_blas(
            &CreateBlasDescriptor {
                label: Some("BLAS"),
                flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: AccelerationStructureUpdateMode::Build,
            },
            BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![blas_size.clone()],
            },
        );

        let tlas = ctx.device.create_tlas(&CreateTlasDescriptor {
            label: Some("TLAS"),
            max_instances: 1,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        });

        let mut tlas_package = TlasPackage::new(tlas);
        tlas_package[0] = Some(TlasInstance::new(
            &blas,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            0,
            0xFF,
        ));

        Self {
            vertices,
            blas_size,
            blas,
            tlas_package,
        }
    }

    pub fn blas_build_entry(&self) -> BlasBuildEntry {
        BlasBuildEntry {
            blas: &self.blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &self.blas_size,
                vertex_buffer: &self.vertices,
                first_vertex: 0,
                vertex_stride: mem::size_of::<[f32; 3]>() as BufferAddress,
                index_buffer: None,
                first_index: None,
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }
    }
}
