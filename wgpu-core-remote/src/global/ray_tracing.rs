use alloc::{boxed::Box, sync::Arc};
use wgpu_core::{ray_tracing as core_rt, resource};
use wgpu_core_remote_types::ray_tracing as remote_rt;

use crate::{global::Global, id};

fn build_flags(preference: remote_rt::BuildPreference) -> wgt::AccelerationStructureFlags {
    match preference {
        remote_rt::BuildPreference::FastTrace => wgt::AccelerationStructureFlags::PREFER_FAST_TRACE,
        remote_rt::BuildPreference::FastBuild => wgt::AccelerationStructureFlags::PREFER_FAST_BUILD,
    }
}

fn triangle_size(
    size: &remote_rt::TriangleGeometrySize,
) -> wgt::BlasTriangleGeometrySizeDescriptor {
    wgt::BlasTriangleGeometrySizeDescriptor {
        vertex_format: wgt::VertexFormat::Float32x3,
        vertex_count: size.vertex_count,
        index_format: size.index.map(|index| index.format),
        index_count: size.index.map(|index| index.count),
        flags: if size.opaque {
            wgt::AccelerationStructureGeometryFlags::OPAQUE
        } else {
            wgt::AccelerationStructureGeometryFlags::empty()
        },
    }
}

impl Global {
    pub fn device_create_blas(
        &self,
        device_id: id::DeviceId,
        desc: &remote_rt::BlasDescriptor,
        id_in: id::BlasId,
    ) {
        let mut hub = self.hub.borrow_mut();
        let device = hub.devices.get(device_id);
        let native_desc = resource::BlasDescriptor {
            label: desc.label.clone(),
            flags: build_flags(desc.build_preference),
            update_mode: wgt::AccelerationStructureUpdateMode::Build,
        };
        let sizes = wgt::BlasGeometrySizeDescriptors::Triangles {
            descriptors: desc.geometries.iter().map(triangle_size).collect(),
        };
        let (blas, error) = device.create_blas(&native_desc, sizes);
        if let Some(error) = error {
            device.handle_error(error, desc.label.as_deref(), "Device::create_blas");
        }
        hub.blases.assign(id_in, blas);
    }

    pub fn device_create_tlas(
        &self,
        device_id: id::DeviceId,
        desc: &remote_rt::TlasDescriptor,
        id_in: id::TlasId,
    ) {
        let mut hub = self.hub.borrow_mut();
        let device = hub.devices.get(device_id);
        let native_desc = resource::TlasDescriptor {
            label: desc.label.clone(),
            max_instances: desc.max_instances,
            flags: build_flags(desc.build_preference),
            update_mode: wgt::AccelerationStructureUpdateMode::Build,
        };
        let (tlas, error) = device.create_tlas(&native_desc);
        if let Some(error) = error {
            device.handle_error(error, desc.label.as_deref(), "Device::create_tlas");
        }
        hub.tlases.assign(id_in, tlas);
    }

    pub fn blas_remove(&self, blas_id: id::BlasId) -> Arc<resource::Blas> {
        self.hub.borrow_mut().blases.remove(blas_id)
    }

    pub fn tlas_remove(&self, tlas_id: id::TlasId) -> Arc<resource::Tlas> {
        self.hub.borrow_mut().tlases.remove(tlas_id)
    }

    pub fn command_encoder_build_acceleration_structures(
        &self,
        encoder_id: id::CommandEncoderId,
        blases: &[remote_rt::BlasBuild],
        tlases: &[remote_rt::TlasBuild],
    ) {
        let hub = self.hub.borrow();
        let encoder = hub.command_encoders.get(encoder_id);
        let sizes: Vec<Vec<_>> = blases
            .iter()
            .map(|build| {
                build
                    .geometries
                    .iter()
                    .map(|geometry| triangle_size(&geometry.size))
                    .collect()
            })
            .collect();
        let entries = blases
            .iter()
            .zip(&sizes)
            .map(|(build, sizes)| core_rt::BlasBuildEntry {
                blas: hub.blases.get(build.blas),
                geometries: core_rt::BlasGeometries::TriangleGeometries(Box::new(
                    build.geometries.iter().zip(sizes).map(|(geometry, size)| {
                        core_rt::BlasTriangleGeometry {
                            size,
                            vertex_buffer: hub.buffers.get(geometry.vertex_buffer),
                            index_buffer: geometry.index_buffer.map(|id| hub.buffers.get(id)),
                            transform_buffer: None,
                            first_vertex: geometry.first_vertex,
                            vertex_stride: geometry.vertex_stride,
                            first_index: geometry.first_index,
                            transform_buffer_offset: None,
                        }
                    }),
                )),
            });
        let tlases = tlases
            .iter()
            .map(|build| core_rt::ArcTlasPackage {
                tlas: hub.tlases.get(build.tlas),
                instances: build
                    .instances
                    .iter()
                    .map(|instance| {
                        Some(core_rt::ArcTlasInstance {
                            blas: hub.blases.get(instance.blas),
                            transform: instance.transform,
                            custom_data: instance.custom_index,
                            mask: instance.mask,
                            intersection_index: 0,
                        })
                    })
                    .collect(),
                lowest_unmodified: 0,
            })
            .collect();
        encoder.build_acceleration_structures(entries, tlases);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_build_preferences_do_not_enable_other_native_extensions() {
        assert_eq!(
            build_flags(remote_rt::BuildPreference::FastTrace),
            wgt::AccelerationStructureFlags::PREFER_FAST_TRACE
        );
        assert_eq!(
            build_flags(remote_rt::BuildPreference::FastBuild),
            wgt::AccelerationStructureFlags::PREFER_FAST_BUILD
        );
    }

    #[test]
    fn triangle_sizes_pair_index_format_and_count() {
        let mut size = remote_rt::TriangleGeometrySize {
            vertex_count: 3,
            index: None,
            opaque: true,
        };
        let native = triangle_size(&size);
        assert_eq!(native.vertex_format, wgt::VertexFormat::Float32x3);
        assert_eq!(native.index_format, None);
        assert_eq!(native.index_count, None);
        assert_eq!(
            native.flags,
            wgt::AccelerationStructureGeometryFlags::OPAQUE
        );
        size.index = Some(remote_rt::TriangleIndexSize {
            format: wgt::IndexFormat::Uint32,
            count: 6,
        });
        size.opaque = false;
        let native = triangle_size(&size);
        assert_eq!(native.index_format, Some(wgt::IndexFormat::Uint32));
        assert_eq!(native.index_count, Some(6));
        assert!(native.flags.is_empty());
    }
}
