use wgpu_core::ray_tracing::{BlasPrepareCompactError, CreateBlasError, CreateTlasError};
use wgpu_core::resource::{self, BlasCompactCallback, InvalidResourceError};
use wgpu_core::SubmissionIndex;

use crate::global::Global;
use crate::id::{self, BlasId, TlasId};

impl Global {
    pub fn device_create_blas(
        &self,
        device_id: id::DeviceId,
        desc: &resource::BlasDescriptor,
        sizes: wgt::BlasGeometrySizeDescriptors,
        id_in: Option<BlasId>,
    ) -> (BlasId, Option<u64>, Option<CreateBlasError>) {
        let fid = self.hub.blas_s.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        let (blas, error) = device.create_blas(desc, sizes);

        let handle = blas.handle();

        let id = fid.assign(blas);

        (id, handle, error)
    }

    pub fn device_create_tlas(
        &self,
        device_id: id::DeviceId,
        desc: &resource::TlasDescriptor,
        id_in: Option<TlasId>,
    ) -> (TlasId, Option<CreateTlasError>) {
        let fid = self.hub.tlas_s.prepare(id_in);

        let device = self.hub.devices.get(device_id);

        let (tlas, error) = device.create_tlas(desc);

        let id = fid.assign(tlas);

        (id, error)
    }

    pub fn blas_drop(&self, blas_id: BlasId) {
        let _blas = self.hub.blas_s.remove(blas_id);
    }

    pub fn tlas_drop(&self, tlas_id: TlasId) {
        let _tlas = self.hub.tlas_s.remove(tlas_id);
    }

    pub fn blas_prepare_compact_async(
        &self,
        blas_id: BlasId,
        callback: Option<BlasCompactCallback>,
    ) -> Result<SubmissionIndex, BlasPrepareCompactError> {
        let hub = &self.hub;

        let blas = hub.blas_s.get(blas_id);

        blas.prepare_compact_async(callback)
    }

    pub fn ready_for_compaction(&self, blas_id: BlasId) -> Result<bool, InvalidResourceError> {
        let hub = &self.hub;

        let blas = hub.blas_s.get(blas_id);

        blas.ready_for_compaction()
    }
}
