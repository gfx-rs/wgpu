use wgpu_core::present::{SurfaceError, SurfaceOutput};
use wgt::SurfaceStatus;

use crate::global::Global;
use crate::id;

impl Global {
    pub fn surface_get_current_texture(
        &self,
        surface_id: id::SurfaceId,
        texture_id_in: Option<id::TextureId>,
    ) -> Result<SurfaceOutput<id::TextureId>, SurfaceError> {
        let surface = self.surfaces.get(surface_id);

        let fid = self.hub.textures.prepare(texture_id_in);

        let output = surface.get_current_texture()?;

        let status = output.status;
        let texture_id = output.texture.map(|texture| fid.assign(texture));

        Ok(SurfaceOutput {
            status,
            texture: texture_id,
        })
    }

    pub fn surface_present(
        &self,
        surface_id: id::SurfaceId,
    ) -> Result<SurfaceStatus, SurfaceError> {
        let surface = self.surfaces.get(surface_id);

        surface.present()
    }

    pub fn surface_texture_discard(&self, surface_id: id::SurfaceId) -> Result<(), SurfaceError> {
        let surface = self.surfaces.get(surface_id);

        surface.discard()
    }

    pub fn surface_texture_release(&self, surface_id: id::SurfaceId) -> Result<(), SurfaceError> {
        let surface = self.surfaces.get(surface_id);

        surface.release()
    }
}
