use crate::{context::Context, CommandEncoder, Texture};
use hal::TextureUses;
use wgc::{
    id::{CommandEncoderId, TextureId},
    track::TextureSelector,
};

#[cfg(any(not(target_arch = "wasm32"), feature = "emscripten"))]
pub trait CommandEncoderExt {
    fn transition_textures(&mut self, textures: &[(&Texture, TextureUses, TextureSelector)]);
}

#[cfg(any(not(target_arch = "wasm32"), feature = "emscripten"))]
impl CommandEncoderExt for CommandEncoder {
    fn transition_textures(&mut self, texture_uses: &[(&Texture, TextureUses, TextureSelector)]) {
        let encoder_id = CommandEncoderId::from(*self.id.as_ref().unwrap());
        let texture_uses = texture_uses.iter().map(|(texture, usage, selector)| {
            (TextureId::from(texture.id), *usage, selector.clone())
        });

        self.context
            .as_any()
            .downcast_ref::<crate::backend::Context>()
            .unwrap()
            .command_encoder_transition_textures(&encoder_id, texture_uses);
    }
}
