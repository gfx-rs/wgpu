use alloc::{borrow::Cow, string::String};
use std::io::Write as _;

use crate::command::IdReferences;

use super::{Action, FILE_NAME};

pub(crate) fn new_render_bundle_encoder_descriptor<'a>(
    label: crate::Label<'a>,
    context: &'a crate::device::RenderPassContext,
    depth_read_only: bool,
    stencil_read_only: bool,
) -> crate::command::RenderBundleEncoderDescriptor<'a> {
    crate::command::RenderBundleEncoderDescriptor {
        label,
        color_formats: Cow::Borrowed(&context.attachments.colors),
        depth_stencil: context.attachments.depth_stencil.map(|format| {
            wgt::RenderBundleDepthStencil {
                format,
                depth_read_only,
                stencil_read_only,
            }
        }),
        sample_count: context.sample_count,
        multiview: context.multiview,
    }
}

#[derive(Debug)]
pub struct Trace {
    path: std::path::PathBuf,
    file: std::fs::File,
    config: ron::ser::PrettyConfig,
    binary_id: usize,
}

impl Trace {
    pub fn new(path: std::path::PathBuf) -> Result<Self, std::io::Error> {
        log::info!("Tracing into '{path:?}'");
        let mut file = std::fs::File::create(path.join(FILE_NAME))?;
        file.write_all(b"[\n")?;
        Ok(Self {
            path,
            file,
            config: ron::ser::PrettyConfig::default(),
            binary_id: 0,
        })
    }

    pub fn make_binary(&mut self, kind: &str, data: &[u8]) -> String {
        self.binary_id += 1;
        let name = std::format!("data{}.{}", self.binary_id, kind);
        let _ = std::fs::write(self.path.join(&name), data);
        name
    }

    pub(crate) fn add(&mut self, action: Action<'_, IdReferences>)
    where
        for<'a> Action<'a, IdReferences>: serde::Serialize,
    {
        match ron::ser::to_string_pretty(&action, self.config.clone()) {
            Ok(string) => {
                let _ = writeln!(self.file, "{string},");
            }
            Err(e) => {
                log::warn!("RON serialization failure: {e:?}");
            }
        }
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        let _ = self.file.write_all(b"]");
    }
}
