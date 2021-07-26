use core::fmt;
use std::error::Error;

use crate::{
    gfx_select,
    hub::{Global, IdentityManagerFactory},
};

pub trait AsDisplay {
    fn as_display(&self) -> &dyn fmt::Display;
}

impl<T: fmt::Display> AsDisplay for T {
    fn as_display(&self) -> &dyn fmt::Display {
        self
    }
}

pub trait PrettyError: Error {
    fn fmt_pretty(&self, _global: &Global<IdentityManagerFactory>) -> String {
        format_error_line(self.as_display())
    }
}

impl PrettyError for super::command::RenderCommandError {
    fn fmt_pretty(&self, global: &Global<IdentityManagerFactory>) -> String {
        let mut ret = format_error_line(self);
        match *self {
            Self::InvalidBindGroup(id) => {
                let name = gfx_select!(id => global.bind_group_label(id));
                ret.push_str(&format_label_line("bind group", &name));
            }
            Self::InvalidPipeline(id) => {
                let name = gfx_select!(id => global.render_pipeline_label(id));
                ret.push_str(&format_label_line("render pipeline", &name));
            }
            Self::Buffer(id, ..) | Self::DestroyedBuffer(id) => {
                let name = gfx_select!(id => global.buffer_label(id));
                ret.push_str(&format_label_line("buffer", &name));
            }
            _ => {}
        };
        ret
    }
}
impl PrettyError for crate::binding_model::CreateBindGroupError {
    fn fmt_pretty(&self, global: &Global<IdentityManagerFactory>) -> String {
        let mut ret = format_error_line(self);
        match *self {
            Self::InvalidBuffer(id) => {
                let name = crate::gfx_select!(id => global.buffer_label(id));
                ret.push_str(&format_label_line("buffer", &name));
            }
            Self::InvalidTextureView(id) => {
                let name = crate::gfx_select!(id => global.texture_view_label(id));
                ret.push_str(&format_label_line("texture view", &name));
            }
            Self::InvalidSampler(id) => {
                let name = crate::gfx_select!(id => global.sampler_label(id));
                ret.push_str(&format_label_line("sampler", &name));
            }
            _ => {}
        };
        ret
    }
}

impl PrettyError for crate::binding_model::CreatePipelineLayoutError {
    fn fmt_pretty(&self, global: &Global<IdentityManagerFactory>) -> String {
        let mut ret = format_error_line(self);
        if let Self::InvalidBindGroupLayout(id) = *self {
            let name = crate::gfx_select!(id => global.bind_group_layout_label(id));
            ret.push_str(&format_label_line("bind group layout", &name));
        };
        ret
    }
}

impl PrettyError for crate::command::ExecutionError {
    fn fmt_pretty(&self, global: &Global<IdentityManagerFactory>) -> String {
        let mut ret = format_error_line(self);
        match *self {
            Self::DestroyedBuffer(id) => {
                let name = crate::gfx_select!(id => global.buffer_label(id));
                ret.push_str(&format_label_line("buffer", &name));
            }
            Self::Unimplemented(_reason) => {}
        };
        ret
    }
}

impl PrettyError for crate::command::RenderPassErrorInner {
    fn fmt_pretty(&self, global: &Global<IdentityManagerFactory>) -> String {
        let mut ret = format_error_line(self);
        if let Self::InvalidAttachment(id) = *self {
            let name = crate::gfx_select!(id => global.texture_view_label(id));
            ret.push_str(&format_label_line("attachment", &name));
        };
        ret
    }
}

impl PrettyError for crate::command::RenderPassError {
    fn fmt_pretty(&self, global: &Global<IdentityManagerFactory>) -> String {
        // This error is wrapper for the inner error,
        // but the scope has useful labels
        format_error_line(self) + &self.scope.fmt_pretty(global)
    }
}

impl PrettyError for crate::command::ComputePassError {
    fn fmt_pretty(&self, global: &Global<IdentityManagerFactory>) -> String {
        // This error is wrapper for the inner error,
        // but the scope has useful labels
        format_error_line(self) + &self.scope.fmt_pretty(global)
    }
}
impl PrettyError for crate::command::RenderBundleError {
    fn fmt_pretty(&self, global: &Global<IdentityManagerFactory>) -> String {
        // This error is wrapper for the inner error,
        // but the scope has useful labels
        format_error_line(self) + &self.scope.fmt_pretty(global)
    }
}

impl PrettyError for crate::command::ComputePassErrorInner {
    fn fmt_pretty(&self, global: &Global<IdentityManagerFactory>) -> String {
        let mut ret = format_error_line(self);
        match *self {
            Self::InvalidBindGroup(id) => {
                let name = crate::gfx_select!(id => global.bind_group_label(id));
                ret.push_str(&format_label_line("bind group", &name));
            }
            Self::InvalidPipeline(id) => {
                let name = crate::gfx_select!(id => global.compute_pipeline_label(id));
                ret.push_str(&format_label_line("pipeline", &name));
            }
            Self::InvalidIndirectBuffer(id) => {
                let name = crate::gfx_select!(id => global.buffer_label(id));
                ret.push_str(&format_label_line("indirect buffer", &name));
            }
            _ => {}
        };
        ret
    }
}

impl PrettyError for crate::command::TransferError {
    fn fmt_pretty(&self, global: &Global<IdentityManagerFactory>) -> String {
        let mut ret = format_error_line(self);
        match *self {
            Self::InvalidBuffer(id) => {
                let name = crate::gfx_select!(id => global.buffer_label(id));
                ret.push_str(&format_label_line("label", &name));
            }
            Self::InvalidTexture(id) => {
                let name = crate::gfx_select!(id => global.texture_label(id));
                ret.push_str(&format_label_line("texture", &name));
            }
            // Self::MissingCopySrcUsageFlag(buf_opt, tex_opt) => {
            //     if let Some(buf) = buf_opt {
            //         let name = crate::gfx_select!(buf => global.buffer_label(buf));
            //         ret.push_str(&format_label_line("source", &name));
            //     }
            //     if let Some(tex) = tex_opt {
            //         let name = crate::gfx_select!(tex => global.texture_label(tex));
            //         ret.push_str(&format_label_line("source", &name));
            //     }
            // }
            Self::MissingCopyDstUsageFlag(buf_opt, tex_opt) => {
                if let Some(buf) = buf_opt {
                    let name = crate::gfx_select!(buf => global.buffer_label(buf));
                    ret.push_str(&format_label_line("destination", &name));
                }
                if let Some(tex) = tex_opt {
                    let name = crate::gfx_select!(tex => global.texture_label(tex));
                    ret.push_str(&format_label_line("destination", &name));
                }
            }
            _ => {}
        };
        ret
    }
}

impl PrettyError for crate::command::PassErrorScope {
    fn fmt_pretty(&self, global: &Global<IdentityManagerFactory>) -> String {
        // This error is not in the error chain, only notes are needed
        match *self {
            Self::Pass(id) => {
                let name = crate::gfx_select!(id => global.command_buffer_label(id));
                format_label_line("command buffer", &name)
            }
            Self::SetBindGroup(id) => {
                let name = crate::gfx_select!(id => global.bind_group_label(id));
                format_label_line("bind group", &name)
            }
            Self::SetPipelineRender(id) => {
                let name = crate::gfx_select!(id => global.render_pipeline_label(id));
                format_label_line("render pipeline", &name)
            }
            Self::SetPipelineCompute(id) => {
                let name = crate::gfx_select!(id => global.compute_pipeline_label(id));
                format_label_line("compute pipeline", &name)
            }
            Self::SetVertexBuffer(id) => {
                let name = crate::gfx_select!(id => global.buffer_label(id));
                format_label_line("buffer", &name)
            }
            Self::SetIndexBuffer(id) => {
                let name = crate::gfx_select!(id => global.buffer_label(id));
                format_label_line("buffer", &name)
            }
            Self::Draw { pipeline, .. } => {
                if let Some(id) = pipeline {
                    let name = crate::gfx_select!(id => global.render_pipeline_label(id));
                    format_label_line("render pipeline", &name)
                } else {
                    String::new()
                }
            }
            Self::Dispatch { pipeline, .. } => {
                if let Some(id) = pipeline {
                    let name = crate::gfx_select!(id => global.compute_pipeline_label(id));
                    format_label_line("compute pipeline", &name)
                } else {
                    String::new()
                }
            }
            _ => String::new(),
        }
    }
}

impl PrettyError for ContextError {
    fn fmt_pretty(&self, _global: &Global<IdentityManagerFactory>) -> String {
        format_error_line(self.as_display()) + &format_label_line(self.label_key, &self.label)
    }
}

pub fn format_error_line(err: &dyn fmt::Display) -> String {
    format!("    {}\n", err)
}

pub fn format_note_line(note: &dyn fmt::Display) -> String {
    format!("      note: {}\n", note)
}

pub fn format_label_line(label_key: &str, label_value: &str) -> String {
    if label_key.is_empty() || label_value.is_empty() {
        String::new()
    } else {
        format_note_line(&format!("{} = `{}`", label_key, label_value))
    }
}

pub fn format_pretty_any(
    global: &Global<IdentityManagerFactory>,
    error: &(dyn Error + 'static),
) -> String {
    if let Some(pretty_err) = error.downcast_ref::<ContextError>() {
        return pretty_err.fmt_pretty(global);
    }

    if let Some(pretty_err) = error.downcast_ref::<crate::command::RenderCommandError>() {
        return pretty_err.fmt_pretty(global);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::binding_model::CreateBindGroupError>() {
        return pretty_err.fmt_pretty(global);
    }
    if let Some(pretty_err) =
        error.downcast_ref::<crate::binding_model::CreatePipelineLayoutError>()
    {
        return pretty_err.fmt_pretty(global);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::ExecutionError>() {
        return pretty_err.fmt_pretty(global);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::RenderPassErrorInner>() {
        return pretty_err.fmt_pretty(global);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::RenderPassError>() {
        return pretty_err.fmt_pretty(global);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::ComputePassErrorInner>() {
        return pretty_err.fmt_pretty(global);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::ComputePassError>() {
        return pretty_err.fmt_pretty(global);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::RenderBundleError>() {
        return pretty_err.fmt_pretty(global);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::TransferError>() {
        return pretty_err.fmt_pretty(global);
    }

    // default
    format_error_line(error.as_display())
}

#[derive(Debug)]
pub struct ContextError {
    pub string: &'static str,
    pub cause: Box<dyn Error + Send + Sync + 'static>,
    pub label_key: &'static str,
    pub label: String,
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "In {}", self.string)
    }
}

impl Error for ContextError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.cause.as_ref())
    }
}
