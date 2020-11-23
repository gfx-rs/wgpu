use std::{error::Error, fmt::Display};

use super::Context;

pub(crate) fn format_error(err: &(impl Error + 'static), context: &Context) -> String {
    let mut err_descs = Vec::new();

    err_descs.push(fmt_pretty_any(err, context));

    let mut source_opt = err.source();
    while let Some(source) = source_opt {
        err_descs.push(fmt_pretty_any(source, context));
        source_opt = source.source();
    }

    let desc = format!("Validation Error\n\nCaused by:\n{}", err_descs.join(""));
    desc
}

fn fmt_pretty_any(error: &(dyn Error + 'static), context: &Context) -> String {
    if let Some(pretty_err) = error.downcast_ref::<super::direct::ContextError>() {
        return pretty_err.fmt_pretty(context);
    }
    if let Some(pretty_err) = error.downcast_ref::<wgc::command::RenderCommandError>() {
        return pretty_err.fmt_pretty(context);
    }
    if let Some(pretty_err) = error.downcast_ref::<wgc::binding_model::CreateBindGroupError>() {
        return pretty_err.fmt_pretty(context);
    }
    if let Some(pretty_err) = error.downcast_ref::<wgc::binding_model::CreatePipelineLayoutError>()
    {
        return pretty_err.fmt_pretty(context);
    }
    if let Some(pretty_err) = error.downcast_ref::<wgc::command::ExecutionError>() {
        return pretty_err.fmt_pretty(context);
    }
    if let Some(pretty_err) = error.downcast_ref::<wgc::command::RenderPassErrorInner>() {
        return pretty_err.fmt_pretty(context);
    }
    if let Some(pretty_err) = error.downcast_ref::<wgc::command::RenderPassError>() {
        return pretty_err.fmt_pretty(context);
    }
    if let Some(pretty_err) = error.downcast_ref::<wgc::command::ComputePassErrorInner>() {
        return pretty_err.fmt_pretty(context);
    }
    if let Some(pretty_err) = error.downcast_ref::<wgc::command::ComputePassError>() {
        return pretty_err.fmt_pretty(context);
    }
    if let Some(pretty_err) = error.downcast_ref::<wgc::command::RenderBundleError>() {
        return pretty_err.fmt_pretty(context);
    }
    if let Some(pretty_err) = error.downcast_ref::<wgc::command::TransferError>() {
        return pretty_err.fmt_pretty(context);
    }

    // default
    format_error_line(error.as_display())
}

pub(crate) fn format_error_line(err: &dyn Display) -> String {
    format!("    {}\n", err)
}

pub(crate) fn format_note_line(note: &dyn Display) -> String {
    format!("      note: {}\n", note)
}

pub(crate) fn format_label_line(label_key: &str, label_value: &str) -> String {
    if label_key.is_empty() || label_value.is_empty() {
        String::new()
    } else {
        format_note_line(&format!("{} = `{}`", label_key, label_value))
    }
}

trait AsDisplay {
    fn as_display(&self) -> &dyn Display;
}

impl<T: Display> AsDisplay for T {
    fn as_display(&self) -> &dyn Display {
        self
    }
}

pub trait PrettyError: Error {
    fn fmt_pretty(&self, _context: &Context) -> String {
        format_error_line(self.as_display())
    }
}

impl PrettyError for super::direct::ContextError {
    fn fmt_pretty(&self, _context: &Context) -> String {
        format_error_line(self.as_display()) + &format_label_line(self.label_key, &self.label)
    }
}

impl PrettyError for wgc::command::RenderCommandError {
    fn fmt_pretty(&self, context: &Context) -> String {
        let global = context.global();
        let mut ret = format_error_line(self);
        match *self {
            Self::InvalidBindGroup(id) => {
                let name = wgc::gfx_select!(id => global.bind_group_label(id));
                ret.push_str(&format_label_line("bind group", &name));
            }
            Self::InvalidPipeline(id) => {
                let name = wgc::gfx_select!(id => global.render_pipeline_label(id));
                ret.push_str(&format_label_line("render pipeline", &name));
            }
            Self::Buffer(id, ..) | Self::DestroyedBuffer(id) => {
                let name = wgc::gfx_select!(id => global.buffer_label(id));
                ret.push_str(&format_label_line("buffer", &name));
            }
            _ => {}
        };
        ret
    }
}
impl PrettyError for wgc::binding_model::CreateBindGroupError {
    fn fmt_pretty(&self, context: &Context) -> String {
        let global = context.global();
        let mut ret = format_error_line(self);
        match *self {
            Self::InvalidBuffer(id) => {
                let name = wgc::gfx_select!(id => global.buffer_label(id));
                ret.push_str(&format_label_line("buffer", &name));
            }
            Self::InvalidTextureView(id) => {
                let name = wgc::gfx_select!(id => global.texture_view_label(id));
                ret.push_str(&format_label_line("texture view", &name));
            }
            Self::InvalidSampler(id) => {
                let name = wgc::gfx_select!(id => global.sampler_label(id));
                ret.push_str(&format_label_line("sampler", &name));
            }
            _ => {}
        };
        ret
    }
}

impl PrettyError for wgc::binding_model::CreatePipelineLayoutError {
    fn fmt_pretty(&self, context: &Context) -> String {
        let global = context.global();
        let mut ret = format_error_line(self);
        match *self {
            Self::InvalidBindGroupLayout(id) => {
                let name = wgc::gfx_select!(id => global.bind_group_layout_label(id));
                ret.push_str(&format_label_line("bind group layout", &name));
            }
            _ => {}
        };
        ret
    }
}

impl PrettyError for wgc::command::ExecutionError {
    fn fmt_pretty(&self, context: &Context) -> String {
        let global = context.global();
        let mut ret = format_error_line(self);
        match *self {
            Self::DestroyedBuffer(id) => {
                let name = wgc::gfx_select!(id => global.buffer_label(id));
                ret.push_str(&format_label_line("buffer", &name));
            }
        };
        ret
    }
}

impl PrettyError for wgc::command::RenderPassErrorInner {
    fn fmt_pretty(&self, context: &Context) -> String {
        let global = context.global();
        let mut ret = format_error_line(self);
        match *self {
            Self::InvalidAttachment(id) => {
                let name = wgc::gfx_select!(id => global.texture_view_label(id));
                ret.push_str(&format_label_line("attachment", &name));
            }
            _ => {}
        };
        ret
    }
}

impl PrettyError for wgc::command::RenderPassError {
    fn fmt_pretty(&self, context: &Context) -> String {
        // This error is wrapper for the inner error,
        // but the scope has useful labels
        format_error_line(self) + &self.scope.fmt_pretty(context)
    }
}

impl PrettyError for wgc::command::ComputePassError {
    fn fmt_pretty(&self, context: &Context) -> String {
        // This error is wrapper for the inner error,
        // but the scope has useful labels
        format_error_line(self) + &self.scope.fmt_pretty(context)
    }
}
impl PrettyError for wgc::command::RenderBundleError {
    fn fmt_pretty(&self, context: &Context) -> String {
        // This error is wrapper for the inner error,
        // but the scope has useful labels
        format_error_line(self) + &self.scope.fmt_pretty(context)
    }
}

impl PrettyError for wgc::command::ComputePassErrorInner {
    fn fmt_pretty(&self, context: &Context) -> String {
        let global = context.global();
        let mut ret = format_error_line(self);
        match *self {
            Self::InvalidBindGroup(id) => {
                let name = wgc::gfx_select!(id => global.bind_group_label(id));
                ret.push_str(&format_label_line("bind group", &name));
            }
            Self::InvalidPipeline(id) => {
                let name = wgc::gfx_select!(id => global.compute_pipeline_label(id));
                ret.push_str(&format_label_line("pipeline", &name));
            }
            Self::InvalidIndirectBuffer(id) => {
                let name = wgc::gfx_select!(id => global.buffer_label(id));
                ret.push_str(&format_label_line("indirect buffer", &name));
            }
            _ => {}
        };
        ret
    }
}

impl PrettyError for wgc::command::TransferError {
    fn fmt_pretty(&self, context: &Context) -> String {
        let global = context.global();
        let mut ret = format_error_line(self);
        match *self {
            Self::InvalidBuffer(id) => {
                let name = wgc::gfx_select!(id => global.buffer_label(id));
                ret.push_str(&format_label_line("label", &name));
            }
            Self::InvalidTexture(id) => {
                let name = wgc::gfx_select!(id => global.texture_label(id));
                ret.push_str(&format_label_line("texture", &name));
            }
            // Self::MissingCopySrcUsageFlag(buf_opt, tex_opt) => {
            //     if let Some(buf) = buf_opt {
            //         let name = wgc::gfx_select!(buf => global.buffer_label(buf));
            //         ret.push_str(&format_label_line("source", &name));
            //     }
            //     if let Some(tex) = tex_opt {
            //         let name = wgc::gfx_select!(tex => global.texture_label(tex));
            //         ret.push_str(&format_label_line("source", &name));
            //     }
            // }
            Self::MissingCopyDstUsageFlag(buf_opt, tex_opt) => {
                if let Some(buf) = buf_opt {
                    let name = wgc::gfx_select!(buf => global.buffer_label(buf));
                    ret.push_str(&format_label_line("destination", &name));
                }
                if let Some(tex) = tex_opt {
                    let name = wgc::gfx_select!(tex => global.texture_label(tex));
                    ret.push_str(&format_label_line("destination", &name));
                }
            }
            _ => {}
        };
        ret
    }
}

impl PrettyError for wgc::command::PassErrorScope {
    fn fmt_pretty(&self, context: &Context) -> String {
        // This error is not in the error chain, only notes are needed
        let global = context.global();
        match *self {
            Self::Pass(id) => {
                let name = wgc::gfx_select!(id => global.command_buffer_label(id));
                format_label_line("command buffer", &name)
            }
            Self::SetBindGroup(id) => {
                let name = wgc::gfx_select!(id => global.bind_group_label(id));
                format_label_line("bind group", &name)
            }
            Self::SetPipelineRender(id) => {
                let name = wgc::gfx_select!(id => global.render_pipeline_label(id));
                format_label_line("render pipeline", &name)
            }
            Self::SetPipelineCompute(id) => {
                let name = wgc::gfx_select!(id => global.compute_pipeline_label(id));
                format_label_line("compute pipeline", &name)
            }
            Self::SetVertexBuffer(id) => {
                let name = wgc::gfx_select!(id => global.buffer_label(id));
                format_label_line("buffer", &name)
            }
            Self::SetIndexBuffer(id) => {
                let name = wgc::gfx_select!(id => global.buffer_label(id));
                format_label_line("buffer", &name)
            }
            _ => String::new(),
        }
    }
}
