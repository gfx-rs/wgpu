use core::fmt;
use std::error::Error;

use crate::{gfx_select, global::Global, identity::IdentityManagerFactory};

pub struct ErrorFormatter<'a> {
    writer: &'a mut dyn fmt::Write,
    global: &'a Global<IdentityManagerFactory>,
}

impl<'a> ErrorFormatter<'a> {
    pub fn error(&mut self, err: &dyn Error) {
        writeln!(self.writer, "    {err}").expect("Error formatting error");
    }

    pub fn note(&mut self, note: &dyn fmt::Display) {
        writeln!(self.writer, "      note: {note}").expect("Error formatting error");
    }

    pub fn label(&mut self, label_key: &str, label_value: &String) {
        if !label_key.is_empty() && !label_value.is_empty() {
            self.note(&format!("{label_key} = `{label_value}`"));
        }
    }

    pub fn bind_group_label(&mut self, id: &crate::id::BindGroupId) {
        let global = self.global;
        let label: String = gfx_select!(id => global.bind_group_label(*id));
        self.label("bind group", &label);
    }

    pub fn bind_group_layout_label(&mut self, id: &crate::id::BindGroupLayoutId) {
        let global = self.global;
        let label: String = gfx_select!(id => global.bind_group_layout_label(*id));
        self.label("bind group layout", &label);
    }

    pub fn render_pipeline_label(&mut self, id: &crate::id::RenderPipelineId) {
        let global = self.global;
        let label: String = gfx_select!(id => global.render_pipeline_label(*id));
        self.label("render pipeline", &label);
    }

    pub fn compute_pipeline_label(&mut self, id: &crate::id::ComputePipelineId) {
        let global = self.global;
        let label: String = gfx_select!(id => global.compute_pipeline_label(*id));
        self.label("compute pipeline", &label);
    }

    pub fn buffer_label_with_key(&mut self, id: &crate::id::BufferId, key: &str) {
        let global = self.global;
        let label: String = gfx_select!(id => global.buffer_label(*id));
        self.label(key, &label);
    }

    pub fn buffer_label(&mut self, id: &crate::id::BufferId) {
        self.buffer_label_with_key(id, "buffer");
    }

    pub fn texture_label_with_key(&mut self, id: &crate::id::TextureId, key: &str) {
        let global = self.global;
        let label: String = gfx_select!(id => global.texture_label(*id));
        self.label(key, &label);
    }

    pub fn texture_label(&mut self, id: &crate::id::TextureId) {
        self.texture_label_with_key(id, "texture");
    }

    pub fn texture_view_label_with_key(&mut self, id: &crate::id::TextureViewId, key: &str) {
        let global = self.global;
        let label: String = gfx_select!(id => global.texture_view_label(*id));
        self.label(key, &label);
    }

    pub fn texture_view_label(&mut self, id: &crate::id::TextureViewId) {
        self.texture_view_label_with_key(id, "texture view");
    }

    pub fn sampler_label(&mut self, id: &crate::id::SamplerId) {
        let global = self.global;
        let label: String = gfx_select!(id => global.sampler_label(*id));
        self.label("sampler", &label);
    }

    pub fn command_buffer_label(&mut self, id: &crate::id::CommandBufferId) {
        let global = self.global;
        let label: String = gfx_select!(id => global.command_buffer_label(*id));
        self.label("command buffer", &label);
    }

    pub fn query_set_label(&mut self, id: &crate::id::QuerySetId) {
        let global = self.global;
        let label: String = gfx_select!(id => global.query_set_label(*id));
        self.label("query set", &label);
    }
}

pub trait PrettyError: Error + Sized {
    fn fmt_pretty(&self, fmt: &mut ErrorFormatter) {
        fmt.error(self);
    }
}

pub fn format_pretty_any(
    writer: &mut dyn fmt::Write,
    global: &Global<IdentityManagerFactory>,
    error: &(dyn Error + 'static),
) {
    let mut fmt = ErrorFormatter { writer, global };

    if let Some(pretty_err) = error.downcast_ref::<ContextError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }

    if let Some(pretty_err) = error.downcast_ref::<crate::command::RenderCommandError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::binding_model::CreateBindGroupError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) =
        error.downcast_ref::<crate::binding_model::CreatePipelineLayoutError>()
    {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::ExecutionError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::RenderPassErrorInner>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::RenderPassError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::ComputePassErrorInner>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::ComputePassError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::RenderBundleError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::TransferError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::PassErrorScope>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::track::UsageConflict>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }
    if let Some(pretty_err) = error.downcast_ref::<crate::command::QueryError>() {
        return pretty_err.fmt_pretty(&mut fmt);
    }

    // default
    fmt.error(error)
}

#[derive(Debug)]
pub struct ContextError {
    pub string: &'static str,
    #[cfg(any(
        not(target_arch = "wasm32"),
        all(
            feature = "fragile-send-sync-non-atomic-wasm",
            not(target_feature = "atomics")
        )
    ))]
    pub cause: Box<dyn Error + Send + Sync + 'static>,
    #[cfg(not(any(
        not(target_arch = "wasm32"),
        all(
            feature = "fragile-send-sync-non-atomic-wasm",
            not(target_feature = "atomics")
        )
    )))]
    pub cause: Box<dyn Error + 'static>,
    pub label_key: &'static str,
    pub label: String,
}

impl PrettyError for ContextError {
    fn fmt_pretty(&self, fmt: &mut ErrorFormatter) {
        fmt.error(self);
        fmt.label(self.label_key, &self.label);
    }
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
