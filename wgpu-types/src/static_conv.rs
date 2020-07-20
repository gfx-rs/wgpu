use super::*;

use std::borrow::Cow;

pub trait ToStatic: Clone {
    type Static: Clone + 'static;
    fn to_static(&self) -> Self::Static;
}

macro_rules! impl_to_static {
    (clone $t:ty) => {
        impl ToStatic for $t {
            type Static = $t;
            fn to_static(&self) -> Self::Static {
                <$t as Clone>::clone(self)
            }
        }
    };
}

impl<'a> ToStatic for Cow<'a, str> {
    type Static = Cow<'static, str>;
    fn to_static(&self) -> Self::Static {
        self[..].to_owned().into()
    }
}

impl<'a, T> ToStatic for Cow<'a, [T]>
where
    T: ToStatic,
{
    type Static = Cow<'static, [T::Static]>;
    fn to_static(&self) -> Self::Static {
        Cow::Owned(self.iter().map(|v| v.to_static()).collect())
    }
}

impl<T> ToStatic for Option<T>
where
    T: ToStatic,
{
    type Static = Option<T::Static>;
    fn to_static(&self) -> Self::Static {
        self.as_ref().map(|v| v.to_static())
    }
}

impl<L, B> ToStatic for BindGroupDescriptor<'_, L, B>
where
    B: ToStatic,
    L: ToStatic,
{
    type Static = BindGroupDescriptor<'static, L::Static, B::Static>;
    fn to_static(&self) -> Self::Static {
        BindGroupDescriptor {
            label: self.label.to_static(),
            layout: self.layout.to_static(),
            entries: self.entries.to_static(),
        }
    }
}

impl_to_static!(clone PushConstantRange);

impl<B> ToStatic for PipelineLayoutDescriptor<'_, B>
where
    B: ToStatic,
{
    type Static = PipelineLayoutDescriptor<'static, B::Static>;
    fn to_static(&self) -> Self::Static {
        PipelineLayoutDescriptor {
            bind_group_layouts: self.bind_group_layouts.to_static(),
            push_constant_ranges: self.push_constant_ranges.to_static(),
        }
    }
}

impl<M> ToStatic for ProgrammableStageDescriptor<'_, M>
where
    M: ToStatic,
{
    type Static = ProgrammableStageDescriptor<'static, M::Static>;
    fn to_static(&self) -> Self::Static {
        ProgrammableStageDescriptor {
            module: self.module.to_static(),
            entry_point: self.entry_point.to_static(),
        }
    }
}

impl_to_static!(clone ColorStateDescriptor);

impl<L, D> ToStatic for RenderPipelineDescriptor<'_, L, D>
where
    L: ToStatic,
    D: ToStatic,
{
    type Static = RenderPipelineDescriptor<'static, L::Static, D::Static>;
    fn to_static(&self) -> Self::Static {
        RenderPipelineDescriptor {
            layout: self.layout.to_static(),
            vertex_stage: self.vertex_stage.to_static(),
            fragment_stage: self.fragment_stage.to_static(),
            rasterization_state: self.rasterization_state.clone(),
            primitive_topology: self.primitive_topology,
            color_states: self.color_states.to_static(),
            depth_stencil_state: self.depth_stencil_state.clone(),
            vertex_state: self.vertex_state.to_static(),
            sample_count: self.sample_count,
            sample_mask: self.sample_mask,
            alpha_to_coverage_enabled: self.alpha_to_coverage_enabled,
        }
    }
}

impl_to_static!(clone TextureFormat);

impl ToStatic for RenderBundleEncoderDescriptor<'_> {
    type Static = RenderBundleEncoderDescriptor<'static>;
    fn to_static(&self) -> Self::Static {
        RenderBundleEncoderDescriptor {
            label: self.label.to_static(),
            color_formats: self.color_formats.to_static(),
            depth_stencil_format: self.depth_stencil_format,
            sample_count: self.sample_count,
        }
    }
}

impl_to_static!(clone BindGroupLayoutEntry);

impl ToStatic for BindGroupLayoutDescriptor<'_> {
    type Static = BindGroupLayoutDescriptor<'static>;
    fn to_static(&self) -> Self::Static {
        BindGroupLayoutDescriptor {
            label: self.label.to_static(),
            entries: self.entries.to_static(),
        }
    }
}

impl_to_static!(clone VertexAttributeDescriptor);

impl ToStatic for VertexBufferDescriptor<'_> {
    type Static = VertexBufferDescriptor<'static>;
    fn to_static(&self) -> Self::Static {
        VertexBufferDescriptor {
            stride: self.stride,
            step_mode: self.step_mode,
            attributes: self.attributes.to_static(),
        }
    }
}

impl ToStatic for VertexStateDescriptor<'_> {
    type Static = VertexStateDescriptor<'static>;
    fn to_static(&self) -> Self::Static {
        VertexStateDescriptor {
            index_format: self.index_format,
            vertex_buffers: self.vertex_buffers.to_static(),
        }
    }
}

impl<C, D> ToStatic for RenderPassDescriptor<'_, C, D>
where
    C: ToStatic,
    D: ToStatic,
{
    type Static = RenderPassDescriptor<'static, C::Static, D::Static>;
    fn to_static(&self) -> Self::Static {
        RenderPassDescriptor {
            color_attachments: self.color_attachments.to_static(),
            depth_stencil_attachment: self.depth_stencil_attachment.to_static(),
        }
    }
}

impl<L, D> ToStatic for ComputePipelineDescriptor<L, D>
where
    L: ToStatic,
    D: ToStatic,
{
    type Static = ComputePipelineDescriptor<L::Static, D::Static>;
    fn to_static(&self) -> Self::Static {
        ComputePipelineDescriptor {
            layout: self.layout.to_static(),
            compute_stage: self.compute_stage.to_static(),
        }
    }
}
impl<R> ToStatic for BindGroupEntry<R>
where
    R: ToStatic,
{
    type Static = BindGroupEntry<R::Static>;
    fn to_static(&self) -> Self::Static {
        BindGroupEntry {
            binding: self.binding,
            resource: self.resource.to_static(),
        }
    }
}

impl<L> ToStatic for RenderBundleDescriptor<L>
where
    L: ToStatic,
{
    type Static = RenderBundleDescriptor<L::Static>;
    fn to_static(&self) -> Self::Static {
        RenderBundleDescriptor {
            label: self.label.to_static(),
        }
    }
}
