use core::fmt::{self, Debug, Display, Formatter};

#[cfg(doc)]
#[expect(unused_imports)]
use crate::validation::StageError;

/// A [`ShaderIoDeduction`] for vertex shader output. Used by
/// [`StageError::TooManyUserDefinedVertexOutputs`] and
/// [`StageError::VertexOutputLocationTooLarge`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxVertexShaderOutputDeduction {
    /// When a pipeline's [`crate::pipeline::RenderPipelineDescriptor::primitive`] is set to
    /// [`wgt::PrimitiveTopology::PointList`].
    PointListPrimitiveTopology,
}

impl MaxVertexShaderOutputDeduction {
    pub fn for_variables(self) -> u32 {
        match self {
            Self::PointListPrimitiveTopology => 1,
        }
    }

    pub fn for_location(self) -> u32 {
        match self {
            Self::PointListPrimitiveTopology => 0,
        }
    }
}

/// A [`ShaderIoDeduction`] for vertex shader output. Used by TODO.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxFragmentShaderInputDeduction {
    InterStageBuiltIn(InterStageBuiltIn),
}

impl MaxFragmentShaderInputDeduction {
    pub fn for_variables(self) -> u32 {
        match self {
            Self::InterStageBuiltIn(_builtin) => 1,
        }
    }

    pub fn from_inter_stage_builtin(builtin: crate::validation::BuiltIn) -> Option<Self> {
        use crate::validation::BuiltIn;

        Some(Self::InterStageBuiltIn(match builtin {
            BuiltIn::FrontFacing => InterStageBuiltIn::FrontFacing,
            BuiltIn::SampleIndex => InterStageBuiltIn::SampleIndex,
            BuiltIn::SampleMask => InterStageBuiltIn::SampleMask,
            BuiltIn::PrimitiveIndex => InterStageBuiltIn::PrimitiveIndex,
            BuiltIn::SubgroupSize => InterStageBuiltIn::SubgroupSize,
            BuiltIn::SubgroupInvocationId => InterStageBuiltIn::SubgroupInvocationId,

            BuiltIn::Position { .. } => InterStageBuiltIn::Position,
            BuiltIn::ViewIndex => InterStageBuiltIn::ViewIndex,

            BuiltIn::BaseInstance
            | BuiltIn::BaseVertex
            | BuiltIn::ClipDistance { .. }
            | BuiltIn::CullDistance
            | BuiltIn::InstanceIndex
            | BuiltIn::PointSize
            | BuiltIn::VertexIndex
            | BuiltIn::DrawID
            | BuiltIn::FragDepth
            | BuiltIn::PointCoord
            | BuiltIn::Barycentric
            | BuiltIn::GlobalInvocationId
            | BuiltIn::LocalInvocationId
            | BuiltIn::LocalInvocationIndex
            | BuiltIn::WorkGroupId
            | BuiltIn::WorkGroupSize
            | BuiltIn::NumWorkGroups
            | BuiltIn::NumSubgroups
            | BuiltIn::SubgroupId
            | BuiltIn::MeshTaskSize
            | BuiltIn::CullPrimitive
            | BuiltIn::PointIndex
            | BuiltIn::LineIndices
            | BuiltIn::TriangleIndices
            | BuiltIn::VertexCount
            | BuiltIn::Vertices
            | BuiltIn::PrimitiveCount
            | BuiltIn::Primitives => None,
        }))
    }
}

/// A [`naga::BuiltIn`] that counts towards
/// a [`MaxFragmentShaderInputDeduction::InterStageBuiltIn`].
///
/// See also <https://www.w3.org/TR/webgpu/#inter-stage-builtins>.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterStageBuiltIn {
    // Standard for WebGPU
    FrontFacing,
    SampleIndex,
    SampleMask,
    PrimitiveIndex,
    SubgroupInvocationId,
    SubgroupSize,

    // Non-standard
    // TODO: Is this list actually good?
    Position,
    ViewIndex,
}

pub(in crate::validation) fn display_deductions_as_optional_list<T>(
    deductions: &[T],
    accessor: fn(&T) -> u32,
) -> impl Display + '_ {
    struct DisplayFromFn<F>(F);

    impl<F> Display for DisplayFromFn<F>
    where
        F: Fn(&mut Formatter<'_>) -> fmt::Result,
    {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            let Self(inner) = self;
            inner(f)
        }
    }

    DisplayFromFn(move |f: &mut Formatter<'_>| {
        let relevant_deductions = deductions
            .iter()
            .map(|deduction| (deduction, accessor(deduction)))
            .filter(|(_, effective_deduction)| *effective_deduction > 0);
        if relevant_deductions.clone().next().is_some() {
            writeln!(f, "; note that some deductions apply during validation:")?;
            for deduction in deductions {
                let deduction = accessor(deduction);
                if deduction > 0 {
                    writeln!(f, "\n- {deduction:?}: {}", deduction)?;
                }
            }
        }
        Ok(())
    })
}
