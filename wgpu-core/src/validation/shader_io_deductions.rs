use core::fmt::{self, Debug, Display, Formatter};

#[cfg(doc)]
#[expect(unused_imports)]
use crate::validation::StageError;

/// Max shader I/O variable deductions for vertex shader output. Used by
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

pub(in crate::validation) fn display_deductions_as_optional_list<T>(
    deductions: &[T],
    accessor: fn(&T) -> u32,
) -> impl Display + '_
where
    T: Debug,
{
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
                writeln!(f, "\n- {deduction:?}: {}", accessor(deduction))?;
            }
        }
        Ok(())
    })
}
