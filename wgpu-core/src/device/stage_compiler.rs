/*! The [`StageCompiler`] type, for validating programmable stages. */

use alloc::{string::String, sync::Arc};

use crate::{
    binding_model::PipelineLayout,
    device::Device,
    pipeline::{self, CreatePipelineError, ShaderModuleState},
    resource::ParentDevice as _,
    validation,
};

/// State for validating a pipeline's programmable stages.
///
/// Lifecycle:
///
/// -   Call [`StageCompiler::new`] to begin building a pipeline.
///
/// -   For each programmable stage, in processing order:
///
///     -   Call [`StageCompiler::compile`] to validate the stage. On success,
///         this updates the IO tracking to reflect the stage's outputs and
///         returns a [`CompiledStage`] referring to the hal module from the
///         descriptor.
///
/// -   Construct your `wgpu_hal` pipeline, populating the descriptor with
///     [`wgpu_hal::ProgrammableStage`]s obtained by calling [`to_hal`] on each
///     `CompiledStage`.
///
/// [`to_hal`]: CompiledStage::to_hal
pub(super) struct StageCompiler<'d> {
    device: &'d Arc<Device>,

    /// The primitive topology for render and mesh pipelines. `None` on compute pipelines.
    primitive_topology: Option<wgt::PrimitiveTopology>,

    /// Resource usage accumulated across all of the pipeline's stages.
    pub resources: validation::PipelineResources,

    /// The outputs of the last stage passed to `compile`.
    pub io: validation::StageIo,

    /// Stages checked against their Naga module's interface.
    pub validated: wgt::ShaderStages,
}

/// A validated pipeline stage, ready to be passed to `wgpu_hal`.
///
/// This borrows some stage's hal module while we create a pipeline
/// that uses it. Call [`to_hal`] to borrow a
/// [`wgpu_hal::ProgrammableStage`] from this that you can use in the
/// hal pipeline descriptor.
///
/// [`to_hal`]: Self::to_hal
pub(super) struct CompiledStage<'a> {
    module: &'a dyn hal::DynShaderModule,
    pub entry_point: String,
    constants: &'a naga::back::PipelineConstants,
    zero_initialize_workgroup_memory: bool,
}

impl<'d> StageCompiler<'d> {
    /// Begin validating a pipeline's stages.
    ///
    /// If `layout` is `None`, derive the pipeline's layout from its stages.
    /// The first stage's inputs are `inputs`.
    pub fn new(
        device: &'d Arc<Device>,
        layout: Option<Arc<PipelineLayout>>,
        inputs: validation::StageIo,
        primitive_topology: Option<wgt::PrimitiveTopology>,
    ) -> Self {
        let layout = match layout {
            Some(layout) => validation::BindingLayoutSource::Provided(layout),
            None => validation::BindingLayoutSource::new_derived(&device.limits),
        };
        Self {
            device,
            primitive_topology,
            resources: validation::PipelineResources::new(layout),
            io: inputs,
            validated: wgt::ShaderStages::empty(),
        }
    }

    /// Validate the next stage of the pipeline.
    ///
    /// For Naga shader modules, validate the module, and update `self` to
    /// reflect this stage's outputs and resource usage.
    ///
    /// Passthrough modules cannot be checked, and are an error if the
    /// pipeline's layout is being derived.
    ///
    /// In either case, on success, return a [`CompiledStage`] that borrows the
    /// hal module.
    pub fn compile<'a>(
        &mut self,
        desc: &'a pipeline::ProgrammableStageDescriptor<'_>,
        stage: validation::ShaderStageForValidation,
    ) -> Result<CompiledStage<'a>, CreatePipelineError> {
        let stage_bit = stage.to_wgt_bit();
        let stage_err = |error| CreatePipelineError::Stage {
            stage: stage_bit,
            error,
        };

        let shader_module = &desc.module;
        let state = shader_module
            .state()
            .map_err(Into::into)
            .map_err(stage_err)?;
        shader_module.same_device(self.device)?;

        let entry_point = shader_module
            .finalize_entry_point_name(stage.to_naga(), desc.entry_point.as_deref())
            .map_err(stage_err)?;

        let module = match *state {
            ShaderModuleState::NagaModule {
                ref interface,
                ref raw,
            } => {
                self.io = interface
                    .check_stage(
                        &mut self.resources,
                        &entry_point,
                        stage,
                        core::mem::take(&mut self.io),
                        self.primitive_topology,
                    )
                    .map_err(stage_err)?;
                self.validated |= stage_bit;
                &**raw
            }
            ShaderModuleState::Passthrough { ref raw, .. } => {
                if let validation::BindingLayoutSource::Derived(_) = self.resources.layout {
                    return Err(CreatePipelineError::Implicit(
                        pipeline::ImplicitLayoutError::Passthrough(stage_bit),
                    ));
                }
                &**raw
            }
        };

        Ok(CompiledStage {
            module,
            entry_point,
            constants: &desc.constants,
            zero_initialize_workgroup_memory: desc.zero_initialize_workgroup_memory,
        })
    }
}

impl CompiledStage<'_> {
    pub fn to_hal(&self) -> hal::ProgrammableStage<'_, dyn hal::DynShaderModule> {
        hal::ProgrammableStage {
            module: self.module,
            entry_point: &self.entry_point,
            constants: self.constants,
            zero_initialize_workgroup_memory: self.zero_initialize_workgroup_memory,
        }
    }
}
