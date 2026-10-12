use alloc::{string::String, sync::Arc};

use crate::{
    binding_model::PipelineLayout,
    device::Device,
    pipeline::{self, CreatePipelineError, ShaderModuleState},
    resource::ParentDevice as _,
    validation,
};

/// Validates and compiles a pipeline's programmable stages, in pipeline order.
pub(super) struct StageCompiler<'d> {
    device: &'d Arc<Device>,
    primitive_topology: Option<wgt::PrimitiveTopology>,
    pub resources: validation::PipelineResources,
    /// The outputs of the most recently added stage.
    pub io: validation::StageIo,
    /// Stages checked against their Naga module's interface.
    pub validated: wgt::ShaderStages,
    /// Stages using passthrough shader modules, which we cannot check.
    pub passthrough: wgt::ShaderStages,
}

/// A validated stage, ready to be passed to `wgpu_hal`.
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
            passthrough: wgt::ShaderStages::empty(),
        }
    }

    /// Validate the next stage of the pipeline.
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
                self.passthrough |= stage_bit;
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
