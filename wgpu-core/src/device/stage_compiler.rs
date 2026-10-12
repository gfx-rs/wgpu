use alloc::{
    borrow::Cow,
    boxed::Box,
    string::{String, ToString as _},
    sync::Arc,
};
use core::mem::ManuallyDrop;

use crate::{
    binding_model::PipelineLayout,
    device::Device,
    hal_label,
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
    module: StageModule<'a>,
    pub entry_point: String,
    constants: Cow<'a, naga::back::PipelineConstants>,
    zero_initialize_workgroup_memory: bool,
}

enum StageModule<'a> {
    Generated(GeneratedModule<'a>),
    Passthrough(&'a dyn hal::DynShaderModule),
}

/// A `wgpu_hal` shader module generated for a single pipeline stage.
struct GeneratedModule<'a> {
    device: &'a Device,
    raw: ManuallyDrop<Box<dyn hal::DynShaderModule>>,
}

impl Drop for GeneratedModule<'_> {
    fn drop(&mut self) {
        // SAFETY: We never use `self.raw` again.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe { self.device.raw().destroy_shader_module(raw) };
    }
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
    ///
    /// For Naga shader modules, apply `desc`'s override values, validate the
    /// result, and generate a `wgpu_hal` shader module from it.
    pub fn compile<'a>(
        &mut self,
        desc: &'a pipeline::ProgrammableStageDescriptor<'_>,
        stage: validation::ShaderStageForValidation,
    ) -> Result<CompiledStage<'a>, CreatePipelineError>
    where
        'd: 'a,
    {
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

        match *state {
            ShaderModuleState::NagaModule {
                ref shader,
                runtime_checks,
            } => {
                let (module, info) = naga::back::pipeline_constants::process_overrides(
                    &shader.module,
                    &shader.info,
                    Some((stage.to_naga(), &entry_point)),
                    &desc.constants,
                )
                .map_err(|error| CreatePipelineError::PipelineConstants {
                    stage: stage_bit,
                    message: error.to_string(),
                })?;

                let interface =
                    validation::Interface::new(&module, &info, self.device.limits.clone());
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

                let hal_shader = hal::ShaderInput::Naga(hal::NagaShader {
                    module: Cow::Owned(module.into_owned()),
                    info: info.into_owned(),
                    debug_source: shader.debug_source.clone(),
                });
                let hal_desc = hal::ShaderModuleDescriptor {
                    label: hal_label(
                        Some(shader_module.label.as_str()).filter(|label| !label.is_empty()),
                        self.device.instance_flags,
                    ),
                    runtime_checks,
                };
                let raw = unsafe {
                    self.device
                        .raw()
                        .create_shader_module(&hal_desc, hal_shader)
                }
                .map_err(|error| match error {
                    hal::ShaderError::Device(error) => {
                        CreatePipelineError::Device(self.device.handle_hal_error(error))
                    }
                    hal::ShaderError::Compilation(message) => CreatePipelineError::Internal {
                        stage: stage_bit,
                        message,
                    },
                })?;

                Ok(CompiledStage {
                    module: StageModule::Generated(GeneratedModule {
                        device: self.device,
                        raw: ManuallyDrop::new(raw),
                    }),
                    entry_point,
                    // The overrides have already been applied.
                    constants: Cow::Owned(Default::default()),
                    zero_initialize_workgroup_memory: desc.zero_initialize_workgroup_memory,
                })
            }
            ShaderModuleState::Passthrough { ref raw, .. } => {
                self.passthrough |= stage_bit;
                Ok(CompiledStage {
                    module: StageModule::Passthrough(&**raw),
                    entry_point,
                    constants: Cow::Borrowed(&desc.constants),
                    zero_initialize_workgroup_memory: desc.zero_initialize_workgroup_memory,
                })
            }
        }
    }
}

impl CompiledStage<'_> {
    pub fn to_hal(&self) -> hal::ProgrammableStage<'_, dyn hal::DynShaderModule> {
        let module = match self.module {
            StageModule::Generated(ref generated) => &**generated.raw,
            StageModule::Passthrough(raw) => raw,
        };
        hal::ProgrammableStage {
            module,
            entry_point: &self.entry_point,
            constants: &self.constants,
            zero_initialize_workgroup_memory: self.zero_initialize_workgroup_memory,
        }
    }
}
