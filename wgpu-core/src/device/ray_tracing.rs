use alloc::{string::ToString, sync::Arc, vec::Vec};
use core::mem::{size_of, ManuallyDrop};

#[cfg(feature = "trace")]
use crate::device::trace::{Action, IntoTrace};
use crate::device::{DeviceError, ENTRYPOINT_FAILURE_ERROR};
use crate::resource::{ParentDevice, ResourceState};
use crate::{
    api_log,
    device::Device,
    hal_label,
    lock::RwLock,
    lock::{rank, Mutex},
    ray_tracing::{CreateBlasError, CreateTlasError},
    resource,
    resource::{BlasCompactState, TrackingData},
    snatch::Snatchable,
    LabelHelpers,
};
use crate::{pipeline, FastHashMap};
use hal::AccelerationStructureTriangleIndices;
use wgt::{Features, AABB_GEOMETRY_MIN_STRIDE};

impl Device {
    pub fn create_blas(
        self: &Arc<Self>,
        blas_desc: &resource::BlasDescriptor,
        sizes: wgt::BlasGeometrySizeDescriptors,
    ) -> (Arc<resource::Blas>, Option<CreateBlasError>) {
        #[cfg(feature = "trace")]
        let trace_sizes = sizes.clone();

        let (blas, error) = match self.create_blas_inner(blas_desc, sizes) {
            Ok(blas) => (blas, None),
            Err(err) => (resource::Blas::invalid(self.clone(), blas_desc), Some(err)),
        };

        #[cfg(feature = "trace")]
        if let Some(trace) = self.trace.lock().as_mut() {
            trace.add(Action::CreateBlas {
                id: blas.to_trace(),
                desc: blas_desc.clone(),
                sizes: trace_sizes,
            });
        }

        api_log!("Device::create_blas -> {:?}", Arc::as_ptr(&blas));
        (blas, error)
    }
    pub(crate) fn create_blas_inner(
        self: &Arc<Self>,
        blas_desc: &resource::BlasDescriptor,
        sizes: wgt::BlasGeometrySizeDescriptors,
    ) -> Result<Arc<resource::Blas>, CreateBlasError> {
        self.check_is_valid()?;
        self.require_features(Features::EXPERIMENTAL_RAY_QUERY)
            .or_else(|_| self.require_features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES))?;

        if blas_desc
            .flags
            .contains(wgt::AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN)
        {
            self.require_features(Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN)?;
        }

        let size_info = match &sizes {
            wgt::BlasGeometrySizeDescriptors::Triangles { descriptors } => {
                if descriptors.len() as u32 > self.limits.max_blas_geometry_count {
                    return Err(CreateBlasError::TooManyGeometries(
                        self.limits.max_blas_geometry_count,
                        descriptors.len() as u32,
                    ));
                }

                let mut entries =
                    Vec::<hal::AccelerationStructureTriangles<dyn hal::DynBuffer>>::with_capacity(
                        descriptors.len(),
                    );
                for desc in descriptors {
                    if desc.index_count.is_some() != desc.index_format.is_some() {
                        return Err(CreateBlasError::MissingIndexData);
                    }
                    let indices =
                        desc.index_count
                            .map(|count| AccelerationStructureTriangleIndices::<
                                dyn hal::DynBuffer,
                            > {
                                format: desc.index_format.unwrap(),
                                buffer: Some(self.zero_buffer.as_ref()),
                                offset: 0,
                                count,
                            });
                    if !self
                        .features
                        .allowed_vertex_formats_for_blas()
                        .contains(&desc.vertex_format)
                    {
                        return Err(CreateBlasError::InvalidVertexFormat(
                            desc.vertex_format,
                            self.features.allowed_vertex_formats_for_blas(),
                        ));
                    }

                    let mut transform = None;

                    if blas_desc
                        .flags
                        .contains(wgt::AccelerationStructureFlags::USE_TRANSFORM)
                    {
                        transform = Some(wgpu_hal::AccelerationStructureTriangleTransform {
                            buffer: self.zero_buffer.as_ref(),
                            offset: 0,
                        })
                    }

                    if desc.vertex_count > self.limits.max_blas_primitive_count {
                        return Err(CreateBlasError::TooManyPrimitives(
                            self.limits.max_blas_primitive_count,
                            desc.vertex_count,
                        ));
                    }

                    entries.push(hal::AccelerationStructureTriangles::<dyn hal::DynBuffer> {
                        vertex_buffer: Some(self.zero_buffer.as_ref()),
                        vertex_format: desc.vertex_format,
                        first_vertex: 0,
                        vertex_count: desc.vertex_count,
                        vertex_stride: 0,
                        indices,
                        transform,
                        flags: desc.flags,
                    });
                }
                unsafe {
                    self.raw().get_acceleration_structure_build_sizes(
                        &hal::GetAccelerationStructureBuildSizesDescriptor {
                            entries: &hal::AccelerationStructureEntries::Triangles(entries),
                            flags: blas_desc.flags,
                        },
                    )
                }
            }
            wgt::BlasGeometrySizeDescriptors::AABBs { descriptors } => {
                if descriptors.len() as u32 > self.limits.max_blas_geometry_count {
                    return Err(CreateBlasError::TooManyGeometries(
                        self.limits.max_blas_geometry_count,
                        descriptors.len() as u32,
                    ));
                }

                let mut entries =
                    Vec::<hal::AccelerationStructureAABBs<dyn hal::DynBuffer>>::with_capacity(
                        descriptors.len(),
                    );
                for desc in descriptors {
                    if desc.primitive_count > self.limits.max_blas_primitive_count {
                        return Err(CreateBlasError::TooManyPrimitives(
                            self.limits.max_blas_primitive_count,
                            desc.primitive_count,
                        ));
                    }

                    entries.push(hal::AccelerationStructureAABBs::<dyn hal::DynBuffer> {
                        buffer: Some(self.zero_buffer.as_ref()),
                        offset: 0,
                        count: desc.primitive_count,
                        stride: AABB_GEOMETRY_MIN_STRIDE,
                        flags: desc.flags,
                    });
                }
                unsafe {
                    self.raw().get_acceleration_structure_build_sizes(
                        &hal::GetAccelerationStructureBuildSizesDescriptor {
                            entries: &hal::AccelerationStructureEntries::AABBs(entries),
                            flags: blas_desc.flags,
                        },
                    )
                }
            }
        };

        let raw = unsafe {
            self.raw()
                .create_acceleration_structure(&hal::AccelerationStructureDescriptor {
                    label: hal_label(blas_desc.label.as_deref(), self.instance_flags),
                    size: size_info.acceleration_structure_size,
                    format: hal::AccelerationStructureFormat::BottomLevel,
                    allow_compaction: blas_desc
                        .flags
                        .contains(wgpu_types::AccelerationStructureFlags::ALLOW_COMPACTION),
                })
        }
        .map_err(|e| self.handle_hal_error_with_nonfatal_oom(e))?;

        let compaction_buffer = if blas_desc
            .flags
            .contains(wgpu_types::AccelerationStructureFlags::ALLOW_COMPACTION)
        {
            Some(ManuallyDrop::new(unsafe {
                self.raw()
                    .create_buffer(&hal::BufferDescriptor {
                        label: hal_label(
                            Some("(wgpu internal) compaction read-back buffer"),
                            self.instance_flags,
                        ),
                        size: size_of::<wgpu_types::BufferAddress>() as wgpu_types::BufferAddress,
                        usage: wgpu_types::BufferUses::ACCELERATION_STRUCTURE_QUERY
                            | wgpu_types::BufferUses::MAP_READ,
                        memory_flags: hal::MemoryFlags::PREFER_COHERENT,
                    })
                    .map_err(DeviceError::from_hal)?
            }))
        } else {
            None
        };

        let handle = unsafe {
            self.raw()
                .get_acceleration_structure_device_address(raw.as_ref())
        };

        Ok(Arc::new(resource::Blas {
            state: ResourceState::Valid(resource::BlasState {
                raw: Snatchable::new(raw),
            }),
            device: self.clone(),
            size_info,
            sizes,
            flags: blas_desc.flags,
            update_mode: blas_desc.update_mode,
            handle,
            label: blas_desc.label.to_string(),
            built_index: RwLock::new(rank::BLAS_BUILT_INDEX, None),
            tracking_data: TrackingData::new(self.tracker_indices.blas_s.clone()),
            compaction_buffer,
            compacted_state: Mutex::new(rank::BLAS_COMPACTION_STATE, BlasCompactState::Idle),
        }))
    }

    pub fn create_tlas(
        self: &Arc<Self>,
        desc: &resource::TlasDescriptor,
    ) -> (Arc<resource::Tlas>, Option<CreateTlasError>) {
        let (tlas, error) = match self.create_tlas_inner(desc) {
            Ok(tlas) => (tlas, None),
            Err(e) => (resource::Tlas::invalid(Arc::clone(self), desc), Some(e)),
        };
        #[cfg(feature = "trace")]
        if let Some(trace) = self.trace.lock().as_mut() {
            trace.add(Action::CreateTlas {
                id: tlas.to_trace(),
                desc: desc.clone(),
            });
        }

        api_log!("Device::create_tlas -> {:?}", Arc::as_ptr(&tlas));

        (tlas, error)
    }

    pub(crate) fn create_tlas_inner(
        self: &Arc<Self>,
        desc: &resource::TlasDescriptor,
    ) -> Result<Arc<resource::Tlas>, CreateTlasError> {
        self.check_is_valid()?;
        self.require_features(Features::EXPERIMENTAL_RAY_QUERY)
            .or_else(|_| self.require_features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES))?;

        if desc.max_instances > self.limits.max_tlas_instance_count {
            return Err(CreateTlasError::TooManyInstances(
                self.limits.max_tlas_instance_count,
                desc.max_instances,
            ));
        }

        if desc
            .flags
            .contains(wgt::AccelerationStructureFlags::USE_TRANSFORM)
        {
            return Err(CreateTlasError::DisallowedFlag(
                wgt::AccelerationStructureFlags::USE_TRANSFORM,
            ));
        }

        if desc
            .flags
            .contains(wgt::AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN)
        {
            self.require_features(Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN)?;
        }

        let size_info = unsafe {
            self.raw().get_acceleration_structure_build_sizes(
                &hal::GetAccelerationStructureBuildSizesDescriptor {
                    entries: &hal::AccelerationStructureEntries::Instances(
                        hal::AccelerationStructureInstances {
                            buffer: Some(self.zero_buffer.as_ref()),
                            offset: 0,
                            count: desc.max_instances,
                        },
                    ),
                    flags: desc.flags,
                },
            )
        };

        let raw = unsafe {
            self.raw()
                .create_acceleration_structure(&hal::AccelerationStructureDescriptor {
                    label: hal_label(desc.label.as_deref(), self.instance_flags),
                    size: size_info.acceleration_structure_size,
                    format: hal::AccelerationStructureFormat::TopLevel,
                    allow_compaction: false,
                })
        }
        .map_err(|e| self.handle_hal_error_with_nonfatal_oom(e))?;

        let instance_buffer_size = self
            .alignments
            .raw_tlas_instance_size
            .checked_mul(desc.max_instances.max(1))
            .expect("max_tlas_instance_count should not allow excessive buffer size");
        let instance_buffer = unsafe {
            self.raw().create_buffer(&hal::BufferDescriptor {
                label: hal_label(Some("(wgpu-core) instances_buffer"), self.instance_flags),
                size: u64::from(instance_buffer_size),
                usage: wgt::BufferUses::COPY_DST
                    | wgt::BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
                memory_flags: hal::MemoryFlags::PREFER_COHERENT,
            })
        }
        .map_err(|e| self.handle_hal_error_with_nonfatal_oom(e))?;

        Ok(Arc::new(resource::Tlas {
            state: ResourceState::Valid(resource::TlasState {
                raw: Snatchable::new(raw),
                instance_buffer,
            }),
            device: self.clone(),
            size_info,
            flags: desc.flags,
            update_mode: desc.update_mode,
            built_index: RwLock::new(rank::TLAS_BUILT_INDEX, None),
            dependencies: RwLock::new(rank::TLAS_DEPENDENCIES, Vec::new()),
            label: desc.label.to_string(),
            max_instance_count: desc.max_instances,
            tracking_data: TrackingData::new(self.tracker_indices.tlas_s.clone()),
            max_intersection_index: RwLock::new(rank::TLAS_MAX_INTERSECTION_IDX, 0),
        }))
    }

    pub fn create_ray_tracing_pipeline(
        self: &Arc<Self>,
        desc: pipeline::RayTracingPipelineDescriptor,
    ) -> (
        Arc<pipeline::RayTracingPipeline>,
        Option<pipeline::CreateRayTracingPipelineError>,
    ) {
        let (ray_tracing_pipeline, error) =
            match self.create_ray_tracing_pipeline_inner(desc.clone()) {
                Ok(ray_tracing_pipeline) => (ray_tracing_pipeline, None),
                Err(error) => (
                    pipeline::RayTracingPipeline::invalid(self.clone(), desc.label.to_string()),
                    Some(error),
                ),
            };
        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *self.trace.lock() {
            use crate::device::trace::IntoTrace;
            trace.add(Action::CreateRayTracingPipeline {
                id: ray_tracing_pipeline.to_trace(),
                desc: desc.to_trace(),
            });
        }
        (ray_tracing_pipeline, error)
    }

    pub fn create_ray_tracing_pipeline_inner(
        self: &Arc<Self>,
        desc: pipeline::RayTracingPipelineDescriptor,
    ) -> Result<Arc<pipeline::RayTracingPipeline>, pipeline::CreateRayTracingPipelineError> {
        use crate::validation;

        self.check_is_valid()?;
        self.require_features(Features::EXPERIMENTAL_RAY_TRACING_PIPELINES)?;

        let mut shader_binding_sizes = FastHashMap::default();

        let mut io = validation::StageIo::default();

        let is_auto_layout = desc.layout.is_none();

        // Get the pipeline layout from the desc if it is provided.
        let pipeline_layout = match desc.layout {
            Some(pipeline_layout) => {
                pipeline_layout.same_device(self)?;
                Some(pipeline_layout)
            }
            None => None,
        };

        let mut binding_layout_source = match pipeline_layout {
            Some(pipeline_layout) => validation::BindingLayoutSource::Provided(pipeline_layout),
            None => validation::BindingLayoutSource::new_derived(&self.limits),
        };

        let final_ray_gen_name;
        let ray_generation = {
            let stage = validation::ShaderStageForValidation::RayGeneration;
            let stage_bit = stage.to_wgt_bit();

            final_ray_gen_name = desc
                .ray_generation
                .module
                .finalize_entry_point_name(
                    stage.to_naga(),
                    desc.ray_generation
                        .entry_point
                        .as_ref()
                        .map(|ep| ep.as_ref()),
                )
                .map_err(|e| pipeline::CreateRayTracingPipelineError::Stage {
                    stage: stage_bit,
                    error: e,
                })?;

            let shader_module = &desc.ray_generation.module;
            let shader_module_state = shader_module.state()?;
            shader_module.same_device(self)?;
            if let Some(interface) = shader_module_state.interface.interface() {
                io = interface
                    .check_stage(
                        &mut binding_layout_source,
                        &mut shader_binding_sizes,
                        &final_ray_gen_name,
                        stage,
                        io,
                        None,
                    )
                    .map_err(|e| pipeline::CreateRayTracingPipelineError::Stage {
                        stage: stage_bit,
                        error: e,
                    })?;
            }

            hal::ProgrammableStage {
                module: shader_module_state.raw.as_ref(),
                entry_point: &final_ray_gen_name,
                constants: &desc.ray_generation.constants,
                zero_initialize_workgroup_memory: desc
                    .ray_generation
                    .zero_initialize_workgroup_memory,
            }
        };

        let final_miss_name;
        let miss = {
            let stage = validation::ShaderStageForValidation::Miss;
            let stage_bit = stage.to_wgt_bit();

            final_miss_name = desc
                .miss
                .module
                .finalize_entry_point_name(
                    stage.to_naga(),
                    desc.miss.entry_point.as_ref().map(|ep| ep.as_ref()),
                )
                .map_err(|e| pipeline::CreateRayTracingPipelineError::Stage {
                    stage: stage_bit,
                    error: e,
                })?;

            let shader_module = &desc.miss.module;
            let shader_module_state = shader_module.state()?;
            shader_module.same_device(self)?;
            if let Some(interface) = shader_module_state.interface.interface() {
                io = interface
                    .check_stage(
                        &mut binding_layout_source,
                        &mut shader_binding_sizes,
                        &final_miss_name,
                        stage,
                        io,
                        None,
                    )
                    .map_err(|e| pipeline::CreateRayTracingPipelineError::Stage {
                        stage: stage_bit,
                        error: e,
                    })?;
            }

            hal::ProgrammableStage {
                module: shader_module_state.raw.as_ref(),
                entry_point: &final_miss_name,
                constants: &desc.miss.constants,
                zero_initialize_workgroup_memory: desc.miss.zero_initialize_workgroup_memory,
            }
        };

        if desc.intersections.len() > 1 << 24 {
            return Err(
                pipeline::CreateRayTracingPipelineError::TooManyIntersectionGroups(
                    desc.intersections.len(),
                ),
            );
        }

        if desc.max_recursion_depth > self.limits.max_ray_recursion_depth {
            return Err(
                pipeline::CreateRayTracingPipelineError::TooHighRayRecursionDepth(
                    desc.max_recursion_depth,
                    self.limits.max_ray_recursion_depth,
                ),
            );
        }

        let mut intersections = Vec::with_capacity(desc.intersections.len());
        let mut final_intersection_names = Vec::with_capacity(desc.intersections.len());

        for intersection in &desc.intersections {
            match intersection {
                pipeline::RayTracingIntersectionDescriptor::Triangle {
                    closest_hit,
                    any_hit,
                } => {
                    let stage = validation::ShaderStageForValidation::ClosestHit { triangle: true };
                    let closest_name = closest_hit
                        .module
                        .finalize_entry_point_name(
                            stage.to_naga(),
                            closest_hit.entry_point.as_ref().map(|ep| ep.as_ref()),
                        )
                        .map_err(|e| pipeline::CreateRayTracingPipelineError::Stage {
                            stage: stage.to_wgt_bit(),
                            error: e,
                        })?;

                    let any_hit = match any_hit {
                        Some(any_hit) => {
                            let stage =
                                validation::ShaderStageForValidation::AnyHit { triangle: true };

                            Some(
                                any_hit
                                    .module
                                    .finalize_entry_point_name(
                                        stage.to_naga(),
                                        any_hit.entry_point.as_ref().map(|ep| ep.as_ref()),
                                    )
                                    .map_err(|e| {
                                        pipeline::CreateRayTracingPipelineError::Stage {
                                            stage: stage.to_wgt_bit(),
                                            error: e,
                                        }
                                    })?,
                            )
                        }
                        None => None,
                    };

                    final_intersection_names.push((closest_name, any_hit));
                }
            }
        }

        for (intersection, (final_closest_name, final_any_name)) in desc
            .intersections
            .iter()
            .zip(final_intersection_names.iter())
        {
            intersections.push(match intersection {
                pipeline::RayTracingIntersectionDescriptor::Triangle {
                    closest_hit,
                    any_hit,
                } => {
                    let closest_hit = {
                        let stage =
                            validation::ShaderStageForValidation::ClosestHit { triangle: true };

                        let stage_bits = stage.to_wgt_bit();
                        let shader_module = &closest_hit.module;
                        let shader_module_state = shader_module.state()?;
                        shader_module.same_device(self)?;
                        if let Some(interface) = shader_module_state.interface.interface() {
                            io = interface
                                .check_stage(
                                    &mut binding_layout_source,
                                    &mut shader_binding_sizes,
                                    final_closest_name,
                                    stage,
                                    io,
                                    None,
                                )
                                .map_err(|e| pipeline::CreateRayTracingPipelineError::Stage {
                                    stage: stage_bits,
                                    error: e,
                                })?;
                        }

                        hal::ProgrammableStage {
                            module: shader_module_state.raw.as_ref(),
                            entry_point: final_closest_name,
                            constants: &closest_hit.constants,
                            zero_initialize_workgroup_memory: closest_hit
                                .zero_initialize_workgroup_memory,
                        }
                    };

                    let any_hit = match any_hit {
                        Some(any_hit) => {
                            let stage =
                                validation::ShaderStageForValidation::AnyHit { triangle: true };

                            let final_any_name = final_any_name.as_ref().unwrap();

                            let stage_bits = stage.to_wgt_bit();

                            let shader_module = &any_hit.module;
                            let shader_module_state = shader_module.state()?;
                            shader_module.same_device(self)?;
                            if let Some(interface) = shader_module_state.interface.interface() {
                                io = interface
                                    .check_stage(
                                        &mut binding_layout_source,
                                        &mut shader_binding_sizes,
                                        final_any_name,
                                        stage,
                                        io,
                                        None,
                                    )
                                    .map_err(|e| {
                                        pipeline::CreateRayTracingPipelineError::Stage {
                                            stage: stage_bits,
                                            error: e,
                                        }
                                    })?;
                            }

                            Some(hal::ProgrammableStage {
                                module: shader_module_state.raw.as_ref(),
                                entry_point: final_any_name,
                                constants: &any_hit.constants,
                                zero_initialize_workgroup_memory: any_hit
                                    .zero_initialize_workgroup_memory,
                            })
                        }
                        None => None,
                    };

                    hal::RayObjectIntersectionState {
                        closest_hit,
                        any_hit,
                    }
                }
            });
        }

        if !self
            .downlevel
            .flags
            .contains(wgt::DownlevelFlags::BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED)
        {
            for (binding, size) in shader_binding_sizes.iter() {
                if size.get() % 16 != 0 {
                    return Err(pipeline::CreateRayTracingPipelineError::UnalignedShader {
                        binding: binding.binding,
                        group: binding.group,
                        size: size.get(),
                    });
                }
            }
        }

        let pipeline_layout = match binding_layout_source {
            validation::BindingLayoutSource::Provided(layout) => layout,
            validation::BindingLayoutSource::Derived(entries) => {
                self.create_derived_pipeline_layout(entries, io.immediates.size())?
            }
        };

        let late_sized_buffer_groups =
            Device::make_late_sized_buffer_groups(&shader_binding_sizes, &pipeline_layout);

        let cache = match desc.cache {
            Some(cache) => {
                cache.same_device(self)?;
                Some(cache)
            }
            None => None,
        };

        let raw = {
            let pipeline_desc = hal::RayTracingPipelineDescriptor {
                label: desc.label.to_hal(self.instance_flags),
                layout: pipeline_layout.raw()?,
                ray_generation,
                miss,
                intersection: &intersections,
                max_recursion_depth: desc.max_recursion_depth,
                cache: cache.as_ref().map(|it| it.raw()).transpose()?,
            };
            unsafe { self.raw().create_ray_tracing_pipeline(&pipeline_desc) }.map_err(|err| {
                match err {
                    hal::PipelineError::Device(error) => {
                        pipeline::CreateRayTracingPipelineError::Device(
                            self.handle_hal_error(error),
                        )
                    }
                    hal::PipelineError::Linkage(stage, msg) => {
                        pipeline::CreateRayTracingPipelineError::Internal { stage, error: msg }
                    }
                    hal::PipelineError::EntryPoint(stage) => {
                        pipeline::CreateRayTracingPipelineError::Internal {
                            stage: hal::auxil::map_naga_stage(stage),
                            error: ENTRYPOINT_FAILURE_ERROR.to_string(),
                        }
                    }
                    hal::PipelineError::PipelineConstants(stage, error) => {
                        pipeline::CreateRayTracingPipelineError::PipelineConstants { stage, error }
                    }
                }
            })?
        };

        let shader_modules = {
            let mut shader_modules = Vec::new();
            shader_modules.push(desc.ray_generation.module);
            shader_modules.push(desc.miss.module);
            shader_modules.reserve(desc.intersections.len());
            for intersection in &desc.intersections {
                match intersection {
                    pipeline::RayTracingIntersectionDescriptor::Triangle {
                        closest_hit,
                        any_hit,
                    } => {
                        shader_modules.push(closest_hit.module.clone());
                        if let Some(any) = any_hit {
                            shader_modules.push(any.module.clone());
                        }
                    }
                }
            }
            shader_modules
        };

        // Won't panic because `desc.intersections` is required to be below 2^24 - 1 (see `CreateRayTracingPipelineError::TooManyIntersectionGroups`)
        let shader_binding_data = match pipeline::ShaderBindingData::from_raw_pipeline(
            self.clone(),
            raw.as_ref(),
            desc.intersections.len(),
        ) {
            Ok(sbd) => sbd,
            Err(e) => {
                // We need to destroy the raw ray tracing pipeline first.
                unsafe { self.raw().destroy_ray_tracing_pipeline(raw) };
                return Err(e);
            }
        };

        let naga::valid::ImmediateUsage::Valid {
            slots: immediate_slots_required,
            size: _,
        } = io.immediates
        else {
            unreachable!("Immediates exceeding maxImmediateSize should have been rejected");
        };

        let pipeline = pipeline::RayTracingPipeline {
            state: ResourceState::Valid(pipeline::RayTracingPipelineState {
                raw: ManuallyDrop::new(raw),
                layout: pipeline_layout.clone(),
                _shader_modules: shader_modules,
                shader_binding_data,
            }),
            device: self.clone(),
            late_sized_buffer_groups,
            immediate_slots_required,
            label: desc.label.to_string(),
            tracking_data: TrackingData::new(self.tracker_indices.ray_tracing_pipelines.clone()),
        };

        let pipeline = Arc::new(pipeline);

        if is_auto_layout {
            for bgl in pipeline_layout.bind_group_layouts.iter() {
                let Some(bgl) = bgl else {
                    continue;
                };

                // `bind_group_layouts` might contain duplicate entries, so we need to ignore the result.
                let _ = bgl.exclusive_pipeline.set((&pipeline).into());
            }
        }

        Ok(pipeline)
    }
}
