; SPIR-V
; Version: 1.5
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 12
; Schema: 0
               OpCapability MeshShadingEXT
               OpExtension "SPV_EXT_mesh_shader"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint TaskEXT %main "main"
               OpExecutionMode %main LocalSize 4 1 1
               OpSource GLSL 450
               OpSourceExtension "GL_EXT_mesh_shader"
               OpName %main "main"
               OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_1 = OpConstant %uint 1
     %v3uint = OpTypeVector %uint 3
     %uint_4 = OpConstant %uint 4
%gl_WorkGroupSize = OpConstantComposite %v3uint %uint_4 %uint_1 %uint_1
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpEmitMeshTasksEXT %uint_1 %uint_1 %uint_1
               OpFunctionEnd
