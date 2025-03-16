; SPIR-V
; Version: 1.5
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 30
; Schema: 0
               OpCapability MeshShadingEXT
               OpExtension "SPV_EXT_mesh_shader"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint TaskEXT %main "main" %taskPayload
               OpExecutionMode %main LocalSize 4 1 1
               OpSource GLSL 450
               OpSourceExtension "GL_EXT_mesh_shader"
               OpName %main "main"
               OpName %TaskPayload "TaskPayload"
               OpMemberName %TaskPayload 0 "colorMask"
               OpMemberName %TaskPayload 1 "visible"
               OpName %taskPayload "taskPayload"
               OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
       %bool = OpTypeBool
%TaskPayload = OpTypeStruct %v4float %bool
%_ptr_TaskPayloadWorkgroupEXT_TaskPayload = OpTypePointer TaskPayloadWorkgroupEXT %TaskPayload
%taskPayload = OpVariable %_ptr_TaskPayloadWorkgroupEXT_TaskPayload TaskPayloadWorkgroupEXT
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
    %float_1 = OpConstant %float 1
    %float_0 = OpConstant %float 0
         %16 = OpConstantComposite %v4float %float_1 %float_1 %float_0 %float_1
%_ptr_TaskPayloadWorkgroupEXT_v4float = OpTypePointer TaskPayloadWorkgroupEXT %v4float
      %int_1 = OpConstant %int 1
       %true = OpConstantTrue %bool
%_ptr_TaskPayloadWorkgroupEXT_bool = OpTypePointer TaskPayloadWorkgroupEXT %bool
       %uint = OpTypeInt 32 0
     %uint_3 = OpConstant %uint 3
     %uint_1 = OpConstant %uint 1
     %v3uint = OpTypeVector %uint 3
     %uint_4 = OpConstant %uint 4
%gl_WorkGroupSize = OpConstantComposite %v3uint %uint_4 %uint_1 %uint_1
       %main = OpFunction %void None %3
          %5 = OpLabel
         %18 = OpAccessChain %_ptr_TaskPayloadWorkgroupEXT_v4float %taskPayload %int_0
               OpStore %18 %16
         %22 = OpAccessChain %_ptr_TaskPayloadWorkgroupEXT_bool %taskPayload %int_1
               OpStore %22 %true
               OpEmitMeshTasksEXT %uint_3 %uint_1 %uint_1 %taskPayload
               OpFunctionEnd
