; SPIR-V
; Version: 1.5
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 18
; Schema: 0
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %fragColor %vertexInput
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpSourceExtension "GL_EXT_mesh_shader"
               OpName %main "main"
               OpName %fragColor "fragColor"
               OpName %VertexInput "VertexInput"
               OpMemberName %VertexInput 0 "color"
               OpName %vertexInput "vertexInput"
               OpDecorate %fragColor Location 0
               OpDecorate %VertexInput Block
               OpMemberDecorate %VertexInput 0 Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
  %fragColor = OpVariable %_ptr_Output_v4float Output
%VertexInput = OpTypeStruct %v4float
%_ptr_Input_VertexInput = OpTypePointer Input %VertexInput
%vertexInput = OpVariable %_ptr_Input_VertexInput Input
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_ptr_Input_v4float = OpTypePointer Input %v4float
       %main = OpFunction %void None %3
          %5 = OpLabel
         %16 = OpAccessChain %_ptr_Input_v4float %vertexInput %int_0
         %17 = OpLoad %v4float %16
               OpStore %fragColor %17
               OpReturn
               OpFunctionEnd
