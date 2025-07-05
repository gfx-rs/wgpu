; SPIR-V
; Version: 1.5
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 11
; Schema: 0
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %vertexInput
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpSourceExtension "GL_EXT_mesh_shader"
               OpName %main "main"
               OpName %VertexInput "VertexInput"
               OpMemberName %VertexInput 0 "color"
               OpName %vertexInput "vertexInput"
               OpDecorate %VertexInput Block
               OpMemberDecorate %VertexInput 0 Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%VertexInput = OpTypeStruct %v4float
%_ptr_Input_VertexInput = OpTypePointer Input %VertexInput
%vertexInput = OpVariable %_ptr_Input_VertexInput Input
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpReturn
               OpFunctionEnd
