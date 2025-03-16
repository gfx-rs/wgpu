; SPIR-V
; Version: 1.5
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 24
; Schema: 0
               OpCapability Shader
               OpCapability MeshShadingEXT
               OpExtension "SPV_EXT_mesh_shader"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %fragColor %vertexInput %primitiveInput
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpSourceExtension "GL_EXT_mesh_shader"
               OpName %main "main"
               OpName %fragColor "fragColor"
               OpName %VertexInput "VertexInput"
               OpMemberName %VertexInput 0 "color"
               OpName %vertexInput "vertexInput"
               OpName %PrimitiveInput "PrimitiveInput"
               OpMemberName %PrimitiveInput 0 "colorMask"
               OpName %primitiveInput "primitiveInput"
               OpDecorate %fragColor Location 0
               OpDecorate %VertexInput Block
               OpMemberDecorate %VertexInput 0 Location 0
               OpDecorate %PrimitiveInput Block
               OpMemberDecorate %PrimitiveInput 0 PerPrimitiveEXT
               OpDecorate %primitiveInput Location 1
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
%PrimitiveInput = OpTypeStruct %v4float
%_ptr_Input_PrimitiveInput = OpTypePointer Input %PrimitiveInput
%primitiveInput = OpVariable %_ptr_Input_PrimitiveInput Input
       %main = OpFunction %void None %3
          %5 = OpLabel
         %16 = OpAccessChain %_ptr_Input_v4float %vertexInput %int_0
         %17 = OpLoad %v4float %16
         %21 = OpAccessChain %_ptr_Input_v4float %primitiveInput %int_0
         %22 = OpLoad %v4float %21
         %23 = OpFMul %v4float %17 %22
               OpStore %fragColor %23
               OpReturn
               OpFunctionEnd
