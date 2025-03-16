; SPIR-V
; Version: 1.5
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 55
; Schema: 0
               OpCapability MeshShadingEXT
               OpExtension "SPV_EXT_mesh_shader"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint MeshEXT %main "main" %gl_MeshVerticesEXT %vertexOutput %gl_PrimitiveTriangleIndicesEXT %gl_LocalInvocationIndex
               OpExecutionMode %main LocalSize 1 1 1
               OpExecutionMode %main OutputVertices 3
               OpExecutionMode %main OutputPrimitivesEXT 1
               OpExecutionMode %main OutputTrianglesEXT
               OpSource GLSL 450
               OpSourceExtension "GL_EXT_mesh_shader"
               OpName %main "main"
               OpName %gl_MeshPerVertexEXT "gl_MeshPerVertexEXT"
               OpMemberName %gl_MeshPerVertexEXT 0 "gl_Position"
               OpMemberName %gl_MeshPerVertexEXT 1 "gl_PointSize"
               OpMemberName %gl_MeshPerVertexEXT 2 "gl_ClipDistance"
               OpMemberName %gl_MeshPerVertexEXT 3 "gl_CullDistance"
               OpName %gl_MeshVerticesEXT "gl_MeshVerticesEXT"
               OpName %VertexOutput "VertexOutput"
               OpMemberName %VertexOutput 0 "color"
               OpName %vertexOutput "vertexOutput"
               OpName %gl_PrimitiveTriangleIndicesEXT "gl_PrimitiveTriangleIndicesEXT"
               OpName %gl_LocalInvocationIndex "gl_LocalInvocationIndex"
               OpDecorate %gl_MeshPerVertexEXT Block
               OpMemberDecorate %gl_MeshPerVertexEXT 0 BuiltIn Position
               OpMemberDecorate %gl_MeshPerVertexEXT 1 BuiltIn PointSize
               OpMemberDecorate %gl_MeshPerVertexEXT 2 BuiltIn ClipDistance
               OpMemberDecorate %gl_MeshPerVertexEXT 3 BuiltIn CullDistance
               OpDecorate %VertexOutput Block
               OpMemberDecorate %VertexOutput 0 Location 0
               OpDecorate %gl_PrimitiveTriangleIndicesEXT BuiltIn PrimitiveTriangleIndicesEXT
               OpDecorate %gl_LocalInvocationIndex BuiltIn LocalInvocationIndex
               OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
     %uint_3 = OpConstant %uint 3
     %uint_1 = OpConstant %uint 1
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_arr_float_uint_1 = OpTypeArray %float %uint_1
%gl_MeshPerVertexEXT = OpTypeStruct %v4float %float %_arr_float_uint_1 %_arr_float_uint_1
%_arr_gl_MeshPerVertexEXT_uint_3 = OpTypeArray %gl_MeshPerVertexEXT %uint_3
%_ptr_Output__arr_gl_MeshPerVertexEXT_uint_3 = OpTypePointer Output %_arr_gl_MeshPerVertexEXT_uint_3
%gl_MeshVerticesEXT = OpVariable %_ptr_Output__arr_gl_MeshPerVertexEXT_uint_3 Output
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
         %20 = OpConstantComposite %v4float %float_0 %float_1 %float_0 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
      %int_1 = OpConstant %int 1
   %float_n1 = OpConstant %float -1
         %25 = OpConstantComposite %v4float %float_n1 %float_n1 %float_0 %float_1
      %int_2 = OpConstant %int 2
         %28 = OpConstantComposite %v4float %float_1 %float_n1 %float_0 %float_1
%VertexOutput = OpTypeStruct %v4float
%_arr_VertexOutput_uint_3 = OpTypeArray %VertexOutput %uint_3
%_ptr_Output__arr_VertexOutput_uint_3 = OpTypePointer Output %_arr_VertexOutput_uint_3
%vertexOutput = OpVariable %_ptr_Output__arr_VertexOutput_uint_3 Output
         %35 = OpConstantComposite %v4float %float_0 %float_0 %float_1 %float_1
         %37 = OpConstantComposite %v4float %float_1 %float_0 %float_0 %float_1
     %v3uint = OpTypeVector %uint 3
%_arr_v3uint_uint_1 = OpTypeArray %v3uint %uint_1
%_ptr_Output__arr_v3uint_uint_1 = OpTypePointer Output %_arr_v3uint_uint_1
%gl_PrimitiveTriangleIndicesEXT = OpVariable %_ptr_Output__arr_v3uint_uint_1 Output
%_ptr_Input_uint = OpTypePointer Input %uint
%gl_LocalInvocationIndex = OpVariable %_ptr_Input_uint Input
     %uint_0 = OpConstant %uint 0
     %uint_2 = OpConstant %uint 2
         %48 = OpConstantComposite %v3uint %uint_0 %uint_1 %uint_2
%_ptr_Output_v3uint = OpTypePointer Output %v3uint
%_arr_v4float_uint_3 = OpTypeArray %v4float %uint_3
         %52 = OpConstantComposite %_arr_v4float_uint_3 %20 %25 %28
         %53 = OpConstantComposite %_arr_v4float_uint_3 %20 %35 %37
%gl_WorkGroupSize = OpConstantComposite %v3uint %uint_1 %uint_1 %uint_1
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpSetMeshOutputsEXT %uint_3 %uint_1
         %22 = OpAccessChain %_ptr_Output_v4float %gl_MeshVerticesEXT %int_0 %int_0
               OpStore %22 %20
         %26 = OpAccessChain %_ptr_Output_v4float %gl_MeshVerticesEXT %int_1 %int_0
               OpStore %26 %25
         %29 = OpAccessChain %_ptr_Output_v4float %gl_MeshVerticesEXT %int_2 %int_0
               OpStore %29 %28
         %34 = OpAccessChain %_ptr_Output_v4float %vertexOutput %int_0 %int_0
               OpStore %34 %20
         %36 = OpAccessChain %_ptr_Output_v4float %vertexOutput %int_1 %int_0
               OpStore %36 %35
         %38 = OpAccessChain %_ptr_Output_v4float %vertexOutput %int_2 %int_0
               OpStore %38 %37
         %45 = OpLoad %uint %gl_LocalInvocationIndex
         %50 = OpAccessChain %_ptr_Output_v3uint %gl_PrimitiveTriangleIndicesEXT %45
               OpStore %50 %48
               OpReturn
               OpFunctionEnd
