; SPIR-V
; Version: 1.5
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 89
; Schema: 0
               OpCapability MeshShadingEXT
               OpExtension "SPV_EXT_mesh_shader"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint MeshEXT %main "main" %sharedData %gl_MeshVerticesEXT %vertexOutput %payloadData %gl_PrimitiveTriangleIndicesEXT %gl_LocalInvocationIndex %primitiveOutput %gl_MeshPrimitivesEXT
               OpExecutionMode %main LocalSize 1 1 1
               OpExecutionMode %main OutputVertices 3
               OpExecutionMode %main OutputPrimitivesEXT 1
               OpExecutionMode %main OutputTrianglesEXT
               OpSource GLSL 450
               OpSourceExtension "GL_EXT_mesh_shader"
               OpName %main "main"
               OpName %sharedData "sharedData"
               OpName %gl_MeshPerVertexEXT "gl_MeshPerVertexEXT"
               OpMemberName %gl_MeshPerVertexEXT 0 "gl_Position"
               OpMemberName %gl_MeshPerVertexEXT 1 "gl_PointSize"
               OpMemberName %gl_MeshPerVertexEXT 2 "gl_ClipDistance"
               OpMemberName %gl_MeshPerVertexEXT 3 "gl_CullDistance"
               OpName %gl_MeshVerticesEXT "gl_MeshVerticesEXT"
               OpName %VertexOutput "VertexOutput"
               OpMemberName %VertexOutput 0 "color"
               OpName %vertexOutput "vertexOutput"
               OpName %PayloadData "PayloadData"
               OpMemberName %PayloadData 0 "colorMask"
               OpMemberName %PayloadData 1 "visible"
               OpName %payloadData "payloadData"
               OpName %gl_PrimitiveTriangleIndicesEXT "gl_PrimitiveTriangleIndicesEXT"
               OpName %gl_LocalInvocationIndex "gl_LocalInvocationIndex"
               OpName %PrimitiveOutput "PrimitiveOutput"
               OpMemberName %PrimitiveOutput 0 "colorMask"
               OpName %primitiveOutput "primitiveOutput"
               OpName %gl_MeshPerPrimitiveEXT "gl_MeshPerPrimitiveEXT"
               OpMemberName %gl_MeshPerPrimitiveEXT 0 "gl_PrimitiveID"
               OpMemberName %gl_MeshPerPrimitiveEXT 1 "gl_Layer"
               OpMemberName %gl_MeshPerPrimitiveEXT 2 "gl_ViewportIndex"
               OpMemberName %gl_MeshPerPrimitiveEXT 3 "gl_CullPrimitiveEXT"
               OpName %gl_MeshPrimitivesEXT "gl_MeshPrimitivesEXT"
               OpDecorate %gl_MeshPerVertexEXT Block
               OpMemberDecorate %gl_MeshPerVertexEXT 0 BuiltIn Position
               OpMemberDecorate %gl_MeshPerVertexEXT 1 BuiltIn PointSize
               OpMemberDecorate %gl_MeshPerVertexEXT 2 BuiltIn ClipDistance
               OpMemberDecorate %gl_MeshPerVertexEXT 3 BuiltIn CullDistance
               OpDecorate %VertexOutput Block
               OpMemberDecorate %VertexOutput 0 Location 0
               OpDecorate %gl_PrimitiveTriangleIndicesEXT BuiltIn PrimitiveTriangleIndicesEXT
               OpDecorate %gl_LocalInvocationIndex BuiltIn LocalInvocationIndex
               OpDecorate %PrimitiveOutput Block
               OpMemberDecorate %PrimitiveOutput 0 PerPrimitiveEXT
               OpDecorate %primitiveOutput Location 1
               OpDecorate %gl_MeshPerPrimitiveEXT Block
               OpMemberDecorate %gl_MeshPerPrimitiveEXT 0 BuiltIn PrimitiveId
               OpMemberDecorate %gl_MeshPerPrimitiveEXT 0 PerPrimitiveEXT
               OpMemberDecorate %gl_MeshPerPrimitiveEXT 1 BuiltIn Layer
               OpMemberDecorate %gl_MeshPerPrimitiveEXT 1 PerPrimitiveEXT
               OpMemberDecorate %gl_MeshPerPrimitiveEXT 2 BuiltIn ViewportIndex
               OpMemberDecorate %gl_MeshPerPrimitiveEXT 2 PerPrimitiveEXT
               OpMemberDecorate %gl_MeshPerPrimitiveEXT 3 BuiltIn CullPrimitiveEXT
               OpMemberDecorate %gl_MeshPerPrimitiveEXT 3 PerPrimitiveEXT
               OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
%_ptr_Workgroup_uint = OpTypePointer Workgroup %uint
 %sharedData = OpVariable %_ptr_Workgroup_uint Workgroup
     %uint_5 = OpConstant %uint 5
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
         %23 = OpConstantComposite %v4float %float_0 %float_1 %float_0 %float_1
%_ptr_Output_v4float = OpTypePointer Output %v4float
      %int_1 = OpConstant %int 1
   %float_n1 = OpConstant %float -1
         %28 = OpConstantComposite %v4float %float_n1 %float_n1 %float_0 %float_1
      %int_2 = OpConstant %int 2
         %31 = OpConstantComposite %v4float %float_1 %float_n1 %float_0 %float_1
%VertexOutput = OpTypeStruct %v4float
%_arr_VertexOutput_uint_3 = OpTypeArray %VertexOutput %uint_3
%_ptr_Output__arr_VertexOutput_uint_3 = OpTypePointer Output %_arr_VertexOutput_uint_3
%vertexOutput = OpVariable %_ptr_Output__arr_VertexOutput_uint_3 Output
       %bool = OpTypeBool
%PayloadData = OpTypeStruct %v4float %bool
%_ptr_TaskPayloadWorkgroupEXT_PayloadData = OpTypePointer TaskPayloadWorkgroupEXT %PayloadData
%payloadData = OpVariable %_ptr_TaskPayloadWorkgroupEXT_PayloadData TaskPayloadWorkgroupEXT
%_ptr_TaskPayloadWorkgroupEXT_v4float = OpTypePointer TaskPayloadWorkgroupEXT %v4float
         %46 = OpConstantComposite %v4float %float_0 %float_0 %float_1 %float_1
         %51 = OpConstantComposite %v4float %float_1 %float_0 %float_0 %float_1
     %v3uint = OpTypeVector %uint 3
%_arr_v3uint_uint_1 = OpTypeArray %v3uint %uint_1
%_ptr_Output__arr_v3uint_uint_1 = OpTypePointer Output %_arr_v3uint_uint_1
%gl_PrimitiveTriangleIndicesEXT = OpVariable %_ptr_Output__arr_v3uint_uint_1 Output
%_ptr_Input_uint = OpTypePointer Input %uint
%gl_LocalInvocationIndex = OpVariable %_ptr_Input_uint Input
     %uint_0 = OpConstant %uint 0
     %uint_2 = OpConstant %uint 2
         %65 = OpConstantComposite %v3uint %uint_0 %uint_1 %uint_2
%_ptr_Output_v3uint = OpTypePointer Output %v3uint
%PrimitiveOutput = OpTypeStruct %v4float
%_arr_PrimitiveOutput_uint_1 = OpTypeArray %PrimitiveOutput %uint_1
%_ptr_Output__arr_PrimitiveOutput_uint_1 = OpTypePointer Output %_arr_PrimitiveOutput_uint_1
%primitiveOutput = OpVariable %_ptr_Output__arr_PrimitiveOutput_uint_1 Output
         %72 = OpConstantComposite %v4float %float_1 %float_0 %float_1 %float_1
%gl_MeshPerPrimitiveEXT = OpTypeStruct %int %int %int %bool
%_arr_gl_MeshPerPrimitiveEXT_uint_1 = OpTypeArray %gl_MeshPerPrimitiveEXT %uint_1
%_ptr_Output__arr_gl_MeshPerPrimitiveEXT_uint_1 = OpTypePointer Output %_arr_gl_MeshPerPrimitiveEXT_uint_1
%gl_MeshPrimitivesEXT = OpVariable %_ptr_Output__arr_gl_MeshPerPrimitiveEXT_uint_1 Output
      %int_3 = OpConstant %int 3
%_ptr_TaskPayloadWorkgroupEXT_bool = OpTypePointer TaskPayloadWorkgroupEXT %bool
%_ptr_Output_bool = OpTypePointer Output %bool
%_arr_v4float_uint_3 = OpTypeArray %v4float %uint_3
         %86 = OpConstantComposite %_arr_v4float_uint_3 %23 %28 %31
         %87 = OpConstantComposite %_arr_v4float_uint_3 %23 %46 %51
%gl_WorkGroupSize = OpConstantComposite %v3uint %uint_1 %uint_1 %uint_1
       %main = OpFunction %void None %3
          %5 = OpLabel
               OpStore %sharedData %uint_5
               OpSetMeshOutputsEXT %uint_3 %uint_1
         %25 = OpAccessChain %_ptr_Output_v4float %gl_MeshVerticesEXT %int_0 %int_0
               OpStore %25 %23
         %29 = OpAccessChain %_ptr_Output_v4float %gl_MeshVerticesEXT %int_1 %int_0
               OpStore %29 %28
         %32 = OpAccessChain %_ptr_Output_v4float %gl_MeshVerticesEXT %int_2 %int_0
               OpStore %32 %31
         %42 = OpAccessChain %_ptr_TaskPayloadWorkgroupEXT_v4float %payloadData %int_0
         %43 = OpLoad %v4float %42
         %44 = OpFMul %v4float %23 %43
         %45 = OpAccessChain %_ptr_Output_v4float %vertexOutput %int_0 %int_0
               OpStore %45 %44
         %47 = OpAccessChain %_ptr_TaskPayloadWorkgroupEXT_v4float %payloadData %int_0
         %48 = OpLoad %v4float %47
         %49 = OpFMul %v4float %46 %48
         %50 = OpAccessChain %_ptr_Output_v4float %vertexOutput %int_1 %int_0
               OpStore %50 %49
         %52 = OpAccessChain %_ptr_TaskPayloadWorkgroupEXT_v4float %payloadData %int_0
         %53 = OpLoad %v4float %52
         %54 = OpFMul %v4float %51 %53
         %55 = OpAccessChain %_ptr_Output_v4float %vertexOutput %int_2 %int_0
               OpStore %55 %54
         %62 = OpLoad %uint %gl_LocalInvocationIndex
         %67 = OpAccessChain %_ptr_Output_v3uint %gl_PrimitiveTriangleIndicesEXT %62
               OpStore %67 %65
         %73 = OpAccessChain %_ptr_Output_v4float %primitiveOutput %int_0 %int_0
               OpStore %73 %72
         %80 = OpAccessChain %_ptr_TaskPayloadWorkgroupEXT_bool %payloadData %int_1
         %81 = OpLoad %bool %80
         %82 = OpLogicalNot %bool %81
         %84 = OpAccessChain %_ptr_Output_bool %gl_MeshPrimitivesEXT %int_0 %int_3
               OpStore %84 %82
               OpReturn
               OpFunctionEnd
