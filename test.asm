; SPIR-V
; Version: 1.0
; Generator: Khronos; 28
; Bound: 32
; Schema: 0
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
               OpMemberName %S 0 "a"
               OpMemberName %S 1 "b"
               OpName %S "S"
               OpMemberName %std140_S 0 "a"
               OpMemberName %std140_S 1 "b_col0"
               OpMemberName %std140_S 2 "b_col1"
               OpName %std140_S "std140_S"
               OpName %u "u"
               OpName %S_from_std140 "S_from_std140"
               OpName %main "main"
               OpMemberDecorate %S 0 Offset 0
               OpMemberDecorate %S 1 Offset 8
               OpMemberDecorate %S 1 ColMajor
               OpMemberDecorate %S 1 MatrixStride 8
               OpMemberDecorate %std140_S 0 Offset 0
               OpMemberDecorate %std140_S 1 Offset 8
               OpMemberDecorate %std140_S 2 Offset 16
               OpDecorate %u DescriptorSet 0
               OpDecorate %u Binding 0
               OpDecorate %_struct_10 Block
               OpMemberDecorate %_struct_10 0 Offset 0
       %void = OpTypeVoid
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
    %v2float = OpTypeVector %float 2
%mat2v2float = OpTypeMatrix %v2float 2
          %S = OpTypeStruct %int %mat2v2float
   %std140_S = OpTypeStruct %int %v2float %v2float
 %_struct_10 = OpTypeStruct %std140_S
%_ptr_Uniform__struct_10 = OpTypePointer Uniform %_struct_10
          %u = OpVariable %_ptr_Uniform__struct_10 Uniform
         %13 = OpTypeFunction %S %std140_S
         %23 = OpTypeFunction %void
%_ptr_Uniform_std140_S = OpTypePointer Uniform %std140_S
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
%_ptr_Uniform_S = OpTypePointer Uniform %S
%S_from_std140 = OpFunction %S None %13
         %14 = OpFunctionParameter %std140_S
         %15 = OpLabel
         %16 = OpCompositeExtract %int %14 0
         %18 = OpCompositeExtract %v2float %14 1
         %19 = OpCompositeExtract %v2float %14 2
         %17 = OpCompositeConstruct %mat2v2float %18 %19
         %20 = OpCompositeConstruct %S %16 %17
               OpReturnValue %20
               OpFunctionEnd
       %main = OpFunction %void None %23
         %21 = OpLabel
         %27 = OpAccessChain %_ptr_Uniform_std140_S %u %uint_0
               OpBranch %29
         %29 = OpLabel
         %30 = OpLoad %std140_S %27
         %31 = OpFunctionCall %S %S_from_std140 %30
               OpReturn
               OpFunctionEnd
