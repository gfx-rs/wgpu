; ModuleID = 'repro.air'
source_filename = "test.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx14.0.0"

%"struct.metal::_atomic" = type { i32 }

; Function Attrs: convergent mustprogress nounwind
define void @main_(<3 x i32> noundef %0, ptr addrspace(2) noundef readonly align 4 captures(none) dereferenceable(4) "air-buffer-no-alias" %1, ptr addrspace(1) noundef align 4 captures(none) dereferenceable(4) "air-buffer-no-alias" %2, ptr addrspace(3) noundef captures(none) "air-buffer-no-alias" %3) local_unnamed_addr #0 {
  %5 = icmp eq <3 x i32> %0, zeroinitializer
  %6 = tail call i1 @air.all.v3i1(<3 x i1> %5) #4
  br i1 %6, label %7, label %8

7:                                                ; preds = %4
  store i32 0, ptr addrspace(3) %3, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  br label %8

8:                                                ; preds = %7, %4
  tail call void @air.wg.barrier(i32 2, i32 1) #5
  %9 = extractelement <3 x i32> %0, i64 0
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %11, label %12

11:                                               ; preds = %8
  store i32 -1, ptr addrspace(3) %3, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  br label %12

12:                                               ; preds = %11, %8
  tail call void @air.wg.barrier(i32 2, i32 1) #5
  br i1 %10, label %13, label %15

13:                                               ; preds = %12
  %14 = load i32, ptr addrspace(2) %1, align 4, !tbaa !24, !alias.scope !34, !noalias !35
  store i32 %14, ptr addrspace(3) %3, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  br label %15

15:                                               ; preds = %13, %12
  tail call void @air.wg.barrier(i32 2, i32 1) #5
  %16 = load i32, ptr addrspace(3) %3, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  tail call void @air.wg.barrier(i32 2, i32 1) #5
  %17 = icmp eq i32 %16, 0
  br i1 %17, label %18, label %21

18:                                               ; preds = %15
  %19 = getelementptr inbounds %"struct.metal::_atomic", ptr addrspace(1) %2, i64 0, i32 0
  %20 = tail call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) captures(none) %19, i32 1, i32 0, i32 2, i1 true) #6
  br label %21

21:                                               ; preds = %18, %15
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i1 @air.all.v3i1(<3 x i1>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare void @air.wg.barrier(i32, i32) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind willreturn
declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1) captures(none), i32, i32, i32, i1) local_unnamed_addr #3

attributes #0 = { convergent mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent mustprogress nounwind willreturn }
attributes #3 = { mustprogress nounwind willreturn }
attributes #4 = { nounwind willreturn memory(none) }
attributes #5 = { convergent nounwind willreturn }
attributes #6 = { nounwind willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!17, !18, !19}
!llvm.ident = !{!20}
!air.version = !{!21}
!air.language_version = !{!22}
!air.source_file_name = !{!23}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 14, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @main_, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !16}
!12 = !{i32 0, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"local_id"}
!13 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"flag"}
!14 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !15, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"metal::_atomic", !"air.arg_name", !"output"}
!15 = !{i32 0, i32 4, i32 0, !"uint", !"__s"}
!16 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"shared_flag_ptr"}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_enable"}
!19 = !{!"air.compile.framebuffer_fetch_enable"}
!20 = !{!"Apple metal version 32023.155 (metalfe-32023.155)"}
!21 = !{i32 2, i32 6, i32 0}
!22 = !{!"Metal", i32 3, i32 1, i32 0}
!23 = !{!"/Users/ali/work/wgpu/test.metal"}
!24 = !{!25, !25, i64 0}
!25 = !{!"int", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C++ TBAA"}
!28 = !{!29}
!29 = distinct !{!29, !30, !"air-alias-scope-arg(3)"}
!30 = distinct !{!30, !"air-alias-scopes(main_)"}
!31 = !{!32, !33}
!32 = distinct !{!32, !30, !"air-alias-scope-arg(1)"}
!33 = distinct !{!33, !30, !"air-alias-scope-arg(2)"}
!34 = !{!32}
!35 = !{!33, !29}
