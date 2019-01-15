if (NOT CUTLASS_NATIVE_CUDA)
  get_target_property(CUTLASS_INCLUDES CUTLASS INTERFACE_INCLUDE_DIRECTORIES)
  include_directories("${CUTLASS_INCLUDES}")
endif()

SET (SCALE_PARAMS 4 8 12 16 20)
foreach(SCALE IN LISTS SCALE_PARAMS)
  cutlass_add_executable(
    batched_gemm_${SCALE}
    batched_gemm.cu
    OPTIONS "-DSCALING_PARAM=${SCALE}"
  )
endforeach(SCALE IN LISTS SCALE_PARAMS)
