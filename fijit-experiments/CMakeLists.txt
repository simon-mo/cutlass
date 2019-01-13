set(EXAMPLES_BASIC_CUTLASS_GEMM_SOURCES
  
)

if (NOT CUTLASS_NATIVE_CUDA)
  # cuda_add_executable does not take interface include directories into account
  # Let's fetch them and pass them to CUDA.
  get_target_property(CUTLASS_INCLUDES CUTLASS INTERFACE_INCLUDE_DIRECTORIES)
  include_directories("${CUTLASS_INCLUDES}")
endif()

SET (SCALE_PARAMS 1 2 3 4 5 6 7 8 9 10)
foreach(SCALE IN LISTS SCALE_PARAMS)
  cutlass_add_executable(
    gemm_${SCALE}
    basic_gemm.cu
    gemm_helper.h
    timing_helper.h
    OPTION "-DSCALING_PARAM=${SCALE}"
  )
endforeach(SCALE IN LISTS SCALE_PARAMS)


