set(EXAMPLES_BASIC_CUTLASS_GEMM_SOURCES
  
)

if (NOT CUTLASS_NATIVE_CUDA)
  # cuda_add_executable does not take interface include directories into account
  # Let's fetch them and pass them to CUDA.
  get_target_property(CUTLASS_INCLUDES CUTLASS INTERFACE_INCLUDE_DIRECTORIES)
  include_directories("${CUTLASS_INCLUDES}")
endif()

#SET (SCALE_PARAMS 4 8 12 16 20)
#foreach(SCALE IN LISTS SCALE_PARAMS)
  #cutlass_add_executable(
    #gemm_${SCALE}
    #basic_gemm.cu
    #gemm_helper.h
    #timing_helper.h
    #OPTIONS "-DSCALING_PARAM=${SCALE}"
  #)
#endforeach(SCALE IN LISTS SCALE_PARAMS)


SET (RATIO 8 16 24 32)
SET (SCALE_K 4 8 12 16 20 24)
SET (SCALE_N 4 8 12 16 20 24)
SET (SCALE_M 4 8 12 16 20 24)

foreach(R IN LISTS RATIO)
foreach(K IN LISTS SCALE_K)
	foreach(N IN LISTS SCALE_N)
		foreach(M IN LISTS SCALE_M)
  cutlass_add_executable(
	  gemm_r_${R}_k_${K}_n_${N}_m_${M}
    basic_gemm.cu
    gemm_helper.h
    timing_helper.h
    OPTIONS "-DRATIO=${R} -DSCALE_K=${K} -DSCALE_N=${N} -DSCALE_M=${M}"
  )
endforeach()
endforeach()
endforeach()
endforeach()
