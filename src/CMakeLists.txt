set(GGML_HEADERS_BITNET ../include/ggml-bitnet.h)
set(GGML_SOURCES_BITNET ggml-bitnet-mad.cpp ggml-bitnet-lut.cpp)

include_directories(3rdparty/llama.cpp/ggml/include)
include_directories(../include)

if (NOT (CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "GNU") OR
    NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(FATAL_ERROR "Clang or GCC is required for Bitnet.cpp compilation")
endif()

# Force use of portable fallback code path on non-x86/ARM hosts
add_definitions(-DGGML_DISABLE_AVX)

if (BITNET_RISCV_TL3)
    add_definitions(-DGGML_BITNET_RISCV_TL3)
endif()