cmake_minimum_required(VERSION 3.18)
project(llm-server VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Build options ────────────────────────────────────────────────────────────

option(LLM_SERVER_CUBLAS  "Enable CUDA/cuBLAS GPU offload"     OFF)
option(LLM_SERVER_METAL   "Enable Metal GPU offload (macOS)"   OFF)
option(LLM_SERVER_VULKAN  "Enable Vulkan GPU offload"          OFF)
option(LLM_SERVER_STATIC  "Build fully static binary"          OFF)

# Pass GPU flags through to llama.cpp
if(LLM_SERVER_CUBLAS)
  set(LLAMA_CUBLAS      ON CACHE BOOL "" FORCE)
  set(LLAMA_CUDA        ON CACHE BOOL "" FORCE)
endif()
if(LLM_SERVER_METAL)
  set(LLAMA_METAL       ON CACHE BOOL "" FORCE)
endif()
if(LLM_SERVER_VULKAN)
  set(LLAMA_VULKAN      ON CACHE BOOL "" FORCE)
endif()

# ─── Dependencies via FetchContent ───────────────────────────────────────────

include(FetchContent)

# llama.cpp  –  GGUF model loading, KV cache, batched decode
FetchContent_Declare(llama
  GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
  GIT_TAG        b3447             # pin to a recent stable tag
  GIT_SHALLOW    TRUE
)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(llama)

# cpp-httplib  –  header-only HTTP/1.1 server
FetchContent_Declare(httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG        v0.15.3
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(httplib)

# nlohmann/json  –  header-only JSON
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(json)

# ─── Executable ──────────────────────────────────────────────────────────────

add_executable(llm-server
  src/server.cpp
)

target_include_directories(llm-server PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  # llama.cpp public headers
  ${llama_SOURCE_DIR}/include
  ${llama_SOURCE_DIR}/common
  # httplib (header-only, FetchContent puts it here)
  ${httplib_SOURCE_DIR}
  # nlohmann/json (header-only)
  ${json_SOURCE_DIR}/include
)

target_link_libraries(llm-server PRIVATE
  llama
  common            # llama.cpp sampling / utility helpers
  ${CMAKE_THREAD_LIBS_INIT}
)

find_package(Threads REQUIRED)
target_link_libraries(llm-server PRIVATE Threads::Threads)

# ─── Compile options ─────────────────────────────────────────────────────────

if(MSVC)
  target_compile_options(llm-server PRIVATE /O2 /W3)
else()
  target_compile_options(llm-server PRIVATE
    -O3 -Wall -Wextra -Wno-unused-parameter
    -march=native            # leverage host CPU SIMD (AVX2/AVX-512)
  )
endif()

if(LLM_SERVER_STATIC AND UNIX AND NOT APPLE)
  target_link_options(llm-server PRIVATE -static-libgcc -static-libstdc++)
endif()

# ─── Install ─────────────────────────────────────────────────────────────────

install(TARGETS llm-server DESTINATION bin)

# ─── Print config summary ────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "llm-server build configuration")
message(STATUS "  CUDA/cuBLAS : ${LLM_SERVER_CUBLAS}")
message(STATUS "  Metal       : ${LLM_SERVER_METAL}")
message(STATUS "  Vulkan      : ${LLM_SERVER_VULKAN}")
message(STATUS "  Static      : ${LLM_SERVER_STATIC}")
message(STATUS "")
