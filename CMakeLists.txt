cmake_minimum_required(VERSION 3.20)

project(glsl_compute_engine
    VERSION 1.0.0
    DESCRIPTION "GLSL 4.60 + Modern OpenGL 4.6 DSA GPU Compute & Rendering Engine"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_OPENGL_RUNTIME "Build OpenGL 4.6 DSA runtime and desktop demo" OFF)
option(ENABLE_TESTS "Build C++20 headless CPU parity and core unit tests" ON)

function(glengine_apply_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /WX)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
endfunction()

# =============================================================================
# Headless Core Library (Math, GPU PODs, Preprocessor, Reference Algos, Profiler)
# =============================================================================
add_library(glengine_core STATIC
    src/shader_preprocessor.cpp
    src/reference_algorithms.cpp
    src/gpu_profiler.cpp
)

target_include_directories(glengine_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
glengine_apply_warnings(glengine_core)

# =============================================================================
# Optional OpenGL 4.6 DSA Runtime & Desktop Demo
# =============================================================================
if(ENABLE_OPENGL_RUNTIME)
    find_package(OpenGL REQUIRED)

    add_library(glengine_gl_runtime STATIC
        src/gl/gl_compute_runtime.cpp
    )
    target_link_libraries(glengine_gl_runtime
        PUBLIC
            glengine_core
            OpenGL::GL
    )
    glengine_apply_warnings(glengine_gl_runtime)

    add_executable(glengine_demo
        src/gl/demo_main.cpp
    )
    target_link_libraries(glengine_demo
        PRIVATE
            glengine_gl_runtime
    )
    target_compile_definitions(glengine_demo
        PRIVATE
            GLENGINE_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    )
    glengine_apply_warnings(glengine_demo)
endif()

# =============================================================================
# Headless C++20 Unit Tests (CTest)
# =============================================================================
if(ENABLE_TESTS)
    enable_testing()

    add_executable(test_engine
        tests/test_engine.cpp
    )
    target_link_libraries(test_engine
        PRIVATE
            glengine_core
    )
    target_compile_definitions(test_engine
        PRIVATE
            GLENGINE_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    )
    glengine_apply_warnings(test_engine)

    add_test(NAME glengine_unit_tests COMMAND test_engine)
endif()

# =============================================================================
# Optional Khronos glslangValidator Shader Validation Target
# =============================================================================
find_program(GLSLANG_VALIDATOR glslangValidator)
if(GLSLANG_VALIDATOR)
    set(GLENGINE_SHADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute/curl_particles.comp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute/frustum_cull.comp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute/bitonic_sort.comp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute/tiled_light_cull.comp"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/graphics/pbr_gbuffer.vert"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/graphics/pbr_gbuffer.frag"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/graphics/pbr_lighting.vert"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/graphics/pbr_lighting.frag"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/graphics/raymarch_sdf.frag"
    )

    set(SPIRV_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/validated_spirv")
    set(VALIDATE_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPIRV_OUT_DIR}"
    )

    foreach(SHADER_FILE ${GLENGINE_SHADERS})
        get_filename_component(SHADER_NAME "${SHADER_FILE}" NAME)
        list(APPEND VALIDATE_COMMANDS
            COMMAND "${GLSLANG_VALIDATOR}" "${SHADER_FILE}"
            COMMAND "${GLSLANG_VALIDATOR}" -G "${SHADER_FILE}" -o "${SPIRV_OUT_DIR}/${SHADER_NAME}.spv"
        )
    endforeach()

    add_custom_target(validate_shaders
        ${VALIDATE_COMMANDS}
        COMMENT "Validating all GLSL 4.60 shaders (OpenGL GLSL + OpenGL SPIR-V)"
        VERBATIM
    )
endif()
