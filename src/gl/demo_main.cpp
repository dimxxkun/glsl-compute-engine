#include "glengine/gl_compute_runtime.hpp"
#include "glengine/gpu_profiler.hpp"
#include "glengine/math.hpp"
#include "glengine/reference_algorithms.hpp"
#include "glengine/shader_preprocessor.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::filesystem::path locateShaderRoot(int argc, char** argv) {
    if (argc > 1 && argv[1] != nullptr) {
        return std::filesystem::path(argv[1]);
    }
#ifdef GLENGINE_SHADER_DIR
    return std::filesystem::path(GLENGINE_SHADER_DIR);
#else
    return std::filesystem::path("shaders");
#endif
}

} // namespace

int main(int argc, char** argv) {
    using namespace glengine;

    const std::filesystem::path shaderRoot = locateShaderRoot(argc, argv);
    std::cout << "=== GLSL 4.60 + OpenGL 4.6 DSA Compute Engine Demo ===\n";
    std::cout << "Shader root: " << shaderRoot.string() << "\n\n";

    std::cout << "[1/3] GPU std140/std430 POD Layout Summary:\n";
    std::cout << "  GpuParticle                 : " << sizeof(GpuParticle) << " bytes (align " << alignof(GpuParticle) << ")\n";
    std::cout << "  GpuPointLight               : " << sizeof(GpuPointLight) << " bytes (align " << alignof(GpuPointLight) << ")\n";
    std::cout << "  ObjectBounds                : " << sizeof(ObjectBounds) << " bytes (align " << alignof(ObjectBounds) << ")\n";
    std::cout << "  DrawElementsIndirectCommand : " << sizeof(DrawElementsIndirectCommand) << " bytes (align " << alignof(DrawElementsIndirectCommand) << ")\n";
    std::cout << "  GpuSortEntry                : " << sizeof(GpuSortEntry) << " bytes (align " << alignof(GpuSortEntry) << ")\n";
    std::cout << "  TileCullUbo                 : " << sizeof(TileCullUbo) << " bytes (align " << alignof(TileCullUbo) << ")\n";
    std::cout << "  InstanceDataGpu             : " << sizeof(InstanceDataGpu) << " bytes (align " << alignof(InstanceDataGpu) << ")\n\n";

    std::cout << "[2/3] Preprocessing GLSL 4.60 Compute & Graphics Shaders:\n";
    const std::array<std::string, 9> shaderList = {{
        "compute/curl_particles.comp",
        "compute/frustum_cull.comp",
        "compute/bitonic_sort.comp",
        "compute/tiled_light_cull.comp",
        "graphics/pbr_gbuffer.vert",
        "graphics/pbr_gbuffer.frag",
        "graphics/pbr_lighting.vert",
        "graphics/pbr_lighting.frag",
        "graphics/raymarch_sdf.frag"
    }};

    ShaderPreprocessor preprocessor;
    PreprocessOptions options{};
    options.includeDirs.push_back(shaderRoot);
    options.emitLineDirectives = false;

    for (const auto& relPath : shaderList) {
        const std::filesystem::path fullPath = shaderRoot / relPath;
        try {
            const PreprocessResult res = preprocessor.preprocessFile(fullPath, options);
            std::cout << "  [OK] " << relPath << " (" << shaderStageName(res.stage)
                      << ", " << res.source.size() << " bytes)\n";
        } catch (const std::exception& ex) {
            std::cerr << "  [ERR] " << relPath << ": " << ex.what() << '\n';
            return 1;
        }
    }

    std::cout << "\n[3/3] CPU Reference Parity Check (Frustum Culling + Bitonic Sort):\n";
    const Mat4 view = Mat4::lookAt({0.0f, 2.0f, 10.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    const Mat4 proj = Mat4::perspective(60.0f * kPi / 180.0f, 16.0f / 9.0f, 0.1f, 100.0f);
    const Frustum frustum = Frustum::fromViewProjection(proj * view);

    std::vector<ObjectBounds> objects(4);
    objects[0] = {Vec4{-1.0f, -1.0f, -1.0f, 1.73f}, Vec4{1.0f, 1.0f, 1.0f, 0.0f}, UVec4{0, 100, 0, 1}};
    objects[1] = {Vec4{-2.0f, -1.0f, 18.0f, 1.73f}, Vec4{2.0f, 1.0f, 22.0f, 0.0f}, UVec4{0, 101, 0, 1}}; // behind camera
    objects[2] = {Vec4{0.5f, 0.0f, -4.0f, 1.0f}, Vec4{1.5f, 1.0f, -3.0f, 0.0f}, UVec4{0, 102, 0, 1}};
    objects[3] = {Vec4{0.0f, 0.0f, 0.0f, 1.0f}, Vec4{1.0f, 1.0f, 1.0f, 0.0f}, UVec4{0, 103, 0, 0}}; // disabled

    std::vector<DrawElementsIndirectCommand> commands(1);
    commands[0] = {36u, 0u, 0u, 0, 0u};
    std::vector<std::uint32_t> visibleIds(objects.size(), 0u);

    const std::uint32_t visible = frustumCullAndCompact(frustum, objects, commands, visibleIds);
    std::cout << "  Visible instances compacted: " << visible << " / " << objects.size() << '\n';

    std::vector<GpuSortEntry> depthEntries = {
        {12.5f, 10u}, {3.2f, 20u}, {7.8f, 30u}, {1.1f, 40u}
    };
    bitonicSortCpu(depthEntries, true);
    std::cout << "  Sorted depth keys: ";
    for (const auto& entry : depthEntries) {
        std::cout << entry.depthKey << "(id=" << entry.payloadIndex << ") ";
    }
    std::cout << "\n\nAttach GlComputeRuntime(procLoader) to an active OpenGL 4.6 Core context\n"
              << "to execute curl_particles, frustum_cull, bitonic_sort, tiled_light_cull,\n"
              << "and glMultiDrawElementsIndirect on the GPU.\n";
    return 0;
}
