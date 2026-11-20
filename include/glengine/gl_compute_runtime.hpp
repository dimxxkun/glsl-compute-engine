#pragma once

#include "glengine/gl46_api.hpp"
#include "glengine/gpu_profiler.hpp"
#include "glengine/math.hpp"
#include "glengine/shader_preprocessor.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace glengine {

// RAII immutable OpenGL 4.6 DSA buffer (glCreateBuffers + glNamedBufferStorage).
class GlBuffer {
public:
    GlBuffer() = default;
    GlBuffer(
        const gl46::DispatchTable* gl,
        std::size_t sizeBytes,
        const void* initialData,
        gl46::GLbitfield storageFlags
    );
    ~GlBuffer();

    GlBuffer(const GlBuffer&) = delete;
    GlBuffer& operator=(const GlBuffer&) = delete;

    GlBuffer(GlBuffer&& other) noexcept;
    GlBuffer& operator=(GlBuffer&& other) noexcept;

    void uploadSubData(std::size_t offsetBytes, std::size_t sizeBytes, const void* data) const;
    void downloadSubData(std::size_t offsetBytes, std::size_t sizeBytes, void* outData) const;
    void bindBase(gl46::GLenum target, gl46::GLuint bindingIndex) const;

    gl46::GLuint handle() const noexcept { return handle_; }
    std::size_t sizeBytes() const noexcept { return sizeBytes_; }
    bool valid() const noexcept { return handle_ != 0; }

private:
    void release() noexcept;

    const gl46::DispatchTable* gl_ = nullptr;
    gl46::GLuint handle_ = 0;
    std::size_t sizeBytes_ = 0;
};

// RAII OpenGL 4.6 compute shader program compiled from preprocessed GLSL 4.60 source.
class GlComputeProgram {
public:
    GlComputeProgram() = default;
    GlComputeProgram(const gl46::DispatchTable* gl, const std::string& glslSource);
    ~GlComputeProgram();

    GlComputeProgram(const GlComputeProgram&) = delete;
    GlComputeProgram& operator=(const GlComputeProgram&) = delete;

    GlComputeProgram(GlComputeProgram&& other) noexcept;
    GlComputeProgram& operator=(GlComputeProgram&& other) noexcept;

    void bind() const;

    gl46::GLuint handle() const noexcept { return program_; }
    bool valid() const noexcept { return program_ != 0; }

private:
    void release() noexcept;

    const gl46::DispatchTable* gl_ = nullptr;
    gl46::GLuint program_ = 0;
};

// Modern OpenGL 4.6 DSA GPU-driven compute & indirect-draw runtime orchestrator.
class GlComputeRuntime {
public:
    explicit GlComputeRuntime(gl46::GlProcLoader procLoader);
    explicit GlComputeRuntime(gl46::DispatchTable dispatchTable);

    const gl46::DispatchTable& gl() const noexcept { return gl_; }
    GlTimerBackend makeTimerBackend() const;

    GlBuffer createImmutableBuffer(
        std::size_t sizeBytes,
        const void* initialData = nullptr,
        gl46::GLbitfield storageFlags = gl46::kGlDynamicStorageBit
    ) const;

    GlComputeProgram compileComputeShaderFile(
        const std::filesystem::path& shaderPath,
        const PreprocessOptions& options = {}
    ) const;

    void dispatchCurlParticles(
        const GlComputeProgram& program,
        const GlBuffer& simParamsUbo,
        const GlBuffer& particleSsbo,
        std::uint32_t particleCount
    ) const;

    void dispatchFrustumCull(
        const GlComputeProgram& program,
        const GlBuffer& frustumUbo,
        const GlBuffer& boundsSsbo,
        const GlBuffer& drawCommandSsbo,
        const GlBuffer& visibleInstanceSsbo,
        std::uint32_t objectCount
    ) const;

    // Executes the complete O(log^2 N) GPU bitonic merge sort sequence with SSBO memory barriers.
    void dispatchBitonicSort(
        const GlComputeProgram& program,
        const GlBuffer& sortParamsUbo,
        const GlBuffer& sortEntrySsbo,
        std::uint32_t elementCount,
        bool ascending = true
    ) const;

    void dispatchTiledLightCull(
        const GlComputeProgram& program,
        const GlBuffer& tileCullUbo,
        const GlBuffer& pointLightSsbo,
        const GlBuffer& tileLightIndexSsbo,
        const GlBuffer& tileLightCountSsbo,
        std::uint32_t screenWidth,
        std::uint32_t screenHeight
    ) const;

    // Binds the compacted indirect command buffer and issues glMultiDrawElementsIndirect.
    void submitIndirectDraws(
        const GlBuffer& drawCommandBuffer,
        std::uint32_t drawCount,
        gl46::GLenum primitiveMode = gl46::kGlTriangles,
        gl46::GLenum indexType = gl46::kGlUnsignedInt
    ) const;

    static constexpr std::uint32_t ceilDiv(std::uint32_t value, std::uint32_t divisor) noexcept {
        return (value + divisor - 1u) / divisor;
    }

private:
    gl46::DispatchTable gl_{};
    ShaderPreprocessor preprocessor_{};
};

} // namespace glengine
