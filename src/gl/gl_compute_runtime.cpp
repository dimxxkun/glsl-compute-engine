#include "glengine/gl_compute_runtime.hpp"

#include "glengine/reference_algorithms.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace glengine {

namespace gl46 {

namespace {

template <typename FnPtr>
FnPtr loadProc(GlProcLoader loader, const char* name) {
    static_assert(sizeof(FnPtr) == sizeof(GlProcAddress));
    const GlProcAddress raw = loader(name);
    FnPtr fn = nullptr;
    std::memcpy(&fn, &raw, sizeof(fn));
    return fn;
}

} // namespace

DispatchTable DispatchTable::load(GlProcLoader loader) {
    if (!loader) {
        throw std::invalid_argument("OpenGL 4.6 proc loader callback must not be null");
    }

    DispatchTable table{};
    table.createBuffers = loadProc<PfnGlCreateBuffers>(loader, "glCreateBuffers");
    table.deleteBuffers = loadProc<PfnGlDeleteBuffers>(loader, "glDeleteBuffers");
    table.namedBufferStorage = loadProc<PfnGlNamedBufferStorage>(loader, "glNamedBufferStorage");
    table.namedBufferSubData = loadProc<PfnGlNamedBufferSubData>(loader, "glNamedBufferSubData");
    table.getNamedBufferSubData =
        loadProc<PfnGlGetNamedBufferSubData>(loader, "glGetNamedBufferSubData");
    table.bindBuffer = loadProc<PfnGlBindBuffer>(loader, "glBindBuffer");
    table.bindBufferBase = loadProc<PfnGlBindBufferBase>(loader, "glBindBufferBase");

    table.createShader = loadProc<PfnGlCreateShader>(loader, "glCreateShader");
    table.deleteShader = loadProc<PfnGlDeleteShader>(loader, "glDeleteShader");
    table.shaderSource = loadProc<PfnGlShaderSource>(loader, "glShaderSource");
    table.compileShader = loadProc<PfnGlCompileShader>(loader, "glCompileShader");
    table.getShaderiv = loadProc<PfnGlGetShaderiv>(loader, "glGetShaderiv");
    table.getShaderInfoLog = loadProc<PfnGlGetShaderInfoLog>(loader, "glGetShaderInfoLog");

    table.createProgram = loadProc<PfnGlCreateProgram>(loader, "glCreateProgram");
    table.deleteProgram = loadProc<PfnGlDeleteProgram>(loader, "glDeleteProgram");
    table.attachShader = loadProc<PfnGlAttachShader>(loader, "glAttachShader");
    table.linkProgram = loadProc<PfnGlLinkProgram>(loader, "glLinkProgram");
    table.getProgramiv = loadProc<PfnGlGetProgramiv>(loader, "glGetProgramiv");
    table.getProgramInfoLog = loadProc<PfnGlGetProgramInfoLog>(loader, "glGetProgramInfoLog");
    table.useProgram = loadProc<PfnGlUseProgram>(loader, "glUseProgram");

    table.dispatchCompute = loadProc<PfnGlDispatchCompute>(loader, "glDispatchCompute");
    table.memoryBarrier = loadProc<PfnGlMemoryBarrier>(loader, "glMemoryBarrier");
    table.multiDrawElementsIndirect =
        loadProc<PfnGlMultiDrawElementsIndirect>(loader, "glMultiDrawElementsIndirect");

    table.createQueries = loadProc<PfnGlCreateQueries>(loader, "glCreateQueries");
    table.deleteQueries = loadProc<PfnGlDeleteQueries>(loader, "glDeleteQueries");
    table.beginQuery = loadProc<PfnGlBeginQuery>(loader, "glBeginQuery");
    table.endQuery = loadProc<PfnGlEndQuery>(loader, "glEndQuery");
    table.getQueryObjectui64v = loadProc<PfnGlGetQueryObjectui64v>(loader, "glGetQueryObjectui64v");

    table.getString = loadProc<PfnGlGetString>(loader, "glGetString");
    return table;
}

} // namespace gl46

// =============================================================================
// GlBuffer
// =============================================================================

GlBuffer::GlBuffer(
    const gl46::DispatchTable* gl,
    std::size_t sizeBytes,
    const void* initialData,
    gl46::GLbitfield storageFlags
)
    : gl_(gl), sizeBytes_(sizeBytes) {
    if (!gl_ || !gl_->createBuffers || !gl_->namedBufferStorage) {
        throw std::invalid_argument("GlBuffer requires a valid OpenGL 4.6 dispatch table");
    }
    if (sizeBytes == 0) {
        throw std::invalid_argument("GlBuffer sizeBytes must be greater than zero");
    }

    gl_->createBuffers(1, &handle_);
    gl_->namedBufferStorage(
        handle_,
        static_cast<gl46::GLsizeiptr>(sizeBytes),
        initialData,
        storageFlags
    );
}

GlBuffer::~GlBuffer() {
    release();
}

GlBuffer::GlBuffer(GlBuffer&& other) noexcept
    : gl_(other.gl_), handle_(other.handle_), sizeBytes_(other.sizeBytes_) {
    other.gl_ = nullptr;
    other.handle_ = 0;
    other.sizeBytes_ = 0;
}

GlBuffer& GlBuffer::operator=(GlBuffer&& other) noexcept {
    if (this != &other) {
        release();
        gl_ = other.gl_;
        handle_ = other.handle_;
        sizeBytes_ = other.sizeBytes_;
        other.gl_ = nullptr;
        other.handle_ = 0;
        other.sizeBytes_ = 0;
    }
    return *this;
}

void GlBuffer::release() noexcept {
    if (gl_ && gl_->deleteBuffers && handle_ != 0) {
        gl_->deleteBuffers(1, &handle_);
    }
    handle_ = 0;
    sizeBytes_ = 0;
}

void GlBuffer::uploadSubData(
    std::size_t offsetBytes,
    std::size_t sizeBytes,
    const void* data
) const {
    if (!valid() || !gl_ || !gl_->namedBufferSubData) {
        throw std::logic_error("GlBuffer::uploadSubData called on an invalid buffer");
    }
    if (offsetBytes + sizeBytes > sizeBytes_) {
        throw std::out_of_range("GlBuffer::uploadSubData exceeds buffer capacity");
    }
    gl_->namedBufferSubData(
        handle_,
        static_cast<gl46::GLintptr>(offsetBytes),
        static_cast<gl46::GLsizeiptr>(sizeBytes),
        data
    );
}

void GlBuffer::downloadSubData(
    std::size_t offsetBytes,
    std::size_t sizeBytes,
    void* outData
) const {
    if (!valid() || !gl_ || !gl_->getNamedBufferSubData) {
        throw std::logic_error("GlBuffer::downloadSubData called on an invalid buffer");
    }
    if (offsetBytes + sizeBytes > sizeBytes_) {
        throw std::out_of_range("GlBuffer::downloadSubData exceeds buffer capacity");
    }
    gl_->getNamedBufferSubData(
        handle_,
        static_cast<gl46::GLintptr>(offsetBytes),
        static_cast<gl46::GLsizeiptr>(sizeBytes),
        outData
    );
}

void GlBuffer::bindBase(gl46::GLenum target, gl46::GLuint bindingIndex) const {
    if (!valid() || !gl_ || !gl_->bindBufferBase) {
        throw std::logic_error("GlBuffer::bindBase called on an invalid buffer");
    }
    gl_->bindBufferBase(target, bindingIndex, handle_);
}

// =============================================================================
// GlComputeProgram
// =============================================================================

GlComputeProgram::GlComputeProgram(const gl46::DispatchTable* gl, const std::string& glslSource)
    : gl_(gl) {
    if (!gl_ || !gl_->isComplete()) {
        throw std::invalid_argument("GlComputeProgram requires a complete OpenGL 4.6 dispatch table");
    }

    const gl46::GLuint shader = gl_->createShader(gl46::kGlComputeShader);
    const char* srcPtr = glslSource.c_str();
    const auto srcLen = static_cast<gl46::GLint>(glslSource.size());
    gl_->shaderSource(shader, 1, &srcPtr, &srcLen);
    gl_->compileShader(shader);

    gl46::GLint compiled = 0;
    gl_->getShaderiv(shader, gl46::kGlCompileStatus, &compiled);
    if (compiled == 0) {
        gl46::GLint logLen = 0;
        gl_->getShaderiv(shader, gl46::kGlInfoLogLength, &logLen);
        std::string infoLog(static_cast<std::size_t>(std::max(logLen, 1)), '\0');
        gl_->getShaderInfoLog(shader, logLen, nullptr, infoLog.data());
        gl_->deleteShader(shader);
        throw std::runtime_error("Compute shader compilation failed:\n" + infoLog);
    }

    program_ = gl_->createProgram();
    gl_->attachShader(program_, shader);
    gl_->linkProgram(program_);
    gl_->deleteShader(shader);

    gl46::GLint linked = 0;
    gl_->getProgramiv(program_, gl46::kGlLinkStatus, &linked);
    if (linked == 0) {
        gl46::GLint logLen = 0;
        gl_->getProgramiv(program_, gl46::kGlInfoLogLength, &logLen);
        std::string infoLog(static_cast<std::size_t>(std::max(logLen, 1)), '\0');
        gl_->getProgramInfoLog(program_, logLen, nullptr, infoLog.data());
        release();
        throw std::runtime_error("Compute program link failed:\n" + infoLog);
    }
}

GlComputeProgram::~GlComputeProgram() {
    release();
}

GlComputeProgram::GlComputeProgram(GlComputeProgram&& other) noexcept
    : gl_(other.gl_), program_(other.program_) {
    other.gl_ = nullptr;
    other.program_ = 0;
}

GlComputeProgram& GlComputeProgram::operator=(GlComputeProgram&& other) noexcept {
    if (this != &other) {
        release();
        gl_ = other.gl_;
        program_ = other.program_;
        other.gl_ = nullptr;
        other.program_ = 0;
    }
    return *this;
}

void GlComputeProgram::release() noexcept {
    if (gl_ && gl_->deleteProgram && program_ != 0) {
        gl_->deleteProgram(program_);
    }
    program_ = 0;
}

void GlComputeProgram::bind() const {
    if (!valid() || !gl_ || !gl_->useProgram) {
        throw std::logic_error("GlComputeProgram::bind called on an invalid program");
    }
    gl_->useProgram(program_);
}

// =============================================================================
// GlComputeRuntime
// =============================================================================

GlComputeRuntime::GlComputeRuntime(gl46::GlProcLoader procLoader)
    : GlComputeRuntime(gl46::DispatchTable::load(procLoader)) {}

GlComputeRuntime::GlComputeRuntime(gl46::DispatchTable dispatchTable)
    : gl_(dispatchTable) {
    if (!gl_.isComplete()) {
        throw std::runtime_error(
            "OpenGL 4.6 DSA dispatch table is incomplete; verify an active OpenGL 4.6 core context"
        );
    }
}

GlTimerBackend GlComputeRuntime::makeTimerBackend() const {
    const gl46::DispatchTable glCopy = gl_;
    GlTimerBackend backend{};
    backend.createQueries = [glCopy](std::uint32_t count, std::uint32_t* ids) {
        glCopy.createQueries(gl46::kGlTimeElapsed, static_cast<gl46::GLsizei>(count), ids);
    };
    backend.deleteQueries = [glCopy](std::uint32_t count, const std::uint32_t* ids) {
        glCopy.deleteQueries(static_cast<gl46::GLsizei>(count), ids);
    };
    backend.beginTimeElapsed = [glCopy](std::uint32_t queryId) {
        glCopy.beginQuery(gl46::kGlTimeElapsed, queryId);
    };
    backend.endTimeElapsed = [glCopy]() {
        glCopy.endQuery(gl46::kGlTimeElapsed);
    };
    backend.pollResultNs = [glCopy](std::uint32_t queryId, std::uint64_t& outNs) -> bool {
        gl46::GLuint64 available = 0;
        glCopy.getQueryObjectui64v(queryId, gl46::kGlQueryResultAvailable, &available);
        if (available == 0) {
            return false;
        }
        gl46::GLuint64 elapsedNs = 0;
        glCopy.getQueryObjectui64v(queryId, gl46::kGlQueryResult, &elapsedNs);
        outNs = elapsedNs;
        return true;
    };
    return backend;
}

GlBuffer GlComputeRuntime::createImmutableBuffer(
    std::size_t sizeBytes,
    const void* initialData,
    gl46::GLbitfield storageFlags
) const {
    return GlBuffer(&gl_, sizeBytes, initialData, storageFlags);
}

GlComputeProgram GlComputeRuntime::compileComputeShaderFile(
    const std::filesystem::path& shaderPath,
    const PreprocessOptions& options
) const {
    const PreprocessResult preprocessed = preprocessor_.preprocessFile(shaderPath, options);
    if (preprocessed.stage != ShaderStage::Compute) {
        throw std::invalid_argument(
            "Expected compute shader (.comp), got: " + shaderPath.string()
        );
    }
    return GlComputeProgram(&gl_, preprocessed.source);
}

void GlComputeRuntime::dispatchCurlParticles(
    const GlComputeProgram& program,
    const GlBuffer& simParamsUbo,
    const GlBuffer& particleSsbo,
    std::uint32_t particleCount
) const {
    if (particleCount == 0) {
        return;
    }
    program.bind();
    simParamsUbo.bindBase(gl46::kGlUniformBuffer, 0);
    particleSsbo.bindBase(gl46::kGlShaderStorageBuffer, 0);
    gl_.dispatchCompute(ceilDiv(particleCount, 256u), 1u, 1u);
    gl_.memoryBarrier(gl46::kGlShaderStorageBarrierBit);
}

void GlComputeRuntime::dispatchFrustumCull(
    const GlComputeProgram& program,
    const GlBuffer& frustumUbo,
    const GlBuffer& boundsSsbo,
    const GlBuffer& drawCommandSsbo,
    const GlBuffer& visibleInstanceSsbo,
    std::uint32_t objectCount
) const {
    if (objectCount == 0) {
        return;
    }
    program.bind();
    frustumUbo.bindBase(gl46::kGlUniformBuffer, 0);
    boundsSsbo.bindBase(gl46::kGlShaderStorageBuffer, 0);
    drawCommandSsbo.bindBase(gl46::kGlShaderStorageBuffer, 1);
    visibleInstanceSsbo.bindBase(gl46::kGlShaderStorageBuffer, 2);
    gl_.dispatchCompute(ceilDiv(objectCount, 64u), 1u, 1u);
    gl_.memoryBarrier(gl46::kGlCommandBarrierBit | gl46::kGlShaderStorageBarrierBit);
}

void GlComputeRuntime::dispatchBitonicSort(
    const GlComputeProgram& program,
    const GlBuffer& sortParamsUbo,
    const GlBuffer& sortEntrySsbo,
    std::uint32_t elementCount,
    bool ascending
) const {
    if (elementCount <= 1u) {
        return;
    }
    if (!isPowerOfTwo(elementCount)) {
        throw std::invalid_argument("dispatchBitonicSort requires a power-of-two elementCount");
    }

    program.bind();
    sortParamsUbo.bindBase(gl46::kGlUniformBuffer, 0);
    sortEntrySsbo.bindBase(gl46::kGlShaderStorageBuffer, 0);

    const std::uint32_t groupsX = ceilDiv(elementCount, 256u);
    for (std::uint32_t stageK = 2u; stageK <= elementCount; stageK <<= 1u) {
        for (std::uint32_t passJ = stageK >> 1u; passJ > 0u; passJ >>= 1u) {
            BitonicSortParamsUbo params{};
            params.params = UVec4{elementCount, stageK, passJ, ascending ? 1u : 0u};
            sortParamsUbo.uploadSubData(0, sizeof(BitonicSortParamsUbo), &params);

            gl_.dispatchCompute(groupsX, 1u, 1u);
            gl_.memoryBarrier(gl46::kGlShaderStorageBarrierBit);
        }
    }
}

void GlComputeRuntime::dispatchTiledLightCull(
    const GlComputeProgram& program,
    const GlBuffer& tileCullUbo,
    const GlBuffer& pointLightSsbo,
    const GlBuffer& tileLightIndexSsbo,
    const GlBuffer& tileLightCountSsbo,
    std::uint32_t screenWidth,
    std::uint32_t screenHeight
) const {
    if (screenWidth == 0u || screenHeight == 0u) {
        return;
    }

    program.bind();
    tileCullUbo.bindBase(gl46::kGlUniformBuffer, 0);
    pointLightSsbo.bindBase(gl46::kGlShaderStorageBuffer, 0);
    tileLightIndexSsbo.bindBase(gl46::kGlShaderStorageBuffer, 1);
    tileLightCountSsbo.bindBase(gl46::kGlShaderStorageBuffer, 2);

    const std::uint32_t tilesX = ceilDiv(screenWidth, 16u);
    const std::uint32_t tilesY = ceilDiv(screenHeight, 16u);
    gl_.dispatchCompute(tilesX, tilesY, 1u);
    gl_.memoryBarrier(gl46::kGlShaderStorageBarrierBit);
}

void GlComputeRuntime::submitIndirectDraws(
    const GlBuffer& drawCommandBuffer,
    std::uint32_t drawCount,
    gl46::GLenum primitiveMode,
    gl46::GLenum indexType
) const {
    if (drawCount == 0u) {
        return;
    }
    gl_.bindBuffer(gl46::kGlDrawIndirectBuffer, drawCommandBuffer.handle());
    gl_.multiDrawElementsIndirect(
        primitiveMode,
        indexType,
        nullptr,
        static_cast<gl46::GLsizei>(drawCount),
        static_cast<gl46::GLsizei>(sizeof(DrawElementsIndirectCommand))
    );
}

} // namespace glengine
