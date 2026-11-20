#pragma once

#include <cstddef>
#include <cstdint>

namespace glengine::gl46 {

#if defined(_WIN32)
#define GLENGINE_GL_APIENTRY __stdcall
#else
#define GLENGINE_GL_APIENTRY
#endif

using GLenum = std::uint32_t;
using GLbitfield = std::uint32_t;
using GLuint = std::uint32_t;
using GLint = std::int32_t;
using GLsizei = std::int32_t;
using GLboolean = std::uint8_t;
using GLchar = char;
using GLubyte = std::uint8_t;
using GLsizeiptr = std::ptrdiff_t;
using GLintptr = std::ptrdiff_t;
using GLuint64 = std::uint64_t;

// OpenGL 4.6 Core / DSA Constants
inline constexpr GLenum kGlTriangles = 0x0004u;
inline constexpr GLenum kGlUnsignedInt = 0x1405u;
inline constexpr GLenum kGlFalse = 0u;
inline constexpr GLenum kGlTrue = 1u;

inline constexpr GLenum kGlVendor = 0x1F00u;
inline constexpr GLenum kGlRenderer = 0x1F01u;
inline constexpr GLenum kGlVersion = 0x1F02u;

inline constexpr GLenum kGlUniformBuffer = 0x8A11u;
inline constexpr GLenum kGlShaderStorageBuffer = 0x90D2u;
inline constexpr GLenum kGlDrawIndirectBuffer = 0x8F3Fu;

inline constexpr GLbitfield kGlDynamicStorageBit = 0x0100u;
inline constexpr GLbitfield kGlMapReadBit = 0x0001u;
inline constexpr GLbitfield kGlMapWriteBit = 0x0002u;
inline constexpr GLbitfield kGlMapPersistentBit = 0x0040u;
inline constexpr GLbitfield kGlMapCoherentBit = 0x0080u;

inline constexpr GLbitfield kGlCommandBarrierBit = 0x00000040u;
inline constexpr GLbitfield kGlShaderStorageBarrierBit = 0x00002000u;
inline constexpr GLbitfield kGlBufferUpdateBarrierBit = 0x00000200u;
inline constexpr GLbitfield kGlTextureFetchBarrierBit = 0x00000008u;

inline constexpr GLenum kGlComputeShader = 0x91B9u;
inline constexpr GLenum kGlVertexShader = 0x8B31u;
inline constexpr GLenum kGlFragmentShader = 0x8B30u;
inline constexpr GLenum kGlCompileStatus = 0x8B81u;
inline constexpr GLenum kGlLinkStatus = 0x8B82u;
inline constexpr GLenum kGlInfoLogLength = 0x8B84u;

inline constexpr GLenum kGlTimeElapsed = 0x88BFu;
inline constexpr GLenum kGlQueryResult = 0x8866u;
inline constexpr GLenum kGlQueryResultAvailable = 0x8867u;

using GlProcAddress = void (*)();
using GlProcLoader = GlProcAddress (*)(const char* procName);

using PfnGlCreateBuffers = void(GLENGINE_GL_APIENTRY*)(GLsizei n, GLuint* buffers);
using PfnGlDeleteBuffers = void(GLENGINE_GL_APIENTRY*)(GLsizei n, const GLuint* buffers);
using PfnGlNamedBufferStorage = void(GLENGINE_GL_APIENTRY*)(
    GLuint buffer, GLsizeiptr size, const void* data, GLbitfield flags);
using PfnGlNamedBufferSubData = void(GLENGINE_GL_APIENTRY*)(
    GLuint buffer, GLintptr offset, GLsizeiptr size, const void* data);
using PfnGlGetNamedBufferSubData = void(GLENGINE_GL_APIENTRY*)(
    GLuint buffer, GLintptr offset, GLsizeiptr size, void* data);
using PfnGlBindBuffer = void(GLENGINE_GL_APIENTRY*)(GLenum target, GLuint buffer);
using PfnGlBindBufferBase = void(GLENGINE_GL_APIENTRY*)(GLenum target, GLuint index, GLuint buffer);

using PfnGlCreateShader = GLuint(GLENGINE_GL_APIENTRY*)(GLenum type);
using PfnGlDeleteShader = void(GLENGINE_GL_APIENTRY*)(GLuint shader);
using PfnGlShaderSource = void(GLENGINE_GL_APIENTRY*)(
    GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length);
using PfnGlCompileShader = void(GLENGINE_GL_APIENTRY*)(GLuint shader);
using PfnGlGetShaderiv = void(GLENGINE_GL_APIENTRY*)(GLuint shader, GLenum pname, GLint* params);
using PfnGlGetShaderInfoLog = void(GLENGINE_GL_APIENTRY*)(
    GLuint shader, GLsizei maxLength, GLsizei* length, GLchar* infoLog);

using PfnGlCreateProgram = GLuint(GLENGINE_GL_APIENTRY*)();
using PfnGlDeleteProgram = void(GLENGINE_GL_APIENTRY*)(GLuint program);
using PfnGlAttachShader = void(GLENGINE_GL_APIENTRY*)(GLuint program, GLuint shader);
using PfnGlLinkProgram = void(GLENGINE_GL_APIENTRY*)(GLuint program);
using PfnGlGetProgramiv = void(GLENGINE_GL_APIENTRY*)(GLuint program, GLenum pname, GLint* params);
using PfnGlGetProgramInfoLog = void(GLENGINE_GL_APIENTRY*)(
    GLuint program, GLsizei maxLength, GLsizei* length, GLchar* infoLog);
using PfnGlUseProgram = void(GLENGINE_GL_APIENTRY*)(GLuint program);

using PfnGlDispatchCompute = void(GLENGINE_GL_APIENTRY*)(
    GLuint numGroupsX, GLuint numGroupsY, GLuint numGroupsZ);
using PfnGlMemoryBarrier = void(GLENGINE_GL_APIENTRY*)(GLbitfield barriers);
using PfnGlMultiDrawElementsIndirect = void(GLENGINE_GL_APIENTRY*)(
    GLenum mode, GLenum type, const void* indirect, GLsizei drawcount, GLsizei stride);

using PfnGlCreateQueries = void(GLENGINE_GL_APIENTRY*)(GLenum target, GLsizei n, GLuint* ids);
using PfnGlDeleteQueries = void(GLENGINE_GL_APIENTRY*)(GLsizei n, const GLuint* ids);
using PfnGlBeginQuery = void(GLENGINE_GL_APIENTRY*)(GLenum target, GLuint id);
using PfnGlEndQuery = void(GLENGINE_GL_APIENTRY*)(GLenum target);
using PfnGlGetQueryObjectui64v = void(GLENGINE_GL_APIENTRY*)(
    GLuint id, GLenum pname, GLuint64* params);

using PfnGlGetString = const GLubyte*(GLENGINE_GL_APIENTRY*)(GLenum name);

struct DispatchTable {
    PfnGlCreateBuffers createBuffers = nullptr;
    PfnGlDeleteBuffers deleteBuffers = nullptr;
    PfnGlNamedBufferStorage namedBufferStorage = nullptr;
    PfnGlNamedBufferSubData namedBufferSubData = nullptr;
    PfnGlGetNamedBufferSubData getNamedBufferSubData = nullptr;
    PfnGlBindBuffer bindBuffer = nullptr;
    PfnGlBindBufferBase bindBufferBase = nullptr;

    PfnGlCreateShader createShader = nullptr;
    PfnGlDeleteShader deleteShader = nullptr;
    PfnGlShaderSource shaderSource = nullptr;
    PfnGlCompileShader compileShader = nullptr;
    PfnGlGetShaderiv getShaderiv = nullptr;
    PfnGlGetShaderInfoLog getShaderInfoLog = nullptr;

    PfnGlCreateProgram createProgram = nullptr;
    PfnGlDeleteProgram deleteProgram = nullptr;
    PfnGlAttachShader attachShader = nullptr;
    PfnGlLinkProgram linkProgram = nullptr;
    PfnGlGetProgramiv getProgramiv = nullptr;
    PfnGlGetProgramInfoLog getProgramInfoLog = nullptr;
    PfnGlUseProgram useProgram = nullptr;

    PfnGlDispatchCompute dispatchCompute = nullptr;
    PfnGlMemoryBarrier memoryBarrier = nullptr;
    PfnGlMultiDrawElementsIndirect multiDrawElementsIndirect = nullptr;

    PfnGlCreateQueries createQueries = nullptr;
    PfnGlDeleteQueries deleteQueries = nullptr;
    PfnGlBeginQuery beginQuery = nullptr;
    PfnGlEndQuery endQuery = nullptr;
    PfnGlGetQueryObjectui64v getQueryObjectui64v = nullptr;

    PfnGlGetString getString = nullptr;

    bool isComplete() const noexcept {
        return createBuffers && deleteBuffers && namedBufferStorage && namedBufferSubData &&
               getNamedBufferSubData && bindBuffer && bindBufferBase &&
               createShader && deleteShader && shaderSource && compileShader &&
               getShaderiv && getShaderInfoLog && createProgram && deleteProgram &&
               attachShader && linkProgram && getProgramiv && getProgramInfoLog &&
               useProgram && dispatchCompute && memoryBarrier && multiDrawElementsIndirect &&
               createQueries && deleteQueries && beginQuery && endQuery &&
               getQueryObjectui64v && getString;
    }

    static DispatchTable load(GlProcLoader loader);
};

} // namespace glengine::gl46
