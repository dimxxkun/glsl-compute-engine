#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace glengine {

inline constexpr float kPi = 3.14159265358979323846f;

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;

    constexpr Vec2() = default;
    constexpr Vec2(float xVal, float yVal) : x(xVal), y(yVal) {}
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    constexpr Vec3() = default;
    constexpr explicit Vec3(float s) : x(s), y(s), z(s) {}
    constexpr Vec3(float xVal, float yVal, float zVal) : x(xVal), y(yVal), z(zVal) {}

    constexpr Vec3 operator+(const Vec3& rhs) const { return {x + rhs.x, y + rhs.y, z + rhs.z}; }
    constexpr Vec3 operator-(const Vec3& rhs) const { return {x - rhs.x, y - rhs.y, z - rhs.z}; }
    constexpr Vec3 operator*(const Vec3& rhs) const { return {x * rhs.x, y * rhs.y, z * rhs.z}; }
    constexpr Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    constexpr Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    constexpr Vec3 operator-() const { return {-x, -y, -z}; }

    constexpr Vec3& operator+=(const Vec3& rhs) {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    constexpr float operator[](std::size_t idx) const {
        return idx == 0 ? x : (idx == 1 ? y : z);
    }

    constexpr float& operator[](std::size_t idx) {
        return idx == 0 ? x : (idx == 1 ? y : z);
    }
};

constexpr Vec3 operator*(float s, const Vec3& v) { return v * s; }

inline constexpr float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline constexpr Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

inline float length(const Vec3& v) {
    return std::sqrt(dot(v, v));
}

inline Vec3 normalize(const Vec3& v) {
    const float len = length(v);
    if (len <= 1e-7f) {
        return {0.0f, 0.0f, 0.0f};
    }
    return v / len;
}

inline Vec3 clampVec3(const Vec3& v, float minVal, float maxVal) {
    return {
        std::clamp(v.x, minVal, maxVal),
        std::clamp(v.y, minVal, maxVal),
        std::clamp(v.z, minVal, maxVal)
    };
}

inline constexpr Vec3 lerp(const Vec3& a, const Vec3& b, float t) {
    return a * (1.0f - t) + b * t;
}

struct alignas(16) Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;

    constexpr Vec4() = default;
    constexpr Vec4(float xVal, float yVal, float zVal, float wVal)
        : x(xVal), y(yVal), z(zVal), w(wVal) {}
    constexpr Vec4(const Vec3& v, float wVal) : x(v.x), y(v.y), z(v.z), w(wVal) {}

    constexpr Vec3 xyz() const { return {x, y, z}; }

    constexpr Vec4 operator+(const Vec4& rhs) const {
        return {x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w};
    }
    constexpr Vec4 operator-(const Vec4& rhs) const {
        return {x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w};
    }
    constexpr Vec4 operator*(float s) const {
        return {x * s, y * s, z * s, w * s};
    }

    constexpr float operator[](std::size_t idx) const {
        return idx == 0 ? x : (idx == 1 ? y : (idx == 2 ? z : w));
    }

    constexpr float& operator[](std::size_t idx) {
        return idx == 0 ? x : (idx == 1 ? y : (idx == 2 ? z : w));
    }
};

struct alignas(16) UVec4 {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t z = 0;
    std::uint32_t w = 0;

    constexpr UVec4() = default;
    constexpr UVec4(std::uint32_t xVal, std::uint32_t yVal, std::uint32_t zVal, std::uint32_t wVal)
        : x(xVal), y(yVal), z(zVal), w(wVal) {}
};

// Column-major 4x4 matrix matching OpenGL / GLSL mat4 memory layout.
struct alignas(16) Mat4 {
    std::array<Vec4, 4> cols{{
        Vec4{1.0f, 0.0f, 0.0f, 0.0f},
        Vec4{0.0f, 1.0f, 0.0f, 0.0f},
        Vec4{0.0f, 0.0f, 1.0f, 0.0f},
        Vec4{0.0f, 0.0f, 0.0f, 1.0f}
    }};

    static constexpr Mat4 identity() { return Mat4{}; }

    constexpr float at(std::size_t row, std::size_t col) const {
        return cols[col][row];
    }

    constexpr Vec4 row(std::size_t r) const {
        return {cols[0][r], cols[1][r], cols[2][r], cols[3][r]};
    }

    static constexpr Mat4 translation(const Vec3& t) {
        Mat4 m = identity();
        m.cols[3] = Vec4{t.x, t.y, t.z, 1.0f};
        return m;
    }

    static constexpr Mat4 scale(const Vec3& s) {
        Mat4 m = identity();
        m.cols[0].x = s.x;
        m.cols[1].y = s.y;
        m.cols[2].z = s.z;
        return m;
    }

    // Right-handed OpenGL perspective projection (NDC z in [-1, 1]).
    static Mat4 perspective(float fovYRadians, float aspect, float zNear, float zFar) {
        Mat4 m{};
        m.cols.fill(Vec4{});
        const float tanHalfFov = std::tan(0.5f * fovYRadians);
        const float f = 1.0f / tanHalfFov;

        m.cols[0].x = f / aspect;
        m.cols[1].y = f;
        m.cols[2].z = -(zFar + zNear) / (zFar - zNear);
        m.cols[2].w = -1.0f;
        m.cols[3].z = -(2.0f * zFar * zNear) / (zFar - zNear);
        return m;
    }

    // Right-handed look-at view matrix (-Z forward).
    static Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
        const Vec3 f = normalize(center - eye);
        const Vec3 s = normalize(cross(f, up));
        const Vec3 u = cross(s, f);

        Mat4 m = identity();
        m.cols[0] = Vec4{s.x, u.x, -f.x, 0.0f};
        m.cols[1] = Vec4{s.y, u.y, -f.y, 0.0f};
        m.cols[2] = Vec4{s.z, u.z, -f.z, 0.0f};
        m.cols[3] = Vec4{-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0f};
        return m;
    }

    constexpr Vec4 operator*(const Vec4& v) const {
        return cols[0] * v.x + cols[1] * v.y + cols[2] * v.z + cols[3] * v.w;
    }

    constexpr Mat4 operator*(const Mat4& rhs) const {
        Mat4 result{};
        for (std::size_t c = 0; c < 4; ++c) {
            result.cols[c] = (*this) * rhs.cols[c];
        }
        return result;
    }
};

struct AABB {
    Vec3 minBounds{0.0f, 0.0f, 0.0f};
    Vec3 maxBounds{0.0f, 0.0f, 0.0f};

    constexpr Vec3 center() const { return (minBounds + maxBounds) * 0.5f; }
    constexpr Vec3 extents() const { return (maxBounds - minBounds) * 0.5f; }
};

struct Frustum {
    // Planes stored as (nx, ny, nz, d) normalized inward: dot(n, p) + d >= 0 is inside.
    // Order: Left, Right, Bottom, Top, Near, Far.
    std::array<Vec4, 6> planes{};

    // Gribb-Hartmann 6-plane extraction from column-major OpenGL View-Projection matrix.
    static Frustum fromViewProjection(const Mat4& vp) {
        const Vec4 r0 = vp.row(0);
        const Vec4 r1 = vp.row(1);
        const Vec4 r2 = vp.row(2);
        const Vec4 r3 = vp.row(3);

        auto normalizePlane = [](const Vec4& p) -> Vec4 {
            const float invLen = 1.0f / std::max(length(p.xyz()), 1e-7f);
            return p * invLen;
        };

        Frustum frustum{};
        frustum.planes[0] = normalizePlane(r3 + r0); // Left
        frustum.planes[1] = normalizePlane(r3 - r0); // Right
        frustum.planes[2] = normalizePlane(r3 + r1); // Bottom
        frustum.planes[3] = normalizePlane(r3 - r1); // Top
        frustum.planes[4] = normalizePlane(r3 + r2); // Near
        frustum.planes[5] = normalizePlane(r3 - r2); // Far
        return frustum;
    }

    // Positive-vertex (p-vertex) AABB visibility test matching frustum_cull.comp.
    bool intersectsAABB(const AABB& box) const {
        for (const Vec4& plane : planes) {
            const Vec3 pVertex{
                plane.x >= 0.0f ? box.maxBounds.x : box.minBounds.x,
                plane.y >= 0.0f ? box.maxBounds.y : box.minBounds.y,
                plane.z >= 0.0f ? box.maxBounds.z : box.minBounds.z
            };
            if (dot(plane.xyz(), pVertex) + plane.w < 0.0f) {
                return false;
            }
        }
        return true;
    }
};

// =============================================================================
// GPU POD Structs — Exact std140 / std430 Layout Parity with GLSL 4.60 Shaders
// =============================================================================

struct alignas(16) GpuParticle {
    Vec4 positionAge;   // xyz = world position, w = age [0, 1]
    Vec4 velocityMass;  // xyz = velocity, w = mass
};

struct alignas(16) GpuPointLight {
    Vec4 positionRadius;  // xyz = world position, w = influence radius
    Vec4 colorIntensity;  // rgb = linear color, w = intensity
};

struct alignas(16) ObjectBounds {
    Vec4  aabbMin;  // xyz = world AABB min, w = bounding sphere radius
    Vec4  aabbMax;  // xyz = world AABB max, w = padding
    UVec4 meta;     // x = drawCommandIndex, y = instanceDataIndex, z = materialId, w = flags (bit 0 = enabled)
};

// Matches OpenGL 4.6 DrawElementsIndirectCommand & std430 layout in frustum_cull.comp.
struct DrawElementsIndirectCommand {
    std::uint32_t count = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t firstIndex = 0;
    std::int32_t  baseVertex = 0;
    std::uint32_t baseInstance = 0;
};

struct GpuSortEntry {
    float         depthKey = 0.0f;
    std::uint32_t payloadIndex = 0;
};

struct alignas(16) CameraFrustumUbo {
    std::array<Vec4, 6> planes{};
    UVec4               cullParams{}; // x = objectCount, y = maxVisibleInstances, z = drawCommandCount, w = 0
};

struct alignas(16) CurlSimParamsUbo {
    Vec4  timeParams{};
    Vec4  attractorPosStr{};
    Vec4  boundsAndDrag{};
    Vec4  physicsParams{};
    UVec4 counts{};
};

struct alignas(16) BitonicSortParamsUbo {
    UVec4 params{}; // x = elementCount, y = stageK, z = passJ, w = ascending
};

struct alignas(16) TileCullUbo {
    Mat4  viewMatrix = Mat4::identity();
    Mat4  invProjectionMatrix = Mat4::identity();
    UVec4 screenAndTiles{};
    UVec4 lightParams{};
};

struct alignas(16) InstanceDataGpu {
    Mat4  modelMatrix = Mat4::identity();
    Mat4  normalMatrix = Mat4::identity();
    Vec4  baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
    Vec4  pbrFactors{0.0f, 0.5f, 1.0f, 0.0f};
    UVec4 materialMeta{};
};

// Compile-time std140 / std430 ABI verification
static_assert(sizeof(Vec4) == 16 && alignof(Vec4) == 16);
static_assert(sizeof(UVec4) == 16 && alignof(UVec4) == 16);
static_assert(sizeof(Mat4) == 64 && alignof(Mat4) == 16);

static_assert(sizeof(GpuParticle) == 32 && alignof(GpuParticle) == 16);
static_assert(offsetof(GpuParticle, positionAge) == 0);
static_assert(offsetof(GpuParticle, velocityMass) == 16);

static_assert(sizeof(GpuPointLight) == 32 && alignof(GpuPointLight) == 16);
static_assert(offsetof(GpuPointLight, positionRadius) == 0);
static_assert(offsetof(GpuPointLight, colorIntensity) == 16);

static_assert(sizeof(ObjectBounds) == 48 && alignof(ObjectBounds) == 16);
static_assert(offsetof(ObjectBounds, aabbMin) == 0);
static_assert(offsetof(ObjectBounds, aabbMax) == 16);
static_assert(offsetof(ObjectBounds, meta) == 32);

static_assert(sizeof(DrawElementsIndirectCommand) == 20 && alignof(DrawElementsIndirectCommand) == 4);
static_assert(offsetof(DrawElementsIndirectCommand, count) == 0);
static_assert(offsetof(DrawElementsIndirectCommand, instanceCount) == 4);
static_assert(offsetof(DrawElementsIndirectCommand, firstIndex) == 8);
static_assert(offsetof(DrawElementsIndirectCommand, baseVertex) == 12);
static_assert(offsetof(DrawElementsIndirectCommand, baseInstance) == 16);

static_assert(sizeof(GpuSortEntry) == 8 && alignof(GpuSortEntry) == 4);
static_assert(offsetof(GpuSortEntry, depthKey) == 0);
static_assert(offsetof(GpuSortEntry, payloadIndex) == 4);

static_assert(sizeof(CameraFrustumUbo) == 112 && alignof(CameraFrustumUbo) == 16);
static_assert(offsetof(CameraFrustumUbo, planes) == 0);
static_assert(offsetof(CameraFrustumUbo, cullParams) == 96);

static_assert(sizeof(CurlSimParamsUbo) == 80 && alignof(CurlSimParamsUbo) == 16);
static_assert(sizeof(BitonicSortParamsUbo) == 16 && alignof(BitonicSortParamsUbo) == 16);
static_assert(sizeof(TileCullUbo) == 160 && alignof(TileCullUbo) == 16);
static_assert(sizeof(InstanceDataGpu) == 176 && alignof(InstanceDataGpu) == 16);

} // namespace glengine
