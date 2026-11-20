#pragma once

#include "glengine/math.hpp"

#include <cstdint>
#include <vector>

namespace glengine {

struct PbrSurface {
    Vec3  albedo{0.5f, 0.5f, 0.5f};
    float metallic = 0.0f;
    float roughness = 0.5f;
    Vec3  normal{0.0f, 1.0f, 0.0f};
};

// CPU reference implementation matching shaders/compute/frustum_cull.comp.
// Resets commands[i].instanceCount to 0, culls objects against frustum planes,
// compacts visible instanceDataIndex entries into visibleInstanceIds, and returns
// total visible instances written.
std::uint32_t frustumCullAndCompact(
    const Frustum& frustum,
    const std::vector<ObjectBounds>& objects,
    std::vector<DrawElementsIndirectCommand>& commands,
    std::vector<std::uint32_t>& visibleInstanceIds
);

// Returns true iff n is a non-zero power of two.
constexpr bool isPowerOfTwo(std::size_t n) noexcept {
    return n != 0 && (n & (n - 1)) == 0;
}

// CPU reference implementation matching shaders/compute/bitonic_sort.comp.
// Throws std::invalid_argument if entries.size() is not a power of two (when non-empty).
void bitonicSortCpu(std::vector<GpuSortEntry>& entries, bool ascending = true);

// Cook-Torrance GGX microfacet BRDF functions matching shaders/graphics/pbr_lighting.frag.
float distributionGGX(float nDotH, float roughness);
float geometrySchlickGGX(float nDotX, float roughness);
float geometrySmith(float nDotV, float nDotL, float roughness);
Vec3  fresnelSchlick(float cosTheta, const Vec3& f0);

Vec3 evaluateCookTorranceBRDF(
    const PbrSurface& surface,
    const Vec3& viewDir,
    const Vec3& lightDir,
    const Vec3& radiance
);

float pointLightAttenuation(float distanceToLight, float radius);
Vec3  acesFilmicToneMap(const Vec3& hdr);

} // namespace glengine
