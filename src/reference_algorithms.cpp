#include "glengine/reference_algorithms.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace glengine {

std::uint32_t frustumCullAndCompact(
    const Frustum& frustum,
    const std::vector<ObjectBounds>& objects,
    std::vector<DrawElementsIndirectCommand>& commands,
    std::vector<std::uint32_t>& visibleInstanceIds
) {
    for (auto& cmd : commands) {
        cmd.instanceCount = 0;
    }

    std::uint32_t totalVisible = 0;
    const std::size_t maxVisible = visibleInstanceIds.size();

    for (const ObjectBounds& obj : objects) {
        if ((obj.meta.w & 1u) == 0u) {
            continue;
        }

        const AABB box{obj.aabbMin.xyz(), obj.aabbMax.xyz()};
        if (!frustum.intersectsAABB(box)) {
            continue;
        }

        const std::uint32_t cmdIdx = obj.meta.x;
        if (cmdIdx >= commands.size()) {
            continue;
        }

        const std::uint32_t localSlot = commands[cmdIdx].instanceCount++;
        const std::size_t writeIndex =
            static_cast<std::size_t>(commands[cmdIdx].baseInstance) +
            static_cast<std::size_t>(localSlot);

        if (writeIndex < maxVisible) {
            visibleInstanceIds[writeIndex] = obj.meta.y;
        }
        ++totalVisible;
    }

    return totalVisible;
}

void bitonicSortCpu(std::vector<GpuSortEntry>& entries, bool ascending) {
    const std::size_t n = entries.size();
    if (n <= 1) {
        return;
    }
    if (!isPowerOfTwo(n)) {
        throw std::invalid_argument("bitonicSortCpu requires a power-of-two element count");
    }

    for (std::size_t stageK = 2; stageK <= n; stageK <<= 1) {
        for (std::size_t passJ = stageK >> 1; passJ > 0; passJ >>= 1) {
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t partner = i ^ passJ;
                if (partner <= i || partner >= n) {
                    continue;
                }

                const bool blockAscending = ((i & stageK) == 0);
                const bool sortAscending = ascending ? blockAscending : !blockAscending;

                const GpuSortEntry& a = entries[i];
                const GpuSortEntry& b = entries[partner];

                const bool outOfOrder = sortAscending
                    ? ((a.depthKey > b.depthKey) ||
                       (a.depthKey == b.depthKey && a.payloadIndex > b.payloadIndex))
                    : ((a.depthKey < b.depthKey) ||
                       (a.depthKey == b.depthKey && a.payloadIndex < b.payloadIndex));

                if (outOfOrder) {
                    std::swap(entries[i], entries[partner]);
                }
            }
        }
    }
}

float distributionGGX(float nDotH, float roughness) {
    const float r = std::clamp(roughness, 0.045f, 1.0f);
    const float a = r * r;
    const float a2 = a * a;
    const float denom = (nDotH * nDotH) * (a2 - 1.0f) + 1.0f;
    return a2 / std::max(kPi * denom * denom, 1e-7f);
}

float geometrySchlickGGX(float nDotX, float roughness) {
    const float r = std::clamp(roughness, 0.045f, 1.0f) + 1.0f;
    const float k = (r * r) * 0.125f;
    return nDotX / std::max(nDotX * (1.0f - k) + k, 1e-7f);
}

float geometrySmith(float nDotV, float nDotL, float roughness) {
    return geometrySchlickGGX(nDotV, roughness) * geometrySchlickGGX(nDotL, roughness);
}

Vec3 fresnelSchlick(float cosTheta, const Vec3& f0) {
    const float oneMinusCos = std::clamp(1.0f - cosTheta, 0.0f, 1.0f);
    const float sq = oneMinusCos * oneMinusCos;
    const float factor = sq * sq * oneMinusCos;
    return f0 + (Vec3{1.0f, 1.0f, 1.0f} - f0) * factor;
}

Vec3 evaluateCookTorranceBRDF(
    const PbrSurface& surface,
    const Vec3& viewDir,
    const Vec3& lightDir,
    const Vec3& radiance
) {
    const Vec3 N = normalize(surface.normal);
    const Vec3 V = normalize(viewDir);
    const Vec3 L = normalize(lightDir);

    const float nDotL = std::max(dot(N, L), 0.0f);
    const float nDotV = std::max(dot(N, V), 1e-4f);
    if (nDotL <= 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    const Vec3 H = normalize(V + L);
    const float nDotH = std::max(dot(N, H), 0.0f);
    const float hDotV = std::max(dot(H, V), 0.0f);

    const float metallic = std::clamp(surface.metallic, 0.0f, 1.0f);
    const float roughness = std::clamp(surface.roughness, 0.045f, 1.0f);
    const Vec3 f0 = lerp(Vec3{0.04f, 0.04f, 0.04f}, surface.albedo, metallic);

    const float ndf = distributionGGX(nDotH, roughness);
    const float vis = geometrySmith(nDotV, nDotL, roughness);
    const Vec3 F = fresnelSchlick(hDotV, f0);

    const Vec3 specular = (F * (ndf * vis)) / std::max(4.0f * nDotV * nDotL, 1e-5f);
    const Vec3 kD = (Vec3{1.0f, 1.0f, 1.0f} - F) * (1.0f - metallic);
    const Vec3 diffuse = kD * (surface.albedo / kPi);

    return (diffuse + specular) * radiance * nDotL;
}

float pointLightAttenuation(float distanceToLight, float radius) {
    if (distanceToLight >= radius || radius <= 0.0f) {
        return 0.0f;
    }
    const float ratio = distanceToLight / radius;
    const float ratio2 = ratio * ratio;
    const float window = std::clamp(1.0f - ratio2 * ratio2, 0.0f, 1.0f);
    return (window * window) / (distanceToLight * distanceToLight + 1.0f);
}

Vec3 acesFilmicToneMap(const Vec3& hdr) {
    auto mapChannel = [](float x) -> float {
        constexpr float a = 2.51f;
        constexpr float b = 0.03f;
        constexpr float c = 2.43f;
        constexpr float d = 0.59f;
        constexpr float e = 0.14f;
        const float num = x * (a * x + b);
        const float den = x * (c * x + d) + e;
        return std::clamp(num / den, 0.0f, 1.0f);
    };

    return {mapChannel(hdr.x), mapChannel(hdr.y), mapChannel(hdr.z)};
}

} // namespace glengine
