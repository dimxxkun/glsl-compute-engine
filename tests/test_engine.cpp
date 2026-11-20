#include "glengine/gpu_profiler.hpp"
#include "glengine/math.hpp"
#include "glengine/reference_algorithms.hpp"
#include "glengine/shader_preprocessor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

int gFailures = 0;

void expectTrue(bool condition, const char* expr, int line) {
    if (!condition) {
        std::cerr << "[FAIL] line " << line << ": " << expr << '\n';
        ++gFailures;
    }
}

void expectNear(double actual, double expected, double eps, const char* expr, int line) {
    if (std::fabs(actual - expected) > eps) {
        std::cerr << "[FAIL] line " << line << ": " << expr
                  << " (actual=" << actual << ", expected=" << expected
                  << ", eps=" << eps << ")\n";
        ++gFailures;
    }
}

#define EXPECT_TRUE(cond) expectTrue((cond), #cond, __LINE__)
#define EXPECT_NEAR(a, b, eps) expectNear((a), (b), (eps), #a " ~= " #b, __LINE__)

std::filesystem::path shaderRootPath() {
#ifdef GLENGINE_SHADER_DIR
    return std::filesystem::path(GLENGINE_SHADER_DIR);
#else
    return std::filesystem::path("shaders");
#endif
}

// =============================================================================
// 1. std140 / std430 Struct Layout Parity
// =============================================================================
void testGpuStructLayouts() {
    using namespace glengine;

    EXPECT_TRUE(sizeof(GpuParticle) == 32);
    EXPECT_TRUE(alignof(GpuParticle) == 16);
    EXPECT_TRUE(offsetof(GpuParticle, positionAge) == 0);
    EXPECT_TRUE(offsetof(GpuParticle, velocityMass) == 16);

    EXPECT_TRUE(sizeof(GpuPointLight) == 32);
    EXPECT_TRUE(alignof(GpuPointLight) == 16);
    EXPECT_TRUE(offsetof(GpuPointLight, positionRadius) == 0);
    EXPECT_TRUE(offsetof(GpuPointLight, colorIntensity) == 16);

    EXPECT_TRUE(sizeof(ObjectBounds) == 48);
    EXPECT_TRUE(alignof(ObjectBounds) == 16);
    EXPECT_TRUE(offsetof(ObjectBounds, aabbMin) == 0);
    EXPECT_TRUE(offsetof(ObjectBounds, aabbMax) == 16);
    EXPECT_TRUE(offsetof(ObjectBounds, meta) == 32);

    EXPECT_TRUE(sizeof(DrawElementsIndirectCommand) == 20);
    EXPECT_TRUE(offsetof(DrawElementsIndirectCommand, count) == 0);
    EXPECT_TRUE(offsetof(DrawElementsIndirectCommand, instanceCount) == 4);
    EXPECT_TRUE(offsetof(DrawElementsIndirectCommand, firstIndex) == 8);
    EXPECT_TRUE(offsetof(DrawElementsIndirectCommand, baseVertex) == 12);
    EXPECT_TRUE(offsetof(DrawElementsIndirectCommand, baseInstance) == 16);

    EXPECT_TRUE(sizeof(GpuSortEntry) == 8);
    EXPECT_TRUE(offsetof(GpuSortEntry, depthKey) == 0);
    EXPECT_TRUE(offsetof(GpuSortEntry, payloadIndex) == 4);

    EXPECT_TRUE(sizeof(CameraFrustumUbo) == 112);
    EXPECT_TRUE(offsetof(CameraFrustumUbo, planes) == 0);
    EXPECT_TRUE(offsetof(CameraFrustumUbo, cullParams) == 96);

    EXPECT_TRUE(sizeof(CurlSimParamsUbo) == 80);
    EXPECT_TRUE(sizeof(BitonicSortParamsUbo) == 16);
    EXPECT_TRUE(sizeof(TileCullUbo) == 160);
    EXPECT_TRUE(sizeof(InstanceDataGpu) == 176);
}

// =============================================================================
// 2. Frustum Extraction, AABB Culling & Indirect Command Compaction
// =============================================================================
void testFrustumCullingAndCompaction() {
    using namespace glengine;

    const Mat4 view = Mat4::lookAt({0.0f, 0.0f, 10.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    const Mat4 proj = Mat4::perspective(90.0f * kPi / 180.0f, 1.0f, 1.0f, 50.0f);
    const Frustum frustum = Frustum::fromViewProjection(proj * view);

    // Every extracted frustum plane normal must be unit length.
    for (const Vec4& plane : frustum.planes) {
        EXPECT_NEAR(length(plane.xyz()), 1.0, 1e-5);
    }

    // Center of frustum (world origin, distance 10 in front of camera) -> visible.
    EXPECT_TRUE(frustum.intersectsAABB({{-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}}));
    // Behind camera (z = 15, camera at z = 10 looking toward -Z) -> culled.
    EXPECT_TRUE(!frustum.intersectsAABB({{-1.0f, -1.0f, 14.0f}, {1.0f, 1.0f, 16.0f}}));
    // Closer than near plane (zNear = 1 -> plane at z = 9; box at z in [9.2, 9.8]) -> culled.
    EXPECT_TRUE(!frustum.intersectsAABB({{-0.2f, -0.2f, 9.2f}, {0.2f, 0.2f, 9.8f}}));
    // Beyond far plane (zFar = 50 -> plane at z = -40; box at z in [-55, -45]) -> culled.
    EXPECT_TRUE(!frustum.intersectsAABB({{-1.0f, -1.0f, -55.0f}, {1.0f, 1.0f, -45.0f}}));
    // Far outside right plane at z = 0 (half-width at d=10 with 90deg fov is 10) -> culled.
    EXPECT_TRUE(!frustum.intersectsAABB({{15.0f, -1.0f, -1.0f}, {17.0f, 1.0f, 1.0f}}));
    // Straddling the right plane (x in [9, 12] at z = 0) -> visible.
    EXPECT_TRUE(frustum.intersectsAABB({{9.0f, -1.0f, -1.0f}, {12.0f, 1.0f, 1.0f}}));

    // Multi-draw indirect compaction across 2 distinct draw commands.
    std::vector<ObjectBounds> objects = {
        // Draw 0: visible (id = 10)
        {Vec4{-1.0f, -1.0f, -1.0f, 1.73f}, Vec4{1.0f, 1.0f, 1.0f, 0.0f}, UVec4{0u, 10u, 0u, 1u}},
        // Draw 0: behind camera (culled)
        {Vec4{-1.0f, -1.0f, 20.0f, 1.73f}, Vec4{1.0f, 1.0f, 22.0f, 0.0f}, UVec4{0u, 11u, 0u, 1u}},
        // Draw 0: disabled flag (meta.w = 0 -> culled)
        {Vec4{-1.0f, -1.0f, -1.0f, 1.73f}, Vec4{1.0f, 1.0f, 1.0f, 0.0f}, UVec4{0u, 12u, 0u, 0u}},
        // Draw 0: visible (id = 13)
        {Vec4{2.0f, 1.0f, -5.0f, 1.0f}, Vec4{3.0f, 2.0f, -4.0f, 0.0f}, UVec4{0u, 13u, 0u, 1u}},
        // Draw 1: visible (id = 20)
        {Vec4{-3.0f, -2.0f, -10.0f, 2.0f}, Vec4{-1.0f, 0.0f, -8.0f, 0.0f}, UVec4{1u, 20u, 1u, 1u}},
        // Draw 1: outside left plane (culled)
        {Vec4{-30.0f, -1.0f, -1.0f, 1.0f}, Vec4{-25.0f, 1.0f, 1.0f, 0.0f}, UVec4{1u, 21u, 1u, 1u}}
    };

    std::vector<DrawElementsIndirectCommand> commands = {
        {36u, 99u, 0u, 0, 0u},    // pre-allocated slice [0..3]
        {72u, 99u, 36u, 24, 4u}   // pre-allocated slice [4..7]
    };

    std::vector<std::uint32_t> visibleIds(8, 0xFFFFFFFFu);
    const std::uint32_t totalVisible =
        frustumCullAndCompact(frustum, objects, commands, visibleIds);

    EXPECT_TRUE(totalVisible == 3u);
    EXPECT_TRUE(commands[0].instanceCount == 2u);
    EXPECT_TRUE(commands[1].instanceCount == 1u);
    EXPECT_TRUE(visibleIds[0] == 10u);
    EXPECT_TRUE(visibleIds[1] == 13u);
    EXPECT_TRUE(visibleIds[4] == 20u);
}

// =============================================================================
// 3. Bitonic Sort Correctness (Random, Reverse, Duplicates, Descending, Errors)
// =============================================================================
void testBitonicSort() {
    using namespace glengine;

    // 3a. Random 1024-element array ascending
    {
        std::mt19937 rng(42u);
        std::uniform_real_distribution<float> dist(-500.0f, 500.0f);

        std::vector<GpuSortEntry> entries(1024);
        for (std::uint32_t i = 0; i < 1024u; ++i) {
            entries[i] = {dist(rng), i};
        }

        bitonicSortCpu(entries, true);

        for (std::size_t i = 1; i < entries.size(); ++i) {
            const bool ordered =
                (entries[i - 1].depthKey < entries[i].depthKey) ||
                (entries[i - 1].depthKey == entries[i].depthKey &&
                 entries[i - 1].payloadIndex <= entries[i].payloadIndex);
            EXPECT_TRUE(ordered);
        }
    }

    // 3b. Reverse-sorted 256-element array ascending
    {
        std::vector<GpuSortEntry> entries(256);
        for (std::uint32_t i = 0; i < 256u; ++i) {
            entries[i] = {static_cast<float>(255u - i), i};
        }
        bitonicSortCpu(entries, true);
        for (std::uint32_t i = 0; i < 256u; ++i) {
            EXPECT_NEAR(entries[i].depthKey, static_cast<double>(i), 1e-6);
            EXPECT_TRUE(entries[i].payloadIndex == (255u - i));
        }
    }

    // 3c. Heavy duplicate keys with deterministic payload tie-breaking + descending order
    {
        std::vector<GpuSortEntry> entries(64);
        for (std::uint32_t i = 0; i < 64u; ++i) {
            entries[i] = {static_cast<float>(i % 4u), 63u - i};
        }
        bitonicSortCpu(entries, true);
        for (std::size_t i = 1; i < entries.size(); ++i) {
            const bool ordered =
                (entries[i - 1].depthKey < entries[i].depthKey) ||
                (entries[i - 1].depthKey == entries[i].depthKey &&
                 entries[i - 1].payloadIndex < entries[i].payloadIndex);
            EXPECT_TRUE(ordered);
        }

        bitonicSortCpu(entries, false);
        for (std::size_t i = 1; i < entries.size(); ++i) {
            const bool orderedDesc =
                (entries[i - 1].depthKey > entries[i].depthKey) ||
                (entries[i - 1].depthKey == entries[i].depthKey &&
                 entries[i - 1].payloadIndex > entries[i].payloadIndex);
            EXPECT_TRUE(orderedDesc);
        }
    }

    // 3d. Non-power-of-two must throw std::invalid_argument
    {
        bool threw = false;
        std::vector<GpuSortEntry> badSize(10);
        try {
            bitonicSortCpu(badSize, true);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        EXPECT_TRUE(threw);
    }
}

// =============================================================================
// 4. Cook-Torrance PBR BRDF Energy Conservation & Fresnel Limits
// =============================================================================
void testCookTorrancePbrBrdf() {
    using namespace glengine;

    // 4a. Fresnel limits: F(0 deg -> cosTheta=1) == F0, F(90 deg -> cosTheta=0) == (1, 1, 1)
    const Vec3 f0Dielectric{0.04f, 0.04f, 0.04f};
    const Vec3 fAtNormal = fresnelSchlick(1.0f, f0Dielectric);
    const Vec3 fAtGrazing = fresnelSchlick(0.0f, f0Dielectric);

    EXPECT_NEAR(fAtNormal.x, 0.04, 1e-6);
    EXPECT_NEAR(fAtNormal.y, 0.04, 1e-6);
    EXPECT_NEAR(fAtNormal.z, 0.04, 1e-6);

    EXPECT_NEAR(fAtGrazing.x, 1.0, 1e-6);
    EXPECT_NEAR(fAtGrazing.y, 1.0, 1e-6);
    EXPECT_NEAR(fAtGrazing.z, 1.0, 1e-6);

    const Vec3 f0Gold{1.0f, 0.766f, 0.336f};
    const Vec3 fGoldNormal = fresnelSchlick(1.0f, f0Gold);
    EXPECT_NEAR(fGoldNormal.x, 1.0, 1e-6);
    EXPECT_NEAR(fGoldNormal.y, 0.766, 1e-6);
    EXPECT_NEAR(fGoldNormal.z, 0.336, 1e-6);

    // 4b. GGX & Smith geometry sanity & bounds
    const float ndfSmooth = distributionGGX(1.0f, 0.1f);
    const float ndfRough = distributionGGX(1.0f, 0.8f);
    EXPECT_TRUE(ndfSmooth > ndfRough);
    EXPECT_TRUE(ndfRough > 0.0f);

    const float gVal = geometrySmith(0.7f, 0.8f, 0.5f);
    EXPECT_TRUE(gVal > 0.0f && gVal <= 1.0f);

    // 4c. Energy conservation check: for a pure white matte dielectric under uniform radiance
    // integrated over the upper hemisphere, total outgoing radiance must not exceed incident radiance.
    const PbrSurface whiteMatte{
        Vec3{1.0f, 1.0f, 1.0f},
        0.0f,
        1.0f,
        Vec3{0.0f, 1.0f, 0.0f}
    };
    const Vec3 viewDir{0.0f, 1.0f, 0.0f};
    const Vec3 unitRadiance{1.0f, 1.0f, 1.0f};

    // Numerical polar integration over theta in [0, pi/2] (azimuthally symmetric at normal incidence).
    constexpr int kSteps = 512;
    const float dTheta = (0.5f * kPi) / static_cast<float>(kSteps);
    double hemisphereIntegral = 0.0;
    for (int i = 0; i < kSteps; ++i) {
        const float theta = (static_cast<float>(i) + 0.5f) * dTheta;
        const Vec3 lightDir{std::sin(theta), std::cos(theta), 0.0f};
        const Vec3 lo = evaluateCookTorranceBRDF(whiteMatte, viewDir, lightDir, unitRadiance);
        // Solid angle element dOmega = 2 * pi * sin(theta) * dTheta
        hemisphereIntegral += static_cast<double>(lo.x) * (2.0 * static_cast<double>(kPi) *
                              std::sin(static_cast<double>(theta)) * static_cast<double>(dTheta));
    }
    EXPECT_TRUE(hemisphereIntegral > 0.5 && hemisphereIntegral <= 1.01);

    // 4d. Back-facing light yields strictly zero radiance.
    const Vec3 backLit = evaluateCookTorranceBRDF(
        whiteMatte, viewDir, Vec3{0.0f, -1.0f, 0.0f}, Vec3{5.0f, 5.0f, 5.0f}
    );
    EXPECT_NEAR(backLit.x, 0.0, 1e-6);
    EXPECT_NEAR(backLit.y, 0.0, 1e-6);
    EXPECT_NEAR(backLit.z, 0.0, 1e-6);

    // 4e. Karis point light attenuation & ACES filmic tone map bounds
    EXPECT_TRUE(pointLightAttenuation(0.0f, 10.0f) > 0.99f);
    EXPECT_NEAR(pointLightAttenuation(10.0f, 10.0f), 0.0, 1e-6);
    EXPECT_NEAR(pointLightAttenuation(15.0f, 10.0f), 0.0, 1e-6);

    const Vec3 mappedLow = acesFilmicToneMap({0.2f, 0.2f, 0.2f});
    const Vec3 mappedHigh = acesFilmicToneMap({20.0f, 20.0f, 20.0f});
    EXPECT_TRUE(mappedLow.x > 0.0f && mappedLow.x < mappedHigh.x);
    EXPECT_TRUE(mappedHigh.x <= 1.0f);
}

// =============================================================================
// 5. Shader Preprocessor (#include, #define injection, Real Shaders)
// =============================================================================
void testShaderPreprocessor() {
    using namespace glengine;

    const std::filesystem::path root = shaderRootPath();
    ShaderPreprocessor preprocessor;

    // 5a. #include resolution + #define injection after #version 460 core
    {
        const std::string syntheticShader =
            "#version 460 core\n"
            "#include \"common/pbr_brdf.glsl\"\n"
            "layout(location = 0) out vec4 oColor;\n"
            "void main() {\n"
            "    vec3 f = glengineFresnelSchlick(0.5, vec3(0.04));\n"
            "    oColor = vec4(f * float(TEST_SCALE), 1.0);\n"
            "}\n";

        PreprocessOptions opts{};
        opts.includeDirs.push_back(root);
        opts.defines.emplace_back("TEST_SCALE", "2");
        opts.defines.emplace_back("ENABLE_PBR_FAST_PATH", "");
        opts.emitLineDirectives = true;

        const PreprocessResult res =
            preprocessor.preprocessSource(syntheticShader, root / "virtual_test.frag", opts);

        EXPECT_TRUE(res.stage == ShaderStage::Fragment);
        EXPECT_TRUE(res.includedFiles.size() == 1);
        EXPECT_TRUE(res.sourceFileTable.size() == 2);
        EXPECT_TRUE(res.source.starts_with("#version 460 core\n"));
        EXPECT_TRUE(res.source.find("#define TEST_SCALE 2\n") != std::string::npos);
        EXPECT_TRUE(res.source.find("#define ENABLE_PBR_FAST_PATH\n") != std::string::npos);
        EXPECT_TRUE(res.source.find("glengineDistributionGGX") != std::string::npos);
        EXPECT_TRUE(res.source.find("#line 1 1\n") != std::string::npos);
    }

    // 5b. Missing include throws std::runtime_error
    {
        bool threw = false;
        try {
            (void)preprocessor.preprocessSource(
                "#version 460 core\n#include \"does_not_exist.glsl\"\n",
                root / "broken.comp",
                {}
            );
        } catch (const std::runtime_error&) {
            threw = true;
        }
        EXPECT_TRUE(threw);
    }

    // 5c. Load all 9 real project shaders from shaders/ and validate stage + version
    struct ExpectedShader {
        const char* relPath;
        ShaderStage expectedStage;
    };
    const std::array<ExpectedShader, 9> expectedShaders = {{
        {"compute/curl_particles.comp", ShaderStage::Compute},
        {"compute/frustum_cull.comp", ShaderStage::Compute},
        {"compute/bitonic_sort.comp", ShaderStage::Compute},
        {"compute/tiled_light_cull.comp", ShaderStage::Compute},
        {"graphics/pbr_gbuffer.vert", ShaderStage::Vertex},
        {"graphics/pbr_gbuffer.frag", ShaderStage::Fragment},
        {"graphics/pbr_lighting.vert", ShaderStage::Vertex},
        {"graphics/pbr_lighting.frag", ShaderStage::Fragment},
        {"graphics/raymarch_sdf.frag", ShaderStage::Fragment}
    }};

    PreprocessOptions realOpts{};
    realOpts.includeDirs.push_back(root);
    realOpts.emitLineDirectives = false;

    for (const auto& item : expectedShaders) {
        const PreprocessResult res = preprocessor.preprocessFile(root / item.relPath, realOpts);
        EXPECT_TRUE(res.stage == item.expectedStage);
        EXPECT_TRUE(res.source.starts_with("#version 460 core\n"));
        EXPECT_TRUE(res.source.find("void main()") != std::string::npos);
    }
}

// =============================================================================
// 6. GPU Profiler (EMA, Percentiles & Ring-Buffered GL_TIME_ELAPSED Mock)
// =============================================================================
void testGpuProfiler() {
    using namespace glengine;

    // 6a. TimingAccumulator EMA & percentiles
    {
        TimingAccumulator acc(0.25, 100);
        for (int i = 1; i <= 100; ++i) {
            acc.recordSampleMs(static_cast<double>(i));
        }

        const PassTimingStats s = acc.stats();
        EXPECT_TRUE(s.sampleCount == 100);
        EXPECT_NEAR(s.lastMs, 100.0, 1e-9);
        EXPECT_NEAR(s.minMs, 1.0, 1e-9);
        EXPECT_NEAR(s.maxMs, 100.0, 1e-9);
        EXPECT_NEAR(s.p50Ms, 50.5, 1e-6);
        EXPECT_NEAR(s.p95Ms, 95.05, 1e-6);
        EXPECT_NEAR(s.p99Ms, 99.01, 1e-6);

        // Verify exact closed-form EMA on a tiny 3-sample sequence.
        TimingAccumulator emaCheck(0.5, 16);
        emaCheck.recordSampleMs(10.0); // ema = 10.0
        emaCheck.recordSampleMs(20.0); // ema = 0.5*20 + 0.5*10 = 15.0
        emaCheck.recordSampleMs(30.0); // ema = 0.5*30 + 0.5*15 = 22.5
        EXPECT_NEAR(emaCheck.stats().emaMs, 22.5, 1e-9);
    }

    // 6b. GpuTimerQueryRing with mock OpenGL query lifecycle
    {
        std::uint32_t nextQueryId = 1;
        std::unordered_map<std::uint32_t, std::uint64_t> queryResultsNs;
        std::uint32_t activeQuery = 0;
        std::size_t deletedCount = 0;

        {
            GlTimerBackend mockBackend{};
            mockBackend.createQueries = [&](std::uint32_t count, std::uint32_t* ids) {
                for (std::uint32_t i = 0; i < count; ++i) {
                    ids[i] = nextQueryId++;
                }
            };
            mockBackend.deleteQueries = [&](std::uint32_t count, const std::uint32_t* ids) {
                (void)ids;
                deletedCount += count;
            };
            mockBackend.beginTimeElapsed = [&](std::uint32_t qid) {
                activeQuery = qid;
            };
            mockBackend.endTimeElapsed = [&]() {
                // Record deterministic synthetic GPU duration (1.5 ms = 1,500,000 ns)
                queryResultsNs[activeQuery] = 1'500'000ULL;
                activeQuery = 0;
            };
            mockBackend.pollResultNs = [&](std::uint32_t qid, std::uint64_t& outNs) -> bool {
                const auto it = queryResultsNs.find(qid);
                if (it == queryResultsNs.end()) {
                    return false;
                }
                outNs = it->second;
                queryResultsNs.erase(it);
                return true;
            };

            GpuTimerQueryRing ring(mockBackend, 4);
            const std::size_t cullPass = ring.registerPass("FrustumCull");
            EXPECT_TRUE(ring.passName(cullPass) == "FrustumCull");

            for (int frame = 0; frame < 6; ++frame) {
                ring.beginPass(cullPass);
                ring.endPass(cullPass);
                (void)ring.collectCompleted();
            }

            const PassTimingStats st = ring.passStats(cullPass);
            EXPECT_TRUE(st.sampleCount == 6);
            EXPECT_NEAR(st.emaMs, 1.5, 1e-6);
            EXPECT_NEAR(st.p50Ms, 1.5, 1e-6);
        }

        // Destructor must delete all 4 allocated query objects.
        EXPECT_TRUE(deletedCount == 4);
    }
}

} // namespace

int main() {
    testGpuStructLayouts();
    testFrustumCullingAndCompaction();
    testBitonicSort();
    testCookTorrancePbrBrdf();
    testShaderPreprocessor();
    testGpuProfiler();

    if (gFailures == 0) {
        std::cout << "[PASS] All glsl-compute-engine unit tests passed.\n";
        return 0;
    }

    std::cerr << "[FAIL] " << gFailures << " test assertion(s) failed.\n";
    return 1;
}
