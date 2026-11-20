#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace glengine {

struct PassTimingStats {
    std::uint64_t sampleCount = 0;
    double lastMs = 0.0;
    double emaMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
    double p50Ms = 0.0;
    double p95Ms = 0.0;
    double p99Ms = 0.0;
};

// Headless rolling-window timing statistics aggregator with EMA smoothing and percentiles.
class TimingAccumulator {
public:
    explicit TimingAccumulator(double emaAlpha = 0.15, std::size_t windowSize = 256);

    void recordSampleMs(double ms);
    void recordSampleNs(std::uint64_t nanoseconds);
    PassTimingStats stats() const;
    void reset();

private:
    double emaAlpha_ = 0.15;
    std::size_t windowSize_ = 256;

    std::uint64_t sampleCount_ = 0;
    double lastMs_ = 0.0;
    double emaMs_ = 0.0;
    double minMs_ = 0.0;
    double maxMs_ = 0.0;

    std::vector<double> ringBuffer_;
    std::size_t writeCursor_ = 0;
};

// Backend hooks for OpenGL 4.6 GL_TIME_ELAPSED queries (injectable for headless testing).
struct GlTimerBackend {
    std::function<void(std::uint32_t count, std::uint32_t* ids)> createQueries;
    std::function<void(std::uint32_t count, const std::uint32_t* ids)> deleteQueries;
    std::function<void(std::uint32_t queryId)> beginTimeElapsed;
    std::function<void()> endTimeElapsed;
    std::function<bool(std::uint32_t queryId, std::uint64_t& outNanoseconds)> pollResultNs;
};

// Non-stalling ring-buffered GL_TIME_ELAPSED profiler across multiple frames in flight.
class GpuTimerQueryRing {
public:
    explicit GpuTimerQueryRing(GlTimerBackend backend, std::size_t ringDepth = 4);
    ~GpuTimerQueryRing();

    GpuTimerQueryRing(const GpuTimerQueryRing&) = delete;
    GpuTimerQueryRing& operator=(const GpuTimerQueryRing&) = delete;

    std::size_t registerPass(std::string name, double emaAlpha = 0.15, std::size_t windowSize = 256);

    void beginPass(std::size_t passId);
    void endPass(std::size_t passId);

    // Polls completed queries without blocking the GPU pipeline; returns number of collected samples.
    std::size_t collectCompleted();

    const std::string& passName(std::size_t passId) const;
    PassTimingStats passStats(std::size_t passId) const;
    std::size_t passCount() const noexcept { return passes_.size(); }

private:
    struct QuerySlot {
        std::uint32_t queryId = 0;
        bool inFlight = false;
    };

    struct PassEntry {
        std::string name;
        std::vector<QuerySlot> slots;
        std::size_t nextSlot = 0;
        bool active = false;
        TimingAccumulator accumulator;
    };

    GlTimerBackend backend_;
    std::size_t ringDepth_ = 4;
    std::vector<PassEntry> passes_;
};

} // namespace glengine
