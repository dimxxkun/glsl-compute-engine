#include "glengine/gpu_profiler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace glengine {

namespace {

double computePercentile(const std::vector<double>& sortedSamples, double quantile) {
    if (sortedSamples.empty()) {
        return 0.0;
    }
    if (sortedSamples.size() == 1) {
        return sortedSamples.front();
    }

    const double q = std::clamp(quantile, 0.0, 1.0);
    const double rank = q * static_cast<double>(sortedSamples.size() - 1);
    const std::size_t lowerIdx = static_cast<std::size_t>(std::floor(rank));
    const std::size_t upperIdx = std::min(lowerIdx + 1, sortedSamples.size() - 1);
    const double frac = rank - static_cast<double>(lowerIdx);

    return sortedSamples[lowerIdx] * (1.0 - frac) + sortedSamples[upperIdx] * frac;
}

} // namespace

TimingAccumulator::TimingAccumulator(double emaAlpha, std::size_t windowSize)
    : emaAlpha_(std::clamp(emaAlpha, 0.001, 1.0)),
      windowSize_(std::max<std::size_t>(windowSize, 1)) {
    ringBuffer_.reserve(windowSize_);
}

void TimingAccumulator::recordSampleMs(double ms) {
    if (ms < 0.0) {
        ms = 0.0;
    }

    if (sampleCount_ == 0) {
        emaMs_ = ms;
        minMs_ = ms;
        maxMs_ = ms;
    } else {
        emaMs_ = emaAlpha_ * ms + (1.0 - emaAlpha_) * emaMs_;
        minMs_ = std::min(minMs_, ms);
        maxMs_ = std::max(maxMs_, ms);
    }

    lastMs_ = ms;
    ++sampleCount_;

    if (ringBuffer_.size() < windowSize_) {
        ringBuffer_.push_back(ms);
    } else {
        ringBuffer_[writeCursor_] = ms;
        writeCursor_ = (writeCursor_ + 1) % windowSize_;
    }
}

void TimingAccumulator::recordSampleNs(std::uint64_t nanoseconds) {
    recordSampleMs(static_cast<double>(nanoseconds) * 1e-6);
}

PassTimingStats TimingAccumulator::stats() const {
    PassTimingStats result{};
    result.sampleCount = sampleCount_;
    result.lastMs = lastMs_;
    result.emaMs = emaMs_;
    result.minMs = minMs_;
    result.maxMs = maxMs_;

    if (!ringBuffer_.empty()) {
        std::vector<double> sorted = ringBuffer_;
        std::sort(sorted.begin(), sorted.end());
        result.p50Ms = computePercentile(sorted, 0.50);
        result.p95Ms = computePercentile(sorted, 0.95);
        result.p99Ms = computePercentile(sorted, 0.99);
    }

    return result;
}

void TimingAccumulator::reset() {
    sampleCount_ = 0;
    lastMs_ = 0.0;
    emaMs_ = 0.0;
    minMs_ = 0.0;
    maxMs_ = 0.0;
    ringBuffer_.clear();
    writeCursor_ = 0;
}

GpuTimerQueryRing::GpuTimerQueryRing(GlTimerBackend backend, std::size_t ringDepth)
    : backend_(std::move(backend)),
      ringDepth_(std::max<std::size_t>(ringDepth, 2)) {}

GpuTimerQueryRing::~GpuTimerQueryRing() {
    if (!backend_.deleteQueries) {
        return;
    }

    std::vector<std::uint32_t> ids;
    for (const auto& pass : passes_) {
        for (const auto& slot : pass.slots) {
            if (slot.queryId != 0) {
                ids.push_back(slot.queryId);
            }
        }
    }

    if (!ids.empty()) {
        backend_.deleteQueries(static_cast<std::uint32_t>(ids.size()), ids.data());
    }
}

std::size_t GpuTimerQueryRing::registerPass(
    std::string name,
    double emaAlpha,
    std::size_t windowSize
) {
    PassEntry entry{
        std::move(name),
        std::vector<QuerySlot>(ringDepth_),
        0,
        false,
        TimingAccumulator(emaAlpha, windowSize)
    };

    if (backend_.createQueries) {
        std::vector<std::uint32_t> created(ringDepth_, 0);
        backend_.createQueries(static_cast<std::uint32_t>(ringDepth_), created.data());
        for (std::size_t i = 0; i < ringDepth_; ++i) {
            entry.slots[i].queryId = created[i];
        }
    }

    passes_.push_back(std::move(entry));
    return passes_.size() - 1;
}

void GpuTimerQueryRing::beginPass(std::size_t passId) {
    if (passId >= passes_.size()) {
        throw std::out_of_range("GpuTimerQueryRing::beginPass invalid passId");
    }

    PassEntry& pass = passes_[passId];
    if (pass.active) {
        throw std::logic_error("Nested beginPass on the same GPU timer pass: " + pass.name);
    }

    QuerySlot& slot = pass.slots[pass.nextSlot];

    // If the oldest slot is still marked in-flight, poll it opportunistically before reusing.
    if (slot.inFlight && backend_.pollResultNs) {
        std::uint64_t elapsedNs = 0;
        if (backend_.pollResultNs(slot.queryId, elapsedNs)) {
            pass.accumulator.recordSampleNs(elapsedNs);
            slot.inFlight = false;
        }
    }

    if (!slot.inFlight && backend_.beginTimeElapsed && slot.queryId != 0) {
        backend_.beginTimeElapsed(slot.queryId);
        pass.active = true;
    }
}

void GpuTimerQueryRing::endPass(std::size_t passId) {
    if (passId >= passes_.size()) {
        throw std::out_of_range("GpuTimerQueryRing::endPass invalid passId");
    }

    PassEntry& pass = passes_[passId];
    if (!pass.active) {
        return;
    }

    if (backend_.endTimeElapsed) {
        backend_.endTimeElapsed();
    }

    pass.slots[pass.nextSlot].inFlight = true;
    pass.nextSlot = (pass.nextSlot + 1) % ringDepth_;
    pass.active = false;
}

std::size_t GpuTimerQueryRing::collectCompleted() {
    if (!backend_.pollResultNs) {
        return 0;
    }

    std::size_t collected = 0;
    for (auto& pass : passes_) {
        for (auto& slot : pass.slots) {
            if (!slot.inFlight || slot.queryId == 0) {
                continue;
            }
            std::uint64_t elapsedNs = 0;
            if (backend_.pollResultNs(slot.queryId, elapsedNs)) {
                pass.accumulator.recordSampleNs(elapsedNs);
                slot.inFlight = false;
                ++collected;
            }
        }
    }
    return collected;
}

const std::string& GpuTimerQueryRing::passName(std::size_t passId) const {
    if (passId >= passes_.size()) {
        throw std::out_of_range("GpuTimerQueryRing::passName invalid passId");
    }
    return passes_[passId].name;
}

PassTimingStats GpuTimerQueryRing::passStats(std::size_t passId) const {
    if (passId >= passes_.size()) {
        throw std::out_of_range("GpuTimerQueryRing::passStats invalid passId");
    }
    return passes_[passId].accumulator.stats();
}

} // namespace glengine
