#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>

namespace lsfg_android {

struct AnimeSignature {
    static constexpr uint32_t kColumns = 32;
    static constexpr uint32_t kRows = 24;
    using Rgb = std::array<uint8_t, 3>;
    std::array<Rgb, kColumns * kRows> cells{};
};

// These samples classify captures. They are never used as interpolation input.
inline AnimeSignature sampleAnimeFrame(const uint8_t *pixels, uint32_t width,
                                       uint32_t height, size_t strideBytes) {
    AnimeSignature result;
    for (uint32_t y = 0; y < AnimeSignature::kRows; ++y) {
        for (uint32_t x = 0; x < AnimeSignature::kColumns; ++x) {
            unsigned sum[3]{};
            for (uint32_t dy : {1u, 3u}) {
                const auto row = (4 * y + dy) * height / (4 * AnimeSignature::kRows);
                for (uint32_t dx : {1u, 3u}) {
                    const auto col = (4 * x + dx) * width / (4 * AnimeSignature::kColumns);
                    const auto *p = pixels + row * strideBytes + col * 4;
                    for (size_t channel = 0; channel < 3; ++channel) sum[channel] += p[channel];
                }
            }
            auto &cell = result.cells[y * AnimeSignature::kColumns + x];
            for (size_t channel = 0; channel < 3; ++channel)
                cell[channel] = static_cast<uint8_t>((sum[channel] + 2) / 4);
        }
    }
    return result;
}

struct AnimeFrameDecision {
    bool duplicate = false;
    bool breakBefore = false;
    bool allowHeldGap = false;
};

class AnimeCadence {
public:
    AnimeFrameDecision observe(const std::optional<AnimeSignature> &frame, int64_t timestampNs) {
        if (!frame || timestampNs <= 0) {
            reset();
            return {};
        }
        // Ignore comparisons across seeking, long capture stalls and reversed clocks.
        if (!anchor_ || timestampNs <= lastSeenNs_ || timestampNs - lastSeenNs_ > 160'000'000) {
            anchor_ = *frame;
            anchorNs_ = lastSeenNs_ = timestampNs;
            lastHoldNs_ = 0;
            return {.breakBefore = true};
        }

        lastSeenNs_ = timestampNs;
        unsigned sum = 0;
        unsigned changed = 0;
        unsigned stronglyChanged = 0;
        for (size_t i = 0; i < frame->cells.size(); ++i) {
            const auto &a = anchor_->cells[i];
            const auto &b = frame->cells[i];
            const auto difference = std::max({std::abs(a[0] - b[0]),
                                              std::abs(a[1] - b[1]),
                                              std::abs(a[2] - b[2])});
            sum += difference;
            changed += difference >= 12;
            stronglyChanged += difference >= 30;
        }

        const size_t count = frame->cells.size();
        // Small codec noise is fine, but changed line art in a few cells is motion.
        if (sum <= 2 * count && changed < 3) {
            lastHoldNs_ = timestampNs;
            return {.duplicate = true};
        }
        const bool cut = sum >= 40 * count && stronglyChanged * 3 >= 2 * count;
        const bool held = lastHoldNs_ > anchorNs_ &&
                          lastHoldNs_ - anchorNs_ >= 30'000'000 &&
                          timestampNs - lastHoldNs_ <= 50'000'000;
        anchor_ = *frame;
        anchorNs_ = timestampNs;
        lastHoldNs_ = 0;
        return {.breakBefore = cut, .allowHeldGap = held && !cut};
    }

    void reset() {
        anchor_.reset();
        anchorNs_ = lastSeenNs_ = lastHoldNs_ = 0;
    }

private:
    std::optional<AnimeSignature> anchor_;
    int64_t anchorNs_ = 0;
    int64_t lastSeenNs_ = 0;
    int64_t lastHoldNs_ = 0;
};

inline bool canInterpolateAnimePair(int64_t gapNs, bool allowHeldGap) {
    return gapNs > 0 && gapNs <= (allowHeldGap ? 160'000'000 : 100'000'000);
}

struct AnimeTimelineSlot {
    int64_t midpointNs;
    int64_t latestMidpointNs;
    int64_t realNs;
};

class AnimeTimeline {
public:
    AnimeTimelineSlot schedule(int64_t captureNs, int64_t observedIntervalNs,
                              int64_t refreshNs, int64_t refreshAnchorNs) {
        constexpr int64_t playoutNs = 52'000'000;
        constexpr int64_t correctionNs = 2'000'000;
        const int64_t naturalReal = captureNs + playoutNs;
        if (lastCaptureNs_ == 0 || captureNs <= lastCaptureNs_ ||
            captureNs - lastCaptureNs_ > 160'000'000 || refreshNs != refreshNs_) {
            intervalNs_ = observedIntervalNs;
            lastRealNs_ = naturalReal - intervalNs_;
        }
        const int64_t previousReal = lastRealNs_;
        intervalNs_ += (observedIntervalNs - intervalNs_) / 8;
        const int64_t predictedReal = previousReal + intervalNs_;
        const int64_t correction = std::clamp(naturalReal - predictedReal,
                                              -correctionNs, correctionNs);
        const int64_t real = std::clamp(predictedReal + correction,
                                       captureNs + 44'000'000, captureNs + 64'000'000);
        lastCaptureNs_ = captureNs;
        lastRealNs_ = real;
        refreshNs_ = refreshNs;

        // Aim at the nearest physical refresh when the driver has reported a
        // phase. Use a small lead so a timestamp is eligible for that refresh.
        const auto snap = [refreshNs, refreshAnchorNs](int64_t target) {
            if (refreshAnchorNs <= 0 || refreshNs <= 0) return target;
            const int64_t elapsed = target - refreshAnchorNs;
            if (elapsed < 0) return target;
            return refreshAnchorNs + ((elapsed + refreshNs / 2) / refreshNs) * refreshNs
                   - 750'000;
        };
        const int64_t realSlot = snap(real);
        const int64_t latest = realSlot - refreshNs - 1'000'000;
        const int64_t midpoint = snap(previousReal + (real - previousReal) / 2);
        return {std::min(midpoint, latest), latest, realSlot};
    }

    void reset() { lastCaptureNs_ = 0; }

private:
    int64_t lastCaptureNs_ = 0;
    int64_t lastRealNs_ = 0;
    int64_t intervalNs_ = 0;
    int64_t refreshNs_ = 0;
};

struct AnimeGridFrame {
    int64_t presentationNs = 0;
    float phase = 0.0f;
    // Grid ticks within 10% of an endpoint (~4 ms on 24 fps) reuse the crisp
    // real drawing directly, saving a full GPU interpolation pass with zero
    // visible cadence error.
    bool isReal() const { return phase >= 0.90f; }
    bool isPreviousReal() const { return phase <= 0.10f; }
};

struct AnimeGridPlan {
    static constexpr size_t kMaxFrames = 16;
    std::array<AnimeGridFrame, kMaxFrames> frames{};
    size_t count = 0;
};

// Evenly-spaced display-refresh-locked anime output grid across on-1s (~41.7 ms),
// on-2s (~83.3 ms), and on-3s (~125 ms) holds:
//   - multiplier == 2: 120 Hz -> 60 fps, 90 Hz -> 45 fps, 60 Hz -> 60 fps
//   - multiplier >= 3: full refresh rate (90 Hz -> 90 fps, 120 Hz -> 120 fps)
// ZeroFG 1.2.0 reuses the optical-flow pyramid across all phases of the same
// pair, keeping multi-phase 90/120 fps generation fast and judder-free.
class AnimeOutputGrid {
public:
    static constexpr int64_t kDelayNs = 85'000'000;

    AnimeGridPlan schedule(int64_t previousNs, int64_t currentNs, int64_t refreshNs,
                           int64_t displayAnchorNs, bool allowHeldGap,
                           int targetMultiplier = 2) {
        AnimeGridPlan plan;
        const auto delta = currentNs - previousNs;
        const int64_t maxGapNs = allowHeldGap ? 160'000'000 : 100'000'000;
        if (previousNs <= 0 || delta < 22'000'000 || delta > maxGapNs) {
            reset();
            return plan;
        }
        if (refreshNs < 5'000'000 || refreshNs > 20'000'000) {
            refreshNs = 16'666'667;
        }
        // 2x: 120 Hz -> 60 fps, 90 Hz -> 45 fps, 60 Hz -> 60 fps.
        // 3x/4x: unlock full refresh rate (90 Hz -> 90 fps, 120 Hz -> 120 fps).
        const int64_t divisor = (targetMultiplier >= 3)
            ? 1
            : std::max<int64_t>(1, (16'500'000 + refreshNs - 1) / refreshNs);
        const int64_t period = refreshNs * divisor;
        const auto first = previousNs + kDelayNs;
        const auto last = currentNs + kDelayNs;
        if (period != periodNs_ || nextNs_ == 0 || previousNs != lastCaptureNs_) {
            const auto anchor = displayAnchorNs > 0 ? displayAnchorNs : first;
            const auto offset = first - anchor;
            const auto residue = ((offset % period) + period) % period;
            nextNs_ = first + period - residue;
            periodNs_ = period;
        }
        lastCaptureNs_ = currentNs;
        while (nextNs_ <= first) nextNs_ += period;
        while (nextNs_ <= last && plan.count < plan.frames.size()) {
            const float rawPhase = static_cast<float>(nextNs_ - first) / static_cast<float>(delta);
            const float clampedPhase = std::clamp(rawPhase, 0.0f, 1.0f);
            plan.frames[plan.count++] = {nextNs_, clampedPhase};
            nextNs_ += period;
        }
        return plan;
    }

    void reset() { nextNs_ = 0; lastCaptureNs_ = 0; }

private:
    int64_t nextNs_ = 0;
    int64_t lastCaptureNs_ = 0;
    int64_t periodNs_ = 0;
};

} // namespace lsfg_android
