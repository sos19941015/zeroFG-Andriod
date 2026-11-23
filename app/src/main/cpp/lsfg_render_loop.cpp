#include "lsfg_render_loop.hpp"
#include "anime_cadence.hpp"
#include "android_vk_session.hpp"
#include "ahb_image_bridge.hpp"
#include "gpu_postprocess.hpp"
#include "nnapi_npu.hpp"
#include "nnapi_postprocess.hpp"
#include "cpu_postprocess.hpp"


#include <volk.h>
#include <zerofg/zerofg.h>
#include <future>

#include <android/log.h>
#include <android/native_window.h>
#include <android/hardware_buffer.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <array>
#include <dlfcn.h>

#include "crash_reporter.hpp"

#define LOG_TAG "lsfg-vk-loop"
#define LOGE(...) ::lsfg_android::ring_logf(LOG_TAG, ANDROID_LOG_ERROR, __VA_ARGS__)
#define LOGW(...) ::lsfg_android::ring_logf(LOG_TAG, ANDROID_LOG_WARN,  __VA_ARGS__)
#define LOGI(...) ::lsfg_android::ring_logf(LOG_TAG, ANDROID_LOG_INFO,  __VA_ARGS__)

namespace lsfg_android {

namespace {

uint64_t getAhbId(AHardwareBuffer *ahb) {
    using GetId = int (*)(const AHardwareBuffer *, uint64_t *);
    static const auto getId = [] {
        void *lib = dlopen("libandroid.so", RTLD_NOW);
        return lib ? reinterpret_cast<GetId>(dlsym(lib, "AHardwareBuffer_getId")) : nullptr;
    }();
    uint64_t id = 0;
    return getId && getId(ahb, &id) == 0 ? id : 0;
}

struct State {
    using Clock = std::chrono::steady_clock;
    static constexpr size_t kInputSlotCount = 4;
    static constexpr size_t kOutputSlotCount = 8;

    std::mutex mu;
    std::mutex queueMu;          // serializes VkQueue submit/present/waitIdle only
    bool initialized = false;
    bool performanceMode = false;
    bool framegenFp16 = false;    // load IDs 304..351 (FP16 SPIR-V) instead of 255..302 (DXBC->SPIR-V)
    bool hdr = false;
    float flowScale = 1.0f;
    std::unique_ptr<zerofg::Interpolator> interpolator;
    VkImageView frameViews[kInputSlotCount + kOutputSlotCount]{};
    uint64_t inputSequences[kInputSlotCount]{};
    uint64_t nextSequence = 0;
    int multiplier = 2;          // generationCount
    std::atomic<int> targetMultiplier{2}; // total output factor (2, 3, or 4) for normal mode

    VulkanSession vk{};

    AhbImage inSlot[kInputSlotCount]{}; // ring of real-frame inputs
    uint64_t framesCopied = 0;   // total inputs we've copied into a slot

    // Vulkan swapchain state. When live, blitOutputToWindow takes the
    // GPU-only fast path: vkAcquireNextImageKHR → vkCmdBlitImage from the
    // framegen output AHB → vkQueuePresentKHR. This eliminates the CPU
    // memcpy (AHardwareBuffer_lock(CPU_READ) + ANativeWindow_lock +
    // per-row memcpy) that was the single biggest cost at multiplier ≥ 2.
    //
    // The WSI path is disabled when: the extension chain is missing
    // (hasSwapchain=false), any CPU post-process is active (NPU or CPU
    // filter — both mutate the destination in-place on CPU), or surface
    // creation failed on the current ANativeWindow. In those cases the
    // fallback CPU blit path is used transparently.
    struct SwapchainState {
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
        VkExtent2D extent{};
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::vector<VkImage> images;
        // One acquire semaphore per slot, cycled round-robin. Vulkan spec
        // forbids reusing a semaphore until the corresponding acquire has
        // completed; N+1 is the safe lower bound.
        std::vector<VkSemaphore> acquireSems;
        // One render-done semaphore per swapchain image (vkQueuePresentKHR
        // waits on the image's semaphore before presenting).
        std::vector<VkSemaphore> renderSems;
        uint32_t acquireCursor = 0;
        uint32_t presentId = 0;
        bool outOfDate = false;
        // Set when an attempt to build the swapchain failed (e.g. the compute
        // queue family cannot present, or the surface rejects TRANSFER_DST
        // usage). Once set we stop retrying for the rest of the session and
        // use the CPU blit path exclusively. Cleared by destroySwapchain()
        // so a surface re-attach gets a fresh attempt.
        bool disabledForSession = false;
    } swap;
    // ANativeWindow width/height the swapchain was built for. When the
    // overlay reshapes (rare, mostly on orientation change), we recreate.
    uint32_t swapWinW = 0;
    uint32_t swapWinH = 0;
    std::atomic<bool> bypass{false}; // skip framegen, blit raw input
    // Auto-bypass triggered when framegen returns VK_ERROR_DEVICE_LOST during
    // presentContext. Distinct from the user-controlled `bypass` so the user
    // toggle isn't silently flipped by a recoverable driver event. Cleared on
    // every initRenderLoop / context recreation — if the next session
    // succeeds, framegen runs again. Stuck-on means the next presentContext
    // would error too, so passthrough is the right behaviour anyway.
    std::atomic<bool> framegenAutoDisabled{false};
    std::atomic<bool> antiArtifacts{false};
    std::atomic<uint64_t> cacheHits{0};
    std::atomic<uint64_t> cacheMisses{0};
    bool npuPostProcessing = false;
    int npuPreset = 0;
    int npuUpscaleFactor = 1;
    float npuAmount = 0.5f;
    float npuRadius = 1.0f;
    float npuThreshold = 0.0f;
    bool npuFp16 = true;
    NnapiPostProcessor npuPost{};
    bool cpuPostProcessing = false;
    int cpuPreset = 0;
    float cpuStrength = 0.5f;
    float cpuSaturation = 0.5f;
    float cpuVibrance = 0.0f;
    float cpuVignette = 0.0f;
    CpuPostProcessor cpuPost{};
    std::atomic<bool> gpuPostProcessing{false};
    int gpuStage = 1;
    std::atomic<int> gpuMethod{0};
    float gpuUpscaleFactor = 1.0f;
    std::atomic<float> gpuSharpness{0.65f};
    std::atomic<float> gpuStrength{0.75f};
    std::atomic<bool> gpuNoopLogged{false};
    AhbImage gpuPostImage{};
    GpuPostProcessor gpuPost{};
    std::vector<AhbImage> outputs; // multiplier-many outputs

    // Output surface for the final blit. Owned (acquired from JNI).
    ANativeWindow *outWindow = nullptr;
    uint32_t outWidth = 0;
    uint32_t outHeight = 0;
    // Once we have produced a CPU buffer on this ANativeWindow (via
    // ANativeWindow_lock / setBuffersGeometry), the BufferQueue is bound to a
    // CPU producer. Mali (Bifrost/Valhall) and many other drivers will then
    // return VK_ERROR_NATIVE_WINDOW_IN_USE_KHR (-1000000001) for any subsequent
    // vkCreateAndroidSurfaceKHR on the same native window — the producer slot
    // can't be retargeted live. We track this taint per-attached window and
    // skip the WSI swapchain attempt entirely, avoiding the spammy retries seen
    // on Mali-G57 and the cascade where a failed surface creation correlates
    // with a DEVICE_LOST on framegen's compute queue. Reset on each
    // setOutputSurface(win) call, since a freshly-acquired ANativeWindow has
    // no producer yet.
    bool windowCpuProducerLocked = false;
    // Last geometry successfully passed to ANativeWindow_setBuffersGeometry.
    // We guard the call so it only fires when parameters actually change —
    // calling it on every blit triggers a SurfaceTexture buffer-queue
    // reconfiguration on some drivers that discards queued frames, leaving
    // the display frozen on the first posted frame.
    uint32_t winGeomW = 0;
    uint32_t winGeomH = 0;
    int      winGeomFmt = 0;
    // Feedback-loop gate: suppresses blits when framegen output luma is very
    // dark, indicating setSkipScreenshot failed and MediaProjection is capturing
    // the overlay instead of the game.  Keeping the overlay transparent lets the
    // next capture see the real game and breaks the loop.
    bool     lumaGateOpen      = false;
    uint32_t lumaGateDarkCount = 0;
    int64_t  lumaGateStartNs   = 0; // steady_clock ns at first suppressed dark frame



    // Pending frames to process. We acquire a ref on the AHB so it survives
    // beyond the caller's Image.close(). Drained by the worker thread.
    struct PendingFrame {
        AHardwareBuffer *ahb = nullptr;
        Clock::time_point queuedAt{};
        int64_t captureTimestampNs = 0;
        uint64_t sequence;
        bool animeMode;
        AnimeFrameDecision animeDecision;
        std::shared_ptr<std::promise<void>> copied;
    };
    std::deque<PendingFrame> pending;
    std::condition_variable pendingCv;
    bool stopRequested = false;

    // Decoupled presentation queue consumed by `presentWorker`. This ensures
    // pacing sleeps and swapchain present stalls NEVER block capture copies or
    // ZeroFG GPU compute.
    struct PresentItem {
        bool isGenerated = false;
        size_t slotIndex = 0;
        int64_t desiredPresentNs = 0; // 0 = present immediately
        int64_t latestPresentNs = 0;  // if > 0 and generated, skip when now > latestPresentNs
        int64_t captureTimestampNs = 0;
        bool useDisplayTiming = false;
        int64_t fallbackSleepBudgetNs = 0; // used for vsync-aligned sleep on non-anime path
    };
    std::mutex presentMu;
    std::condition_variable presentCv;
    std::deque<PresentItem> presentQueue;
    std::array<uint32_t, kInputSlotCount> inSlotRefCount{};
    std::array<uint32_t, kOutputSlotCount> outSlotRefCount{};
    bool presentStopRequested = false;

    std::thread worker;
    std::thread presentWorker;
    std::atomic<uint64_t> generatedFrames{0};
    std::atomic<uint64_t> lateMidpointsTotal{0};
    std::atomic<uint64_t> presentBlitNsAccum{0};
    std::atomic<uint64_t> presentCaptureToDisplayNsAccum{0};
    std::atomic<uint64_t> presentBlitCountAccum{0};
    std::atomic<uint64_t> lastActualPresentNs{0};
    // Counts every successful post (CPU blit or WSI present) to the overlay.
    // This is the ground-truth "frames on screen" metric — includes real
    // captures AND LSFG-generated frames. Used by the HUD total-fps counter
    // instead of the old `capturedFps + genFps` which double-counted.
    std::atomic<uint64_t> postedFrames{0};
    // Counts capture frames whose pixel content actually differs from the
    // previous capture. MediaProjection delivers at the display refresh rate
    // (often 120 Hz) regardless of the target app's render rate — most of
    // those captures are pixel-identical duplicates. Detecting uniqueness via
    // an 8×8 luma hash gives the target app's TRUE render rate.
    std::atomic<uint64_t> uniqueCaptures{0};
    // Protected by captureHashMu. The worker/capture side updates `lastCaptureHash`;
    // the mutex is cheap because only pushFrame (called from one thread —
    // the ImageReader listener) writes. Kept as a plain struct so the reader
    // can sample without locking if needed.
    std::mutex captureHashMu;
    uint32_t lastCaptureHash = 0;
    bool lastCaptureHashValid = false;
    bool animeMode = false;  // protected by captureHashMu
    AnimeCadence animeCadence;
    std::atomic<uint64_t> animeDuplicates{0};

    // Ring buffer of recent post timestamps (ns from CLOCK_MONOTONIC, via
    // steady_clock). Consumed by the HUD frame-pacing graph to show real
    // frame-to-frame intervals instead of rolling counts.
    static constexpr size_t kPostRingSize = 128;
    std::atomic<uint64_t> postRingTimestamps[kPostRingSize]{};
    std::atomic<uint64_t> postRingHead{0};

    std::atomic<uint32_t> pushLogCount{0};
    std::atomic<uint32_t> blitLogCount{0};

    // Snapshot of the last completed kProfileWindow rolling window. Updated
    // atomically by the worker thread when a window closes (see workerThread,
    // ProfileAccum). Layout: [copyNs, presentNs, waitIdleNs, blitNs, totalNs,
    // samples]. Each value is the SUM over the window — divide by samples to
    // get the per-frame average. samples == 0 means no window has completed
    // since session start.
    std::atomic<int64_t> profileSnapshotCopyNs{0};
    std::atomic<int64_t> profileSnapshotPresentNs{0};
    std::atomic<int64_t> profileSnapshotWaitIdleNs{0};
    std::atomic<int64_t> profileSnapshotBlitNs{0};
    std::atomic<int64_t> profileSnapshotTotalNs{0};
    std::atomic<int64_t> profileSnapshotQueueNs{0};
    std::atomic<int64_t> profileSnapshotLatencyNs{0};
    std::atomic<int64_t> profileSnapshotBlitCount{0};
    std::atomic<int64_t> profileSnapshotSamples{0};
    std::atomic<bool> shizukuTimingEnabled{false};
    std::atomic<int64_t> shizukuSampleTimestampNs{0};
    std::atomic<int64_t> shizukuFrameTimeNs{0};
    std::atomic<int64_t> shizukuPacingJitterNs{0};

    // Display vsync period in ns (0 = unknown, falls back to plain sleep_until).
    // Set from Kotlin via setVsyncPeriodNs() once the overlay surface reports
    // its refresh rate (OverlayManager.requestedRefreshRateHz).
    std::atomic<int64_t> vsyncPeriodNs{0};

    // User-tunable pacing parameters. Read on every pacing iteration so changes
    // via setPacingParams() take effect without a context re-init.
    std::atomic<int> targetFpsCapHz{0};      // 0 = unlimited
    std::atomic<int64_t> emaAlphaMicro{125000};  // α * 1e6, default 0.125
    std::atomic<int64_t> outlierRatioMicro{4000000}; // ratio * 1e6, default 4.0
    std::atomic<int64_t> vsyncSlackNs{2'000'000};    // default 2 ms
    std::atomic<int> queueDepth{4};
};

State g{};

// ImageReader recycles its buffers. Reuse the Vulkan import across references
// to the same buffer, but still acquire/release its GPU ownership for each copy.
class CaptureImports {
public:
    ~CaptureImports() {
        for (auto &entry : entries_) destroyAhbImage(g.vk, entry.image);
    }

    const AhbImage *findOrImport(AHardwareBuffer *ahb, AhbImage &temporary) {
        const uint64_t id = getAhbId(ahb);
        if (id != 0) {
            for (auto &entry : entries_) {
                if (entry.id != id) continue;
                entry.lastUse = ++clock_;
                g.cacheHits.fetch_add(1, std::memory_order_relaxed);
                return &entry.image;
            }
        }
        g.cacheMisses.fetch_add(1, std::memory_order_relaxed);
        const int rc = importAhbImage(g.vk, ahb, temporary);
        if (rc != kOk) {
            LOGW("importAhbImage failed rc=%d", rc);
            return nullptr;
        }
        if (id == 0) return &temporary;

        auto &slot = *std::min_element(entries_.begin(), entries_.end(),
            [](const Entry &a, const Entry &b) { return a.lastUse < b.lastUse; });
        destroyAhbImage(g.vk, slot.image);
        AHardwareBuffer_acquire(ahb);
        temporary.ownsAhb = true;
        slot = Entry{id, ++clock_, temporary};
        temporary = {};
        return &slot.image;
    }

private:
    struct Entry {
        uint64_t id = 0;
        uint64_t lastUse = 0;
        AhbImage image{};
    };
    std::array<Entry, 8> entries_{};
    uint64_t clock_ = 0;
};


struct ShizukuTimingSample {
    bool enabled = false;
    int64_t timestampNs = 0;
    int64_t frameTimeNs = 0;
    int64_t pacingJitterNs = 0;
};

ShizukuTimingSample loadShizukuTimingSample() {
    return {
        .enabled = g.shizukuTimingEnabled.load(std::memory_order_relaxed),
        .timestampNs = g.shizukuSampleTimestampNs.load(std::memory_order_relaxed),
        .frameTimeNs = g.shizukuFrameTimeNs.load(std::memory_order_relaxed),
        .pacingJitterNs = g.shizukuPacingJitterNs.load(std::memory_order_relaxed),
    };
}

// Apply pacing tunables to `g` with sane clamps. Zero/negative values for
// non-cap fields fall back to defaults so partial updates from JNI can't
// accidentally disable the pacer.
void applyPacingParamsImpl(int targetFpsCap, float emaAlpha, float outlierRatio,
                           float vsyncSlackMs, int queueDepth) {
    if (targetFpsCap < 0) targetFpsCap = 0;
    if (targetFpsCap > 0 && targetFpsCap < 15) targetFpsCap = 15;
    if (targetFpsCap > 240) targetFpsCap = 240;
    g.targetFpsCapHz.store(targetFpsCap, std::memory_order_relaxed);

    if (emaAlpha < 0.05f || emaAlpha > 0.5f || !std::isfinite(emaAlpha)) emaAlpha = 0.125f;
    g.emaAlphaMicro.store(static_cast<int64_t>(emaAlpha * 1'000'000.0), std::memory_order_relaxed);

    if (outlierRatio < 2.0f || outlierRatio > 8.0f || !std::isfinite(outlierRatio)) outlierRatio = 4.0f;
    g.outlierRatioMicro.store(static_cast<int64_t>(outlierRatio * 1'000'000.0), std::memory_order_relaxed);

    if (vsyncSlackMs < 1.0f || vsyncSlackMs > 5.0f || !std::isfinite(vsyncSlackMs)) vsyncSlackMs = 2.0f;
    g.vsyncSlackNs.store(static_cast<int64_t>(vsyncSlackMs * 1'000'000.0), std::memory_order_relaxed);

    if (queueDepth < 2) queueDepth = 2;
    if (queueDepth > 6) queueDepth = 6;
    g.queueDepth.store(queueDepth, std::memory_order_relaxed);
}

float clamp01(float value) {
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

float clampScale(float value) {
    return value < 1.0f ? 1.0f : (value > 2.0f ? 2.0f : value);
}

const char *gpuMethodName(int method) {
    switch (method) {
        case 0: return "FSR1_EASU_RCAS";
        case 1: return "AMD_CAS";
        case 2: return "NVIDIA_NIS";
        case 3: return "LANCZOS";
        case 4: return "BICUBIC";
        case 5: return "BILINEAR";
        case 6: return "CATMULL_ROM";
        case 7: return "MITCHELL_NETRAVALI";
        case 8: return "ANIME4K_ULTRAFAST";
        case 9: return "ANIME4K_RESTORE";
        case 10: return "XBRZ";
        case 11: return "EDGE_DIRECTED";
        case 12: return "UNSHARP_MASK";
        case 13: return "LUMA_SHARPEN";
        case 14: return "CONTRAST_ADAPTIVE";
        case 15: return "DEBAND";
        case 16: return "SGSR_V1";
        default: return "FSR1_EASU_RCAS";
    }
}

const char *gpuStageName(int stage) {
    return stage == 0 ? "before_lsfg" : "after_lsfg";
}

bool gpuWantsPreLsfg() {
    return false;
}

bool gpuWantsPostLsfg() {
    return g.gpuPostProcessing.load(std::memory_order_relaxed);
}

void noteGpuShaderPassPending() {
    if (!g.gpuPostProcessing.load(std::memory_order_relaxed)) return;
    if (g.gpuNoopLogged.exchange(true, std::memory_order_relaxed)) return;
    LOGI("GPU SR %s active sharp=%.2f strength=%.2f",
         gpuMethodName(g.gpuMethod.load(std::memory_order_relaxed)),
         g.gpuSharpness.load(std::memory_order_relaxed),
         g.gpuStrength.load(std::memory_order_relaxed));
}

// Compute a cheap FNV-1a hash of an 8×8 luma grid sampled from the AHB.
// Two identical game frames produce the same hash; two differently-rendered
// frames produce different hashes with overwhelming probability. Quantizing
// each luma by >>2 (dropping 2 LSBs) suppresses encoder dither noise that
// would otherwise mark semantically-identical frames as "different".
//
// Cost: ~50-100 μs total — 1× AHardwareBuffer_lock (the AHB was allocated
// with CPU_READ_OFTEN so the lock is cheap), 64× 4-byte reads, unlock.
// Called once per pushFrame from the ImageReader listener thread.
//
// Returns 0 on any failure (the caller treats 0 as "unknown" and doesn't
// update uniqueCaptures, which is fine — worst case the metric stalls for
// one frame).
uint32_t captureContentHash(AHardwareBuffer *ahb) {
    if (ahb == nullptr) return 0;
    AHardwareBuffer_Desc desc{};
    AHardwareBuffer_describe(ahb, &desc);
    if (desc.format != AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM &&
        desc.format != AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM) {
        // Uncommon AHB format — skip metric rather than risk reading wrong
        // bytes. The counter will look stuck but the app won't misbehave.
        return 0;
    }
    void *ptr = nullptr;
    if (AHardwareBuffer_lock(ahb, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
            -1, nullptr, &ptr) != 0 || ptr == nullptr) {
        return 0;
    }
    const uint32_t stride = desc.stride * 4u;  // 4 bytes per RGBA8 pixel
    const auto *bytes = static_cast<const uint8_t *>(ptr);
    constexpr uint32_t kGrid = 8;  // 8×8 = 64 samples
    uint32_t hash = 0x811c9dc5u;   // FNV-1a offset basis
    for (uint32_t gy = 0; gy < kGrid; ++gy) {
        const uint32_t y = std::min<uint32_t>((gy * desc.height) / kGrid,
                                              desc.height > 0 ? desc.height - 1 : 0);
        for (uint32_t gx = 0; gx < kGrid; ++gx) {
            const uint32_t x = std::min<uint32_t>((gx * desc.width) / kGrid,
                                                  desc.width > 0 ? desc.width - 1 : 0);
            const uint8_t *p = bytes + y * stride + x * 4u;
            // Rec.709 luma, quantized by >>2 to drop the 2 noisiest bits.
            const uint32_t luma = ((77u * p[0] + 150u * p[1] + 29u * p[2]) >> 10);
            hash = (hash ^ luma) * 0x01000193u;  // FNV-1a prime
        }
    }
    AHardwareBuffer_unlock(ahb, nullptr);
    // Avoid returning 0 because that's our "unknown" sentinel: two frames
    // that genuinely hash to 0 would otherwise be miscounted as dupes.
    return hash == 0u ? 1u : hash;
}

std::optional<AnimeSignature> captureAnimeSignature(AHardwareBuffer *ahb) {
    AHardwareBuffer_Desc desc{};
    AHardwareBuffer_describe(ahb, &desc);
    if (desc.format != AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM &&
        desc.format != AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM) return std::nullopt;
    if (desc.width == 0 || desc.height == 0 || desc.stride < desc.width) return std::nullopt;
    void *data = nullptr;
    if (AHardwareBuffer_lock(ahb, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,
                             -1, nullptr, &data) != 0) return std::nullopt;
    std::optional<AnimeSignature> signature;
    if (data) {
        signature = sampleAnimeFrame(static_cast<const uint8_t *>(data), desc.width,
                                     desc.height, static_cast<size_t>(desc.stride) * 4);
    }
    AHardwareBuffer_unlock(ahb, nullptr);
    return signature;
}

bool shouldSuppressGeneratedFrames(const AhbImage &prev, const AhbImage &cur) {
    if (prev.ahb == nullptr || cur.ahb == nullptr) return false;
    if (prev.extent.width == 0 || prev.extent.height == 0) return false;
    if (prev.extent.width != cur.extent.width || prev.extent.height != cur.extent.height) {
        return true;
    }

    AHardwareBuffer_Desc prevDesc{};
    AHardwareBuffer_Desc curDesc{};
    AHardwareBuffer_describe(prev.ahb, &prevDesc);
    AHardwareBuffer_describe(cur.ahb, &curDesc);
    if (prevDesc.format != AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM ||
            curDesc.format != AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM) {
        return false;
    }

    void *prevPtr = nullptr;
    void *curPtr = nullptr;
    if (AHardwareBuffer_lock(prev.ahb, AHARDWAREBUFFER_USAGE_CPU_READ_RARELY,
            -1, nullptr, &prevPtr) != 0 || prevPtr == nullptr) {
        return false;
    }
    if (AHardwareBuffer_lock(cur.ahb, AHARDWAREBUFFER_USAGE_CPU_READ_RARELY,
            -1, nullptr, &curPtr) != 0 || curPtr == nullptr) {
        AHardwareBuffer_unlock(prev.ahb, nullptr);
        return false;
    }

    constexpr uint32_t kGridX = 32;
    constexpr uint32_t kGridY = 18;
    constexpr uint32_t kLumaDeltaThreshold = 26;
    const uint32_t w = std::min(prevDesc.width, curDesc.width);
    const uint32_t h = std::min(prevDesc.height, curDesc.height);
    const uint32_t prevStride = prevDesc.stride * 4;
    const uint32_t curStride = curDesc.stride * 4;
    const auto *prevBytes = static_cast<const uint8_t *>(prevPtr);
    const auto *curBytes = static_cast<const uint8_t *>(curPtr);

    uint64_t totalDelta = 0;
    uint32_t samples = 0;
    for (uint32_t gy = 0; gy < kGridY; ++gy) {
        const uint32_t y = std::min((gy * h) / kGridY, h - 1);
        for (uint32_t gx = 0; gx < kGridX; ++gx) {
            const uint32_t x = std::min((gx * w) / kGridX, w - 1);
            const uint8_t *a = prevBytes + y * prevStride + x * 4;
            const uint8_t *b = curBytes + y * curStride + x * 4;
            const int prevLuma = (77 * a[0] + 150 * a[1] + 29 * a[2]) >> 8;
            const int curLuma = (77 * b[0] + 150 * b[1] + 29 * b[2]) >> 8;
            totalDelta += static_cast<uint32_t>(std::abs(curLuma - prevLuma));
            samples++;
        }
    }

    AHardwareBuffer_unlock(cur.ahb, nullptr);
    AHardwareBuffer_unlock(prev.ahb, nullptr);
    return samples > 0 && (totalDelta / samples) >= kLumaDeltaThreshold;
}

// Sleep until `deadline`, but if we know the display's vsync period, nudge
// the target to a vsync boundary. `lastPostedAt` is the wall time of the most
// recent ANativeWindow_unlockAndPost call — if the computed deadline lies in
// the same vsync slot as that post, we push it to the NEXT boundary so the
// SurfaceFlinger queue doesn't collapse two buffers onto one flip (which is
// what produces the steady-state "bunched" stutter we see with multiplier≥2).
//
// Returns the (possibly adjusted) wake time so the caller can advance its
// own deadline chain consistently.
State::Clock::time_point sleepUntilVsyncAligned(
        State::Clock::time_point deadline,
        State::Clock::time_point lastPostedAt,
        State::Clock::duration slotBudget) {
    const int64_t periodNs = g.vsyncPeriodNs.load(std::memory_order_relaxed);
    if (periodNs <= 0) {
        std::this_thread::sleep_until(deadline);
        return deadline;
    }
    const auto period = std::chrono::nanoseconds(periodNs);
    // Minimum separation from the last post: one full vsync minus a small
    // slack so we don't undershoot a boundary and still land in the same slot.
    // User-tunable (1..5 ms); 2 ms default is below average kernel wake-up jitter.
    const auto slack = std::chrono::nanoseconds(
        g.vsyncSlackNs.load(std::memory_order_relaxed));
    const auto minSeparatedSlot = period - slack;
    if (slotBudget < minSeparatedSlot) {
        std::this_thread::sleep_until(deadline);
        return deadline;
    }
    const auto earliest = lastPostedAt + period - slack;
    if (deadline < earliest) deadline = earliest;
    std::this_thread::sleep_until(deadline);
    return deadline;
}

// Record a successful post (CPU blit or WSI present) for HUD metrics.
// Increments postedFrames and pushes the current steady-clock timestamp into
// the ring buffer so the pacing graph can compute real inter-frame intervals.
void recordOverlayPost() {
    g.postedFrames.fetch_add(1, std::memory_order_relaxed);
    const uint64_t nowNs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            State::Clock::now().time_since_epoch()).count());
    const uint64_t slot = g.postRingHead.fetch_add(1, std::memory_order_relaxed)
                          % State::kPostRingSize;
    g.postRingTimestamps[slot].store(nowNs, std::memory_order_relaxed);
}

// ---- Vulkan swapchain helpers ------------------------------------------------
//
// The swapchain lives on top of the overlay's ANativeWindow and provides the
// GPU-only output path: generated frames are vkCmdBlitImage'd from their AHB
// storage directly into the swapchain image and presented via vkQueuePresentKHR.
// No CPU touch of the pixel data.
//
// DEFAULT OFF. An earlier run on Adreno 840 / Android 14 crashed in a driver
// call during createSwapchain before any of the per-step LOGIs could fire —
// the Adreno compute queue reports `vkGetPhysicalDeviceSurfaceSupportKHR =
// VK_TRUE` but then crashes inside `vkQueuePresentKHR` (known quirk on some
// Qualcomm revisions). Keeping the code path in-tree with aggressive logging
// so future device revisions / validation work can flip this flag without
// another refactor. To enable for testing, set `kEnableWsiSwapchain` to true.
constexpr bool kEnableWsiSwapchain = true;

void destroySwapchain() {
    // Must drain every in-flight ring submission before touching the images
    // — the swapchain images are referenced by recorded CBs via barriers and
    // destroying them while those CBs are still executing = SIGSEGV on some
    // drivers (observed on Qualcomm). vkDeviceWaitIdle is the sledgehammer.
    if (g.vk.device != VK_NULL_HANDLE && g.vk.fn.vkDeviceWaitIdle != nullptr) {
        g.vk.fn.vkDeviceWaitIdle(g.vk.device);
    }

    if (g.vk.fn.vkDestroySemaphore != nullptr) {
        for (VkSemaphore s : g.swap.acquireSems) {
            if (s != VK_NULL_HANDLE) g.vk.fn.vkDestroySemaphore(g.vk.device, s, nullptr);
        }
        for (VkSemaphore s : g.swap.renderSems) {
            if (s != VK_NULL_HANDLE) g.vk.fn.vkDestroySemaphore(g.vk.device, s, nullptr);
        }
    }
    g.swap.acquireSems.clear();
    g.swap.renderSems.clear();
    g.swap.images.clear();
    g.swap.acquireCursor = 0;
    g.swap.outOfDate = false;
    g.swap.disabledForSession = false;

    if (g.swap.swapchain != VK_NULL_HANDLE && g.vk.fn.vkDestroySwapchainKHR != nullptr) {
        g.vk.fn.vkDestroySwapchainKHR(g.vk.device, g.swap.swapchain, nullptr);
        g.swap.swapchain = VK_NULL_HANDLE;
    }
    if (g.swap.surface != VK_NULL_HANDLE && g.vk.instance != VK_NULL_HANDLE) {
        // Surface destruction uses the instance-level function. volk populates
        // it globally after volkLoadInstance.
        if (g.vk.pfnDestroySurfaceKHR != nullptr) {
            g.vk.pfnDestroySurfaceKHR(g.vk.instance, g.swap.surface, nullptr);
        } else {
            vkDestroySurfaceKHR(g.vk.instance, g.swap.surface, nullptr);
        }
        g.swap.surface = VK_NULL_HANDLE;
    }
    g.swap.extent = {0, 0};
    g.swap.format = VK_FORMAT_UNDEFINED;
}

// Build (or rebuild) the swapchain on the current outWindow. Returns true if
// the WSI path is live after this call; false means the caller must fall back
// to the CPU blit path for this session.
//
// Safe to call multiple times; previous swapchain is torn down first.
bool createSwapchain() {
    LOGI("createSwapchain: enter outWindow=%p hasSwapchain=%d enable=%d cpuTainted=%d",
         static_cast<void *>(g.outWindow), (int)g.vk.hasSwapchain,
         (int)kEnableWsiSwapchain, (int)g.windowCpuProducerLocked);
    destroySwapchain();
    if (!kEnableWsiSwapchain) return false;
    if (!g.vk.hasSwapchain) return false;
    if (g.outWindow == nullptr) return false;
    if (g.vk.instance == VK_NULL_HANDLE) return false;
    // Once the worker has produced any CPU buffer on this ANativeWindow
    // (overlay clear, CPU blit fallback, etc.) the BufferQueue is locked to
    // a CPU producer and vkCreateAndroidSurfaceKHR will return
    // VK_ERROR_NATIVE_WINDOW_IN_USE_KHR (-1000000001) on Mali / many other
    // drivers. Don't even attempt — failed surface creation has been observed
    // to correlate with framegen-side DEVICE_LOST on Mali-G57 (likely an
    // instance-level state corruption when the WSI driver path bails late).
    if (g.windowCpuProducerLocked) {
        LOGI("createSwapchain: skipping — window already bound to CPU producer this session");
        return false;
    }
    // Use the session-cached function pointer instead of volk's global. The
    // global gets clobbered to NULL when framegen's Instance::Instance()
    // calls volkLoadInstance() against its own surface-less VkInstance,
    // because that instance can't resolve vkCreateAndroidSurfaceKHR. The
    // session pointer was resolved against OUR instance at session-init time
    // and survives that overwrite.
    PFN_vkCreateAndroidSurfaceKHR pfnCreateSurface = g.vk.pfnCreateAndroidSurfaceKHR;
    if (pfnCreateSurface == nullptr) pfnCreateSurface = vkCreateAndroidSurfaceKHR;
    if (pfnCreateSurface == nullptr) {
        LOGW("vkCreateAndroidSurfaceKHR fn ptr is NULL — volk didn't load it");
        return false;
    }

    LOGI("createSwapchain: calling vkCreateAndroidSurfaceKHR");
    const VkAndroidSurfaceCreateInfoKHR sci{
        .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
        .window = g.outWindow,
    };
    const VkResult sres = pfnCreateSurface(g.vk.instance, &sci, nullptr, &g.swap.surface);
    if (sres != VK_SUCCESS) {
        // VK_ERROR_NATIVE_WINDOW_IN_USE_KHR means the ANativeWindow is locked
        // to a CPU producer (or another consumer). The state is sticky — no
        // amount of retrying on the same window will recover. Pin the taint
        // so subsequent createSwapchain calls bail at the early gate above
        // instead of repeating the same failed driver call (which on some
        // Mali / Adreno revisions can leave the instance in a tainted state
        // and induce a DEVICE_LOST on framegen's separate device).
        if (sres == VK_ERROR_NATIVE_WINDOW_IN_USE_KHR) {
            g.windowCpuProducerLocked = true;
            LOGW("vkCreateAndroidSurfaceKHR: window in use by CPU producer (rc=%d) — WSI disabled for this surface; using CPU blit",
                 (int)sres);
        } else {
            LOGW("vkCreateAndroidSurfaceKHR failed (rc=%d) — falling back to CPU blit", (int)sres);
        }
        g.swap.surface = VK_NULL_HANDLE;
        return false;
    }
    LOGI("createSwapchain: surface=%p", static_cast<void *>(g.swap.surface));

    // Can our compute queue actually present on this surface?
    VkBool32 canPresent = VK_FALSE;
    const VkResult suppr = g.vk.pfnGetPhysicalDeviceSurfaceSupportKHR(g.vk.physicalDevice,
            g.vk.computeFamilyIdx, g.swap.surface, &canPresent);
    LOGI("createSwapchain: surfaceSupport rc=%d canPresent=%d", (int)suppr, (int)canPresent);
    if (suppr != VK_SUCCESS || canPresent != VK_TRUE) {
        LOGW("compute queue family %u cannot present on this surface — fall back to CPU blit",
             g.vk.computeFamilyIdx);
        destroySwapchain();
        return false;
    }

    VkSurfaceCapabilitiesKHR caps{};
    if (g.vk.pfnGetPhysicalDeviceSurfaceCapabilitiesKHR(g.vk.physicalDevice,
            g.swap.surface, &caps) != VK_SUCCESS) {
        LOGW("vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed");
        destroySwapchain();
        return false;
    }

    // Format: prefer RGBA8 UNORM since that's what framegen outputs. Fall back
    // to whatever the driver offers if not present.
    uint32_t fmtCount = 0;
    g.vk.pfnGetPhysicalDeviceSurfaceFormatsKHR(g.vk.physicalDevice, g.swap.surface,
                                         &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtCount);
    if (fmtCount > 0) {
        g.vk.pfnGetPhysicalDeviceSurfaceFormatsKHR(g.vk.physicalDevice, g.swap.surface,
                                             &fmtCount, fmts.data());
    }
    VkSurfaceFormatKHR chosen{VK_FORMAT_R8G8B8A8_UNORM, VK_COLORSPACE_SRGB_NONLINEAR_KHR};
    bool haveChosen = false;
    for (const auto &f : fmts) {
        if (f.format == VK_FORMAT_R8G8B8A8_UNORM ||
            f.format == VK_FORMAT_R8G8B8A8_SRGB) {
            chosen = f;
            haveChosen = true;
            break;
        }
    }
    if (!haveChosen && !fmts.empty()) chosen = fmts.front();

    // Present mode: FIFO is the only guaranteed mode and is exactly what we
    // want — the pacing loop already emits frames at their target slots, and
    // FIFO gives us a vsync-bounded queue. MAILBOX would let frames drop,
    // which defeats the purpose of the generated-frame schedule.
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;

    // Image count: 3 gives a comfortable buffer under FIFO (acquire can run
    // ahead by one while SurfaceFlinger holds the currently-displayed image).
    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
        imageCount = caps.maxImageCount;
    }
    if (imageCount < 2) imageCount = 2;

    // Always request IDENTITY preTransform when supported so SurfaceFlinger/HWC
    // handles 90°/270° landscape fullscreen rotation in hardware instead of
    // forcing a rotated portrait extent into our unrotated copy/blit.
    const VkSurfaceTransformFlagBitsKHR preTransform =
        (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
            ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
            : caps.currentTransform;

    VkExtent2D extent = caps.currentExtent;
    if (preTransform == VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR &&
        g.outWidth > 0 && g.outHeight > 0) {
        extent.width = std::clamp(g.outWidth, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(g.outHeight, caps.minImageExtent.height, caps.maxImageExtent.height);
    } else {
        if (extent.width == 0 || extent.width == UINT32_MAX) {
            extent.width = g.outWidth > 0 ? g.outWidth : 1920;
        }
        if (extent.height == 0 || extent.height == UINT32_MAX) {
            extent.height = g.outHeight > 0 ? g.outHeight : 1080;
        }
    }

    // TRANSFER_DST is mandatory for vkCmdBlitImage into the swapchain images.
    // Some drivers require explicit opt-in via capabilities.supportedUsageFlags.
    if ((caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0) {
        LOGW("swapchain surface does not advertise TRANSFER_DST usage — falling back to CPU blit");
        destroySwapchain();
        return false;
    }

    const VkSwapchainCreateInfoKHR sci2{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = g.swap.surface,
        .minImageCount = imageCount,
        .imageFormat = chosen.format,
        .imageColorSpace = chosen.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = preTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = presentMode,
        .clipped = VK_TRUE,
    };
    LOGI("createSwapchain: vkCreateSwapchainKHR fmt=%d extent=%ux%u imageCount=%u",
         (int)chosen.format, extent.width, extent.height, imageCount);
    if (g.vk.fn.vkCreateSwapchainKHR == nullptr) {
        LOGW("vkCreateSwapchainKHR fn ptr is NULL");
        destroySwapchain();
        return false;
    }
    if (g.vk.fn.vkCreateSwapchainKHR(g.vk.device, &sci2, nullptr,
            &g.swap.swapchain) != VK_SUCCESS) {
        LOGW("vkCreateSwapchainKHR failed");
        destroySwapchain();
        return false;
    }
    LOGI("createSwapchain: swapchain=%p", static_cast<void *>(g.swap.swapchain));

    uint32_t realCount = 0;
    g.vk.fn.vkGetSwapchainImagesKHR(g.vk.device, g.swap.swapchain, &realCount, nullptr);
    g.swap.images.resize(realCount);
    g.vk.fn.vkGetSwapchainImagesKHR(g.vk.device, g.swap.swapchain, &realCount,
                                    g.swap.images.data());

    // Semaphore pools:
    //   acquireSems: one per "in-flight frame" slot (we use realCount + 1)
    //   renderSems:  one per swapchain image (vkQueuePresent waits on this
    //                semaphore for the specific image we rendered to)
    g.swap.acquireSems.resize(realCount + 1, VK_NULL_HANDLE);
    g.swap.renderSems.resize(realCount, VK_NULL_HANDLE);
    const VkSemaphoreCreateInfo semInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    for (auto &s : g.swap.acquireSems) {
        if (g.vk.fn.vkCreateSemaphore(g.vk.device, &semInfo, nullptr, &s) != VK_SUCCESS) {
            LOGW("vkCreateSemaphore(acquire) failed");
            destroySwapchain();
            return false;
        }
    }
    for (auto &s : g.swap.renderSems) {
        if (g.vk.fn.vkCreateSemaphore(g.vk.device, &semInfo, nullptr, &s) != VK_SUCCESS) {
            LOGW("vkCreateSemaphore(render) failed");
            destroySwapchain();
            return false;
        }
    }

    g.swap.extent = extent;
    g.swap.format = chosen.format;
    g.swap.acquireCursor = 0;
    g.swap.outOfDate = false;
    LOGI("Swapchain ready: %ux%u fmt=%d images=%u mode=FIFO",
         extent.width, extent.height, (int)chosen.format, realCount);
    return true;
}

// Blit `src` AHB-backed VkImage to the next swapchain image and present.
// Returns true on success. On VK_ERROR_OUT_OF_DATE_KHR or _SUBOPTIMAL_KHR the
// swapchain is marked dirty; the caller's next blit will recreate it.
bool blitOutputToSwapchain(const AhbImage &src, uint64_t desiredPresentNs) {
    // Note: Called with g.mu held from blitOutputToWindow.
    if (g.swap.swapchain == VK_NULL_HANDLE) return false;
    if (src.image == VK_NULL_HANDLE) return false;
    if (g.swap.outOfDate) {
        if (!createSwapchain()) return false;
    }

    if (g.swap.acquireSems.empty()) return false;

    // Pick an acquire semaphore from the round-robin pool. Using a single
    // semaphore risks "semaphore already has a pending wait" on fast backs.
    VkSemaphore acquireSem = g.swap.acquireSems[g.swap.acquireCursor];
    g.swap.acquireCursor = (g.swap.acquireCursor + 1) % g.swap.acquireSems.size();

    uint32_t imageIdx = 0;
    const VkResult ar = g.vk.fn.vkAcquireNextImageKHR(g.vk.device, g.swap.swapchain,
        500ULL * 1'000'000ULL,  // 500 ms: generous — SurfaceFlinger normally returns in <16ms
        acquireSem, VK_NULL_HANDLE, &imageIdx);
    if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
        g.swap.outOfDate = true;
        return false;
    }
    if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) {
        LOGW("vkAcquireNextImageKHR returned %d", (int)ar);
        return false;
    }

    if (imageIdx >= g.swap.images.size()) {
        LOGW("vkAcquireNextImageKHR returned OOB index %u (size %zu)", imageIdx, g.swap.images.size());
        return false;
    }

    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    if (!acquirePresentRing(g.vk, cb, fence)) return false;
    const VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    g.vk.fn.vkBeginCommandBuffer(cb, &bi);

    // Source: AHB-backed image owned by framegen's device between uses.
    // Acquire for TRANSFER_READ, release back to EXTERNAL after blit.
    VkImageMemoryBarrier srcAcquire{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = src.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    // Destination: swapchain image. Start UNDEFINED → TRANSFER_DST, blit,
    // then → PRESENT_SRC so SurfaceFlinger can pick it up.
    VkImageMemoryBarrier dstToTransfer{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = g.swap.images[imageIdx],
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageMemoryBarrier pre[2] = {srcAcquire, dstToTransfer};
    g.vk.fn.vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 2, pre);

    static std::atomic<bool> loggedBlitMode{false};

    if (src.format == g.swap.format && src.extent.width == g.swap.extent.width &&
        src.extent.height == g.swap.extent.height) {
        if (!loggedBlitMode.exchange(true, std::memory_order_relaxed)) {
            LOGW("blitOutputToSwapchain: 1:1 scale detected. Using FAST-PATH vkCmdCopyImage.");
        }
        const VkImageCopy region{
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffset = {0, 0, 0},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffset = {0, 0, 0},
            .extent = {src.extent.width, src.extent.height, 1},
        };
        g.vk.fn.vkCmdCopyImage(cb,
            src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            g.swap.images[imageIdx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &region);
    } else {
        if (!loggedBlitMode.exchange(true, std::memory_order_relaxed)) {
            LOGW("blitOutputToSwapchain: Scaled output detected (%ux%u -> %ux%u). Using VK_FILTER_LINEAR vkCmdBlitImage.",
                 src.extent.width, src.extent.height, g.swap.extent.width, g.swap.extent.height);
        }
        const VkImageBlit region{
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets = {{0, 0, 0}, {
                static_cast<int32_t>(src.extent.width),
                static_cast<int32_t>(src.extent.height), 1,
            }},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffsets = {{0, 0, 0}, {
                static_cast<int32_t>(g.swap.extent.width),
                static_cast<int32_t>(g.swap.extent.height), 1,
            }},
        };
        g.vk.fn.vkCmdBlitImage(cb,
            src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            g.swap.images[imageIdx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &region, VK_FILTER_LINEAR);
    }

    VkImageMemoryBarrier srcRelease{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .dstAccessMask = 0,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = src.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageMemoryBarrier dstToPresent{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = 0,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = g.swap.images[imageIdx],
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageMemoryBarrier post[2] = {srcRelease, dstToPresent};
    g.vk.fn.vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 2, post);

    g.vk.fn.vkEndCommandBuffer(cb);

    VkSemaphore renderSem = g.swap.renderSems[imageIdx];
    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &acquireSem,
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &renderSem,
    };
    {
        std::lock_guard<std::mutex> qLock(g.queueMu);
        if (g.vk.fn.vkQueueSubmit(g.vk.computeQueue, 1, &si, fence) != VK_SUCCESS) {
            LOGW("vkQueueSubmit(swapchain blit) failed");
            return false;
        }
    }
    markPresentFenceArmed(g.vk, fence);

    const VkPresentTimeGOOGLE presentationTime{
        .presentID = ++g.swap.presentId,
        .desiredPresentTime = desiredPresentNs,
    };
    const VkPresentTimesInfoGOOGLE presentationTimes{
        .sType = VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE,
        .swapchainCount = 1,
        .pTimes = &presentationTime,
    };
    const VkPresentInfoKHR pi{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = g.vk.hasDisplayTiming && desiredPresentNs ? &presentationTimes : nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &renderSem,
        .swapchainCount = 1,
        .pSwapchains = &g.swap.swapchain,
        .pImageIndices = &imageIdx,
    };
    VkResult pr;
    {
        std::lock_guard<std::mutex> qLock(g.queueMu);
        pr = g.vk.fn.vkQueuePresentKHR(g.vk.computeQueue, &pi);
    }
    if (pr == VK_ERROR_OUT_OF_DATE_KHR) {
        // Note for next pass — don't fall back this frame (we already posted).
        g.swap.outOfDate = true;
        recordOverlayPost();
    } else if (pr == VK_SUBOPTIMAL_KHR) {
        // On Android WSI, preTransform=IDENTITY in landscape returns
        // VK_SUBOPTIMAL_KHR on every present because SurfaceFlinger performs
        // rotation in hardware. Never recreate the swapchain for SUBOPTIMAL.
        recordOverlayPost();
    } else if (pr != VK_SUCCESS) {
        LOGW("vkQueuePresentKHR returned %d", (int)pr);
    } else {
        recordOverlayPost();
    }
    return true;
}

// Transition a freshly-created AHB-backed inSlot image UNDEFINED→GENERAL
// and release ownership to VK_QUEUE_FAMILY_EXTERNAL. This must be called once
// per inSlot after createAhbImage so that every subsequent copyAhbImage can
// acquire with oldLayout=GENERAL, preserving Mali AFRC state across frames.
void initInSlotImageLayout(VkImage image) {
    VkCommandBufferAllocateInfo cbai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = g.vk.commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (g.vk.fn.vkAllocateCommandBuffers(g.vk.device, &cbai, &cb) != VK_SUCCESS) {
        LOGE("initInSlotImageLayout: vkAllocateCommandBuffers failed");
        return;
    }
    const VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    g.vk.fn.vkBeginCommandBuffer(cb, &bi);
    // Transition UNDEFINED→GENERAL and release to EXTERNAL in one barrier.
    const VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = 0,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    g.vk.fn.vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    g.vk.fn.vkEndCommandBuffer(cb);
    const VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    g.vk.fn.vkQueueSubmit(g.vk.computeQueue, 1, &si, VK_NULL_HANDLE);
    g.vk.fn.vkQueueWaitIdle(g.vk.computeQueue);
    g.vk.fn.vkFreeCommandBuffers(g.vk.device, g.vk.commandPool, 1, &cb);
    LOGI("initInSlotImageLayout: image=%p initialized to GENERAL/EXTERNAL", (void*)image);
}

// Copy `src` AhbImage into `dst` AhbImage on the compute queue using a
// transient command buffer. Both images are AHB-backed so layouts are
// EXTERNAL initially; we transition to TRANSFER_DST/SRC, copy, transition
// back to GENERAL so framegen can read them.
//
// Uses transient alloc+free per call rather than the session's CB ring:
// on Adreno the per-frame vkAllocateCommandBuffers/vkFreeCommandBuffers
// pair runs in single-digit microseconds, while vkResetCommandBuffer +
// vkWaitForFences (the ring path) was measurably slower in field testing.
void recordAhbCopy(VkCommandBuffer cb, const AhbImage &src, const AhbImage &dst) {
    const uint32_t foreign = g.vk.hasQueueFamilyForeign ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_EXTERNAL;
    VkImageMemoryBarrier toSrc{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = foreign,
        .dstQueueFamilyIndex = g.vk.computeFamilyIdx,
        .image = src.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageMemoryBarrier toDst{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,     // always GENERAL: set by initInSlotImageLayout and preserved by releaseDst
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,           // framegen had it last
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = dst.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageMemoryBarrier preBarriers[2] = {toSrc, toDst};
    g.vk.fn.vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 2, preBarriers);

    const uint32_t w = std::min(src.extent.width,  dst.extent.width);
    const uint32_t h = std::min(src.extent.height, dst.extent.height);
    const VkImageCopy region{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffset = {0, 0, 0},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffset = {0, 0, 0},
        .extent = {w, h, 1},
    };
    g.vk.fn.vkCmdCopyImage(cb,
        src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &region);

    // The capture buffer returns to its producer. The copied input stays on
    // our queue and is immediately visible to the interpolation shaders.
    VkImageMemoryBarrier releaseDst{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = dst.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageMemoryBarrier releaseSrc{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .dstAccessMask = 0,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = g.vk.computeFamilyIdx,
        .dstQueueFamilyIndex = foreign,
        .image = src.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    g.vk.fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &releaseDst);
    g.vk.fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &releaseSrc);
}

bool beginFrameCommands(VkCommandBuffer cb) {
    const VkCommandBufferBeginInfo begin{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    return g.vk.fn.vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS;
}

bool completeFrameCommands(VkCommandBuffer cb, VkFence fence) {
    if (g.vk.fn.vkEndCommandBuffer(cb) != VK_SUCCESS) return false;
    {
        std::lock_guard<std::mutex> qLock(g.queueMu);
        if (!submitCommandRing(g.vk, cb, fence)) return false;
    }
    if (waitCommandRing(g.vk, fence)) return true;
    // The capture Image cannot be recycled while a submitted copy can still
    // access it. Drain on the exceptional timeout or device-loss path.
    std::lock_guard<std::mutex> qLock(g.queueMu);
    return g.vk.fn.vkQueueWaitIdle(g.vk.computeQueue) == VK_SUCCESS;
}

bool copyAhbImage(const AhbImage &src, const AhbImage &dst) {
    VkCommandBuffer cb;
    VkFence fence;
    if (!acquireCommandRing(g.vk, cb, fence) || !beginFrameCommands(cb)) return false;
    recordAhbCopy(cb, src, dst);
    return completeFrameCommands(cb, fence);
}

bool blitAhbImageGpu(const AhbImage &src, const AhbImage &dst) {
    if (src.image == VK_NULL_HANDLE || dst.image == VK_NULL_HANDLE) return false;
    VkCommandBufferAllocateInfo cbai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = g.vk.commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (g.vk.fn.vkAllocateCommandBuffers(g.vk.device, &cbai, &cb) != VK_SUCCESS) return false;

    const VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    g.vk.fn.vkBeginCommandBuffer(cb, &bi);

    const uint32_t foreign = g.vk.hasQueueFamilyForeign ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_EXTERNAL;
    VkImageMemoryBarrier toSrc{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = foreign,
        .dstQueueFamilyIndex = g.vk.computeFamilyIdx,
        .image = src.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageMemoryBarrier toDst{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = dst.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageMemoryBarrier preBarriers[2] = {toSrc, toDst};
    g.vk.fn.vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 2, preBarriers);

    const VkImageBlit region{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffsets = {{0, 0, 0}, {
            static_cast<int32_t>(src.extent.width),
            static_cast<int32_t>(src.extent.height),
            1,
        }},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffsets = {{0, 0, 0}, {
            static_cast<int32_t>(dst.extent.width),
            static_cast<int32_t>(dst.extent.height),
            1,
        }},
    };
    g.vk.fn.vkCmdBlitImage(cb,
        src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &region, VK_FILTER_LINEAR);

    VkImageMemoryBarrier releaseSrc{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .dstAccessMask = 0,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = g.vk.computeFamilyIdx,
        .dstQueueFamilyIndex = foreign,
        .image = src.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageMemoryBarrier releaseDst{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = dst.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageMemoryBarrier postBarriers[2] = {releaseSrc, releaseDst};
    g.vk.fn.vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 2, postBarriers);

    g.vk.fn.vkEndCommandBuffer(cb);
    const VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    bool completed = false;
    {
        std::lock_guard<std::mutex> qLock(g.queueMu);
        const bool ok = g.vk.fn.vkQueueSubmit(g.vk.computeQueue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS;
        completed = ok && g.vk.fn.vkQueueWaitIdle(g.vk.computeQueue) == VK_SUCCESS;
    }
    g.vk.fn.vkFreeCommandBuffers(g.vk.device, g.vk.commandPool, 1, &cb);
    return completed;
}

bool initPresentImageLayout(VkImage image) {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    if (!acquirePresentRing(g.vk, cb, fence)) return false;
    const VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (g.vk.fn.vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) return false;
    const VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = 0,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    g.vk.fn.vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    if (g.vk.fn.vkEndCommandBuffer(cb) != VK_SUCCESS) return false;
    const VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    {
        std::lock_guard<std::mutex> qLock(g.queueMu);
        if (g.vk.fn.vkQueueSubmit(g.vk.computeQueue, 1, &si, fence) != VK_SUCCESS) return false;
    }
    markPresentFenceArmed(g.vk, fence);
    return g.vk.fn.vkWaitForFences(g.vk.device, 1, &fence, VK_TRUE, 500ULL * 1'000'000ULL) == VK_SUCCESS;
}

bool ensureGpuPostImage(uint32_t width, uint32_t height) {
    if (g.gpuPostImage.ahb != nullptr &&
            g.gpuPostImage.extent.width == width &&
            g.gpuPostImage.extent.height == height) {
        return true;
    }
    destroyAhbImage(g.vk, g.gpuPostImage);
    const int rc = createAhbImage(g.vk, width, height, VK_FORMAT_R8G8B8A8_UNORM,
                                  g.gpuPostImage);
    if (rc != kOk) {
        LOGW("GPU post-process intermediate allocation failed rc=%d size=%ux%u",
             rc, width, height);
        return false;
    }
    if (!initPresentImageLayout(g.gpuPostImage.image)) {
        destroyAhbImage(g.vk, g.gpuPostImage);
        return false;
    }
    return true;
}

bool processRealFrameIntoSlot(const AhbImage &src, const AhbImage &dst) {
    if (src.format == dst.format && src.extent.width == dst.extent.width &&
            src.extent.height == dst.extent.height)
        return copyAhbImage(src, dst);
    return blitAhbImageGpu(src, dst);
}

bool createFramegenContext() {
    if (!g.vk.supportsZeroFG) {
        LOGW("ReallyZero requires Vulkan 1.3, synchronization2 and extended storage image formats");
        return false;
    }
    const auto &input = g.inSlot[0];
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(g.vk.physicalDevice, input.format, &properties);
    constexpr VkFormatFeatureFlags required =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
        VK_FORMAT_FEATURE_BLIT_DST_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT;
    if ((properties.optimalTilingFeatures & required) != required) return false;

    const size_t totalViews = State::kInputSlotCount + g.outputs.size();
    for (size_t slot = 0; slot < totalViews; ++slot) {
        const auto &image = slot < State::kInputSlotCount
            ? g.inSlot[slot]
            : g.outputs[slot - State::kInputSlotCount];
        const VkImageViewCreateInfo view{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image.image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = image.format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        if (g.vk.fn.vkCreateImageView(g.vk.device, &view, nullptr, &g.frameViews[slot]) != VK_SUCCESS)
            return false;
    }
    const auto tryBackend = [&](zerofg::Backend backend) -> std::unique_ptr<zerofg::Interpolator> {
        zerofg::CreateInfo info{};
        info.vulkan = {g.vk.instance, g.vk.physicalDevice, g.vk.device, vkGetInstanceProcAddr};
        info.mode = zerofg::Mode::kReallyZero;
        info.backend = backend;
        info.frame_context_count = std::min<uint32_t>(
            8u, std::max<uint32_t>(1u, static_cast<uint32_t>(g.outputs.size())));
        info.capabilities = g.vk.zerofgCaps;
        zerofg::Status status{};
        auto engine = zerofg::Interpolator::Create(info, &status);
        if (engine) {
            status = engine->Resize(input.extent.width, input.extent.height,
                                    input.format, g.outputs.front().format);
        }
        if (!engine || status != zerofg::Status::kSuccess) {
            LOGW("ReallyZero initialization failed (backend=%d, status=%d)",
                 static_cast<int>(backend), static_cast<int>(status));
            return nullptr;
        }
        return engine;
    };

    auto engine = tryBackend(zerofg::Backend::kAuto);
    if (engine) {
        LOGI("ReallyZero initialized with Backend::kAuto (fp16=%d subgroup=%u cubic=%d)",
             (int)g.vk.zerofgCaps.shader_float16_enabled,
             g.vk.zerofgCaps.subgroup_size,
             (int)g.vk.zerofgCaps.filter_cubic_enabled);
    } else {
        engine = tryBackend(zerofg::Backend::kCompat);
        if (engine) {
            LOGI("ReallyZero initialized with Backend::kCompat fallback");
        }
    }
    if (!engine) return false;
    g.interpolator = std::move(engine);
    return true;
}

bool recordMidpoint(VkCommandBuffer cb, int prevSlot, int newSlot,
                    size_t outputIndex = 0, float phase = 0.5f) {
    const float safePhase = std::clamp(phase, 0.02f, 0.98f);
    const auto wrapInput = [](int slot) {
        const auto &image = g.inSlot[slot];
        return zerofg::Image{
            .image = image.image,
            .view = g.frameViews[slot],
            .layout = VK_IMAGE_LAYOUT_GENERAL,
            .format = image.format,
            .width = image.extent.width,
            .height = image.extent.height,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
            .sequence = g.inputSequences[slot],
        };
    };
    const auto &out = g.outputs[outputIndex];
    const zerofg::Image output{
        .image = out.image,
        .view = g.frameViews[State::kInputSlotCount + outputIndex],
        .layout = VK_IMAGE_LAYOUT_GENERAL,
        .format = out.format,
        .width = out.extent.width,
        .height = out.extent.height,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
    };
    const VkMemoryBarrier ready{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
    };
    g.vk.fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &ready, 0, nullptr, 0, nullptr);
    const auto status = g.interpolator->Interpolate(cb, static_cast<uint32_t>(outputIndex), wrapInput(prevSlot),
                                                   wrapInput(newSlot), safePhase, output);
    if (status != zerofg::Status::kSuccess) {
        LOGW("ReallyZero interpolation failed: %d (phase=%.3f)", static_cast<int>(status), safePhase);
        return false;
    }
    return true;
}

bool generateSingleFrame(int prevSlot, int newSlot, size_t outputIndex, float phase) {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    if (!acquireCommandRing(g.vk, cb, fence) || !beginFrameCommands(cb)) return false;
    if (!recordMidpoint(cb, prevSlot, newSlot, outputIndex, phase)) return false;
    return completeFrameCommands(cb, fence);
}

// Output blit. Has two paths:
//
//  1. GPU fast path (WSI): vkCmdBlitImage from the output AHB's VkImage
//     straight into the next swapchain image, then vkQueuePresentKHR. Zero
//     CPU touch of the pixel data. Used when the swapchain is live AND no
//     CPU post-process stage is active (NPU/CPU filters mutate pixels
//     on CPU and need the lock path). Saves ~3-5 ms/blit at 1080p.
//
//  2. CPU path (legacy): AHardwareBuffer_lock(CPU_READ) on the output,
//     ANativeWindow_lock on the window's next buffer, memcpy rows, post.
//     Still used when WSI is unavailable OR NPU/CPU post is on.
//
// GPU post-process (when enabled) runs into g.gpuPostImage via a compute
// shader first, then THAT AHB goes through either path above.
bool transferCpuOwnership(VkImage image, bool toCpu) {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    if (!acquirePresentRing(g.vk, cb, fence)) return false;
    const VkCommandBufferBeginInfo begin{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (g.vk.fn.vkBeginCommandBuffer(cb, &begin) != VK_SUCCESS) return false;
    const uint32_t local = g.vk.computeFamilyIdx;
    const uint32_t foreign = g.vk.hasQueueFamilyForeign
        ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_EXTERNAL;
    const VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = toCpu ? static_cast<VkAccessFlags>(VK_ACCESS_MEMORY_WRITE_BIT) : 0u,
        .dstAccessMask = toCpu
            ? static_cast<VkAccessFlags>(VK_ACCESS_HOST_READ_BIT)
            : static_cast<VkAccessFlags>(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = toCpu ? local : foreign,
        .dstQueueFamilyIndex = toCpu ? foreign : local,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    g.vk.fn.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        toCpu ? VK_PIPELINE_STAGE_HOST_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    if (g.vk.fn.vkEndCommandBuffer(cb) != VK_SUCCESS) return false;
    const VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    {
        std::lock_guard<std::mutex> qLock(g.queueMu);
        if (g.vk.fn.vkQueueSubmit(g.vk.computeQueue, 1, &si, fence) != VK_SUCCESS) return false;
    }
    markPresentFenceArmed(g.vk, fence);
    return g.vk.fn.vkWaitForFences(g.vk.device, 1, &fence, VK_TRUE, 500ULL * 1'000'000ULL) == VK_SUCCESS;
}

void blitOutputToWindow(const AhbImage &out, uint64_t desiredPresentNs = 0, bool allowGpuPost = true) {
    if (g.outWindow == nullptr || out.ahb == nullptr) return;

    if (allowGpuPost && gpuWantsPostLsfg()) {
        noteGpuShaderPassPending();
        const uint32_t scaledW = out.extent.width;
        const uint32_t scaledH = out.extent.height;
        const GpuPostProcessConfig gpuCfg{
            .method = g.gpuMethod.load(std::memory_order_relaxed),
            .sharpness = g.gpuSharpness.load(std::memory_order_relaxed),
            .strength = g.gpuStrength.load(std::memory_order_relaxed),
        };
        if (ensureGpuPostImage(scaledW, scaledH) &&
                g.gpuPost.process(g.vk, g.queueMu, out, g.gpuPostImage, gpuCfg)) {
            blitOutputToWindow(g.gpuPostImage, desiredPresentNs, false);
            return;
        }
        LOGW("GPU post-process failed; falling back to direct overlay blit");
    }

    // WSI fast path: skip the CPU lock/memcpy entirely. Only viable when no
    // CPU-side post-process is active (NPU or CPU filters need to read and
    // mutate pixels through ANativeWindow_lock, which can't coexist with a
    // Vulkan swapchain on the same surface).
    //
    // Build the swapchain lazily on first use — at initContext time the
    // overlay's Surface is still owned by the mirror VirtualDisplay, but by
    // the time blitOutputToWindow runs the VD has been retargeted to the
    // ImageReader by setLsfgMode and the Surface is free.
    const bool cpuPostActive = g.npuPostProcessing || g.cpuPostProcessing;
    if (kEnableWsiSwapchain && !cpuPostActive && g.vk.hasSwapchain
            && !g.swap.disabledForSession) {
        std::lock_guard<std::mutex> lock(g.mu);
        if (g.swap.swapchain == VK_NULL_HANDLE) {
            if (!createSwapchain()) {
                // Mark disabled so subsequent blits don't spin on the
                // failed creation path; use CPU blit for the rest of the
                // session.
                g.swap.disabledForSession = true;
                LOGI("WSI blit unavailable on this surface — using CPU blit for session");
            }
        }
        if (g.swap.swapchain != VK_NULL_HANDLE) {
            if (blitOutputToSwapchain(out, desiredPresentNs)) return;
            // On OUT_OF_DATE we marked swap.outOfDate; next frame triggers
            // the recreate above. No need to fall through — the missed frame
            // costs 16 ms at worst, and forcing a CPU lock now would conflict
            // with the pending-destroy swapchain.
            return;
        }
    }

    if (desiredPresentNs) {
        std::this_thread::sleep_until(State::Clock::time_point(
            std::chrono::nanoseconds(desiredPresentNs)));
    }
    std::lock_guard<std::mutex> outputLock(g.mu);
    if (g.outWindow == nullptr) return;
    const uint32_t npuScale = g.npuPostProcessing && g.npuUpscaleFactor == 2
        ? 2U
        : 1U;
    const uint32_t targetW = out.extent.width * npuScale;
    const uint32_t targetH = out.extent.height * npuScale;
    // Only call setBuffersGeometry when parameters actually change.  On some
    // drivers (MediaTek, certain Mali) calling it on every blit triggers a
    // SurfaceTexture buffer-queue reconfiguration even when nothing changed,
    // which discards already-queued frames and leaves the overlay frozen on
    // the first successfully displayed buffer.
    if (targetW != g.winGeomW || targetH != g.winGeomH || g.winGeomFmt != WINDOW_FORMAT_RGBA_8888) {
        ANativeWindow_setBuffersGeometry(g.outWindow,
            static_cast<int32_t>(targetW),
            static_cast<int32_t>(targetH),
            WINDOW_FORMAT_RGBA_8888);
        g.winGeomW   = targetW;
        g.winGeomH   = targetH;
        g.winGeomFmt = WINDOW_FORMAT_RGBA_8888;
        LOGI("blitOutputToWindow: setBuffersGeometry %ux%u fmt=RGBA8888", targetW, targetH);
    }

    // Synchronization is handled by the producer side before calling into this
    // blit path:
    // - generated outputs: LSFG_3_1::waitIdle() / LSFG_3_1P::waitIdle()
    // - raw current frame: copyAhbImage() waits for the transfer queue submit
    // Doing an extra vkDeviceWaitIdle() here stalls the whole Android-side
    // Vulkan session on every posted frame and injects visible pacing jitter.

    if (!transferCpuOwnership(out.image, true)) return;
    struct ReturnToGpu {
        VkImage image;
        ~ReturnToGpu() { transferCpuOwnership(image, false); }
    } returnToGpu{out.image};

    AHardwareBuffer_Desc desc{};
    AHardwareBuffer_describe(out.ahb, &desc);

    void *srcPtr = nullptr;
    if (AHardwareBuffer_lock(out.ahb,
            AHARDWAREBUFFER_USAGE_CPU_READ_RARELY,
            -1, nullptr, &srcPtr) != 0 || srcPtr == nullptr) {
        LOGW("AHardwareBuffer_lock(read) failed on output");
        return;
    }
    const uint32_t srcStrideBytes = desc.stride * 4;  // RGBA8888

    // Sample luma on every frame: needed for the feedback-loop gate (not just
    // the first 30 blits).  32×18 grid keeps it cheap (~576 pixels).
    uint64_t lumaSum = 0, alphaSum = 0;
    uint32_t samples = 0;
    {
        const auto *srcBytes = static_cast<const uint8_t *>(srcPtr);
        const uint32_t sW = std::min<uint32_t>(desc.width, 32);
        const uint32_t sH = std::min<uint32_t>(desc.height, 18);
        for (uint32_t sy = 0; sy < sH; ++sy) {
            const uint32_t y = sH > 1 ? (sy * (desc.height - 1)) / (sH - 1) : 0;
            for (uint32_t sx = 0; sx < sW; ++sx) {
                const uint32_t x = sW > 1 ? (sx * (desc.width - 1)) / (sW - 1) : 0;
                const uint8_t *p = srcBytes + y * srcStrideBytes + x * 4;
                lumaSum  += (77u * p[0] + 150u * p[1] + 29u * p[2]) >> 8u;
                alphaSum += p[3];
                ++samples;
            }
        }
    }
    const uint32_t avgLuma  = samples > 0 ? static_cast<uint32_t>(lumaSum  / samples) : 0;
    const uint32_t avgAlpha = samples > 0 ? static_cast<uint32_t>(alphaSum / samples) : 0;

    // Log first 30 blits.
    {
        const uint32_t idx = g.blitLogCount.fetch_add(1, std::memory_order_relaxed);
        if (idx < 30) {
            LOGI("blit #%u src=%ux%u stride=%u avgLuma=%u avgAlpha=%u outWindow=%ux%u",
                 idx + 1, desc.width, desc.height, desc.stride,
                 avgLuma, avgAlpha, g.outWidth, g.outHeight);
        }
    }

    // Feedback-loop gate: if setSkipScreenshot failed, MediaProjection captures
    // our overlay (opaque dark) instead of the game, locking avgLuma near zero.
    // Keep the overlay transparent (skip the blit) until a bright frame proves
    // we are seeing real game content. Once luma>=18 arrives the gate latches
    // open for the session so dark in-game scenes (fade-to-black, cutscenes,
    // night-mode camera apps) still get LSFG.
    //
    // The threshold was originally 30 — that's too tight for camera/photography
    // apps and dark games (e.g. com.ktzevani.nativecameravulkan averages ~25-29
    // on real content), causing the user to see ~1s of black overlay before the
    // gate opens. A real feedback loop pins luma to 0-5; anything above ~10 is
    // already dim-but-real game content and safe to forward. 18 is a comfortable
    // safety margin that still rejects the dark feedback signal.
    //
    // Additionally: force-open after 120 consecutive dark frames (~2s) regardless
    // of luma. If the user's screenshot exclusion is broken AND the game is
    // genuinely dark, holding the overlay transparent forever is worse than a
    // brief feedback flash that lets them see the game and toggle the overlay
    // off. Without this escape hatch, very dark legitimate content (true
    // black-on-black scenes) would wedge the overlay forever.
    constexpr uint32_t kLumaGateOpenThreshold = 18;
    constexpr uint32_t kLumaGateForceOpenAfter = 120;
    if (!g.lumaGateOpen) {
        if (avgLuma >= kLumaGateOpenThreshold) {
            g.lumaGateOpen = true;
            LOGI("blitOutputToWindow: luma gate opened after %u dark frames (luma=%u)",
                 g.lumaGateDarkCount, avgLuma);
        } else if (g.lumaGateDarkCount >= kLumaGateForceOpenAfter) {
            g.lumaGateOpen = true;
            LOGW("blitOutputToWindow: luma gate force-opened after %u dark frames (luma=%u) — content may be very dim or feedback loop",
                 g.lumaGateDarkCount, avgLuma);
        } else {
            ++g.lumaGateDarkCount;
            if (g.lumaGateDarkCount <= 5u || g.lumaGateDarkCount % 300u == 0u) {
                LOGI("blitOutputToWindow: dark frame suppressed (luma=%u count=%u) — overlay transparent",
                     avgLuma, g.lumaGateDarkCount);
            }
            AHardwareBuffer_unlock(out.ahb, nullptr);
            return; // keep overlay transparent — suppress dark feedback frame
        }
    }

    ANativeWindow_Buffer dst{};
    if (ANativeWindow_lock(g.outWindow, &dst, nullptr) != 0) {
        AHardwareBuffer_unlock(out.ahb, nullptr);
        LOGW("ANativeWindow_lock failed");
        return;
    }
    // The window is now (and stays) bound to a CPU producer for this session.
    // Pre-pin the taint so the next createSwapchain attempt bails fast.
    g.windowCpuProducerLocked = true;
    const uint32_t dstStrideBytes = static_cast<uint32_t>(dst.stride) * 4;

    auto *src = static_cast<const uint8_t *>(srcPtr);
    auto *dest = static_cast<uint8_t *>(dst.bits);
    if (gpuWantsPostLsfg()) {
        noteGpuShaderPassPending();
    }
    bool npuPosted = false;
    uint32_t producedW = desc.width;
    uint32_t producedH = desc.height;
    if (g.npuPostProcessing) {
        const NnapiPostProcessConfig postCfg{
            .preset = static_cast<NpuPreset>(g.npuPreset),
            .upscaleFactor = g.npuUpscaleFactor,
            .amount = g.npuAmount,
            .radius = g.npuRadius,
            .threshold = g.npuThreshold,
            .fp16 = g.npuFp16,
        };
        // configure() returns true only when a graph is compiled on a
        // dedicated NPU. If it returns false we treat this frame as no-op
        // for the NPU stage — but we don't disable the whole category
        // unless processRgba8888 actually fails.
        if (g.npuPost.configure(desc.width, desc.height, postCfg) &&
                g.npuPost.outputWidth() <= static_cast<uint32_t>(dst.width) &&
                g.npuPost.outputHeight() <= static_cast<uint32_t>(dst.height)) {
            npuPosted = g.npuPost.processRgba8888(src, srcStrideBytes, dest, dstStrideBytes);
            if (npuPosted) {
                producedW = g.npuPost.outputWidth();
                producedH = g.npuPost.outputHeight();
            }
        }
        if (!npuPosted && static_cast<NpuPreset>(g.npuPreset) != NpuPreset::OFF) {
            LOGW("NPU post-process failed; disabling NPU category for this session");
            g.npuPostProcessing = false;
            g.npuPost.reset();
        }
    }

    if (!npuPosted) {
        const uint32_t copyW = static_cast<uint32_t>(dst.width);
        const uint32_t copyH = static_cast<uint32_t>(dst.height);
        for (uint32_t y = 0; y < copyH; ++y) {
            const uint32_t srcY = static_cast<uint64_t>(y) * desc.height / copyH;
            const uint32_t *sRow = reinterpret_cast<const uint32_t *>(src + srcY * srcStrideBytes);
                  uint32_t *dRow = reinterpret_cast<      uint32_t *>(dest + y * dstStrideBytes);
            for (uint32_t x = 0; x < copyW; ++x) {
                const uint32_t srcX = static_cast<uint64_t>(x) * desc.width / copyW;
                dRow[x] = sRow[srcX] | 0xFF000000u;
            }
        }
        producedW = copyW;
        producedH = copyH;
    }

    // CPU post-process runs on the destination in-place. Scoped to the
    // region we've actually written so we never read past the blit area.
    if (g.cpuPostProcessing) {
        const CpuPostProcessConfig cpuCfg{
            .preset = static_cast<CpuPreset>(g.cpuPreset),
            .strength = g.cpuStrength,
            .saturation = g.cpuSaturation,
            .vibrance = g.cpuVibrance,
            .vignette = g.cpuVignette,
        };
        const bool active = g.cpuPost.configure(producedW, producedH, cpuCfg);
        if (active) {
            g.cpuPost.process(dest, dstStrideBytes, dest, dstStrideBytes,
                              producedW, producedH);
        }
    }

    const int postResult = ANativeWindow_unlockAndPost(g.outWindow);
    if (postResult != 0) {
        LOGW("ANativeWindow_unlockAndPost failed: %d (blit will not appear)", postResult);
    }
    AHardwareBuffer_unlock(out.ahb, nullptr);
    if (postResult != 0) return;
    // Count this as an overlay post (ground truth for HUD total-fps metric
    // and for the pacing graph's frame-interval samples).
    recordOverlayPost();
}

struct DisplayTimingSamples {
    uint32_t lastId = 0;
    uint64_t lastPresentNs = 0;
    uint64_t smallestGapNs = UINT64_MAX;
    uint64_t largestGapNs = 0;
    uint32_t intervals = 0;

    void collect() {
        std::lock_guard<std::mutex> lock(g.mu);
        if (!g.vk.hasDisplayTiming || !g.vk.fn.vkGetPastPresentationTimingGOOGLE ||
            g.swap.swapchain == VK_NULL_HANDLE) return;
        std::array<VkPastPresentationTimingGOOGLE, 64> history{};
        uint32_t count = static_cast<uint32_t>(history.size());
        const auto result = g.vk.fn.vkGetPastPresentationTimingGOOGLE(
            g.vk.device, g.swap.swapchain, &count, history.data());
        if (result != VK_SUCCESS && result != VK_INCOMPLETE) return;
        std::sort(history.begin(), history.begin() + count, [](const auto &a, const auto &b) {
            return a.presentID < b.presentID;
        });
        for (uint32_t i = 0; i < count; ++i) {
            const auto &entry = history[i];
            if (entry.presentID <= lastId || entry.actualPresentTime <= lastPresentNs) continue;
            if (lastPresentNs != 0) {
                const auto gap = entry.actualPresentTime - lastPresentNs;
                smallestGapNs = std::min(smallestGapNs, gap);
                largestGapNs = std::max(largestGapNs, gap);
                ++intervals;
            }
            lastId = entry.presentID;
            lastPresentNs = entry.actualPresentTime;
        }
    }

    void resetWindow() {
        smallestGapNs = UINT64_MAX;
        largestGapNs = 0;
        intervals = 0;
    }
};

void enqueuePresentItem(const State::PresentItem &item) {
    {
        std::lock_guard<std::mutex> lock(g.presentMu);
        // Bound queue depth so latency can never spiral if presentation stalls.
        while (g.presentQueue.size() >= 16) {
            auto dropIt = std::find_if(g.presentQueue.begin(), g.presentQueue.end(),
                [](const State::PresentItem &p) { return p.isGenerated; });
            if (dropIt == g.presentQueue.end()) dropIt = g.presentQueue.begin();
            if (dropIt->isGenerated) {
                if (g.outSlotRefCount[dropIt->slotIndex] > 0) --g.outSlotRefCount[dropIt->slotIndex];
            } else {
                if (g.inSlotRefCount[dropIt->slotIndex] > 0) --g.inSlotRefCount[dropIt->slotIndex];
            }
            g.presentQueue.erase(dropIt);
        }
        if (item.isGenerated) {
            ++g.outSlotRefCount[item.slotIndex];
        } else {
            ++g.inSlotRefCount[item.slotIndex];
        }
        g.presentQueue.push_back(item);
    }
    g.presentCv.notify_one();
}

void reclaimSlotIfBusy(bool isGenerated, size_t slotIndex) {
    std::unique_lock<std::mutex> lock(g.presentMu);
    auto &refCount = isGenerated ? g.outSlotRefCount[slotIndex] : g.inSlotRefCount[slotIndex];
    if (refCount == 0) return;

    // First evict any unpresented items still waiting in the queue for this slot.
    for (auto it = g.presentQueue.begin(); it != g.presentQueue.end(); ) {
        if (it->isGenerated == isGenerated && it->slotIndex == slotIndex) {
            if (refCount > 0) --refCount;
            it = g.presentQueue.erase(it);
        } else {
            ++it;
        }
    }
    if (refCount == 0) return;

    // If refCount > 0 now, presentThread is actively blitting this exact slot.
    // Wait briefly until the blit finishes (typically < 1 ms).
    g.presentCv.wait_for(lock, std::chrono::milliseconds(100), [&] {
        return refCount == 0 || g.presentStopRequested;
    });
}

void presentThread() {
    DisplayTimingSamples displayTiming;
    auto lastPostedAt = State::Clock::now();
    uint64_t presentCount = 0;

    while (true) {
        State::PresentItem item{};
        {
            std::unique_lock<std::mutex> lock(g.presentMu);
            g.presentCv.wait(lock, [] {
                return g.presentStopRequested || !g.presentQueue.empty();
            });
            if (g.presentStopRequested) return;
            item = g.presentQueue.front();
            g.presentQueue.pop_front();
        }

        struct ReleaseSlot {
            bool isGenerated;
            size_t slotIndex;
            ~ReleaseSlot() {
                {
                    std::lock_guard<std::mutex> lock(g.presentMu);
                    auto &rc = isGenerated ? g.outSlotRefCount[slotIndex] : g.inSlotRefCount[slotIndex];
                    if (rc > 0) --rc;
                }
                g.presentCv.notify_all();
            }
        } releaseSlot{item.isGenerated, item.slotIndex};

        const auto now = State::Clock::now();
        const int64_t nowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();

        if (item.isGenerated && item.latestPresentNs > 0 && nowNs > item.latestPresentNs) {
            g.lateMidpointsTotal.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        int64_t refreshNs = g.vsyncPeriodNs.load(std::memory_order_relaxed);
        if (refreshNs <= 0) refreshNs = 16'666'667;

        uint64_t desiredPresentArg = 0;
        if (item.desiredPresentNs > 0) {
            if (item.useDisplayTiming) {
                // Keep swapchain FIFO from over-queueing far ahead while letting
                // SurfaceFlinger hold the frame until desiredPresentTime.
                const int64_t wakeAtNs = item.desiredPresentNs - (refreshNs * 3) / 2;
                if (wakeAtNs > nowNs) {
                    std::this_thread::sleep_until(State::Clock::time_point(
                        std::chrono::nanoseconds(wakeAtNs)));
                }
                desiredPresentArg = static_cast<uint64_t>(item.desiredPresentNs);
            } else if (item.fallbackSleepBudgetNs > 0) {
                const auto deadline = State::Clock::time_point(
                    std::chrono::nanoseconds(item.desiredPresentNs));
                sleepUntilVsyncAligned(deadline, lastPostedAt,
                    std::chrono::nanoseconds(item.fallbackSleepBudgetNs));
            } else if (item.desiredPresentNs > nowNs) {
                std::this_thread::sleep_until(State::Clock::time_point(
                    std::chrono::nanoseconds(item.desiredPresentNs)));
            }
        }

        const AhbImage &img = item.isGenerated
            ? g.outputs[item.slotIndex]
            : g.inSlot[item.slotIndex];

        const uint64_t beforePosts = g.postedFrames.load(std::memory_order_relaxed);
        const auto t0 = State::Clock::now();
        blitOutputToWindow(img, desiredPresentArg);
        const auto t1 = State::Clock::now();

        if (g.postedFrames.load(std::memory_order_relaxed) > beforePosts) {
            lastPostedAt = t1;
            if (item.isGenerated) {
                g.generatedFrames.fetch_add(1, std::memory_order_relaxed);
            }
            const int64_t blitNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            g.presentBlitNsAccum.fetch_add(static_cast<uint64_t>(blitNs), std::memory_order_relaxed);
            g.presentBlitCountAccum.fetch_add(1, std::memory_order_relaxed);
            const uint64_t postTimeNs = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1.time_since_epoch()).count());
            const uint64_t estPresentNs = std::max(postTimeNs, desiredPresentArg);
            if (item.captureTimestampNs > 0 && estPresentNs > static_cast<uint64_t>(item.captureTimestampNs)) {
                g.presentCaptureToDisplayNsAccum.fetch_add(
                    estPresentNs - static_cast<uint64_t>(item.captureTimestampNs),
                    std::memory_order_relaxed);
            }
            if ((++presentCount % 12) == 0) {
                displayTiming.collect();
                if (displayTiming.lastPresentNs > 0) {
                    g.lastActualPresentNs.store(displayTiming.lastPresentNs, std::memory_order_relaxed);
                }
            }
        }
    }
}

void workerThread() {
    CaptureImports captureImports;
    AnimeOutputGrid animeGrid;
    int64_t prevCaptureTimestampNs = 0;
    bool havePrevCaptureTimestamp = false;
    bool previousAnimeMode = false;
    size_t nextOutputSlot = 0;
    uint32_t consecutiveFramegenErrors = 0;

    double emaNs = 0.0;
    bool emaInit = false;
    int consecutiveOutliers = 0;

    struct ProfileAccum {
        int64_t copyNs    = 0;
        int64_t presentNs = 0;
        int64_t waitIdleNs= 0;
        int64_t totalNs   = 0;
        int64_t queueNs   = 0;
        uint32_t samples  = 0;
        uint32_t gridPairs = 0;
        uint32_t generatedPosts = 0;
        uint32_t realPosts = 0;
    } prof{};
    constexpr uint32_t kProfileWindow = 60;

    while (true) {
        State::PendingFrame pendingFrame{};
        {
            std::unique_lock<std::mutex> lock(g.mu);
            g.pendingCv.wait(lock, []{ return g.stopRequested || !g.pending.empty(); });
            if (g.stopRequested) return;
            pendingFrame = g.pending.front();
            g.pending.pop_front();
        }
        const auto frameWorkStartedAt = State::Clock::now();
        prof.queueNs += std::chrono::duration_cast<std::chrono::nanoseconds>(
            frameWorkStartedAt - pendingFrame.queuedAt).count();
        AHardwareBuffer *ahb = pendingFrame.ahb;
        if (ahb == nullptr) {
            pendingFrame.copied->set_value();
            continue;
        }

        AhbImage temporaryImport{};
        const AhbImage *imported = captureImports.findOrImport(ahb, temporaryImport);
        if (!imported) {
            AHardwareBuffer_release(ahb);
            pendingFrame.copied->set_value();
            continue;
        }
        const AhbImage &src = *imported;

        const int newSlot = static_cast<int>(g.framesCopied % State::kInputSlotCount);
        const int prevSlot = static_cast<int>((g.framesCopied + State::kInputSlotCount - 1) % State::kInputSlotCount);

        reclaimSlotIfBusy(false, static_cast<size_t>(newSlot));

        // Step 1: Fast GPU-only copy into inSlot[newSlot], then IMMEDIATELY
        // release the ImageReader AHB so capture never stalls behind ZeroFG or
        // presentation pacing sleeps!
        const bool copied = processRealFrameIntoSlot(src, g.inSlot[newSlot]);
        destroyAhbImage(g.vk, temporaryImport);
        AHardwareBuffer_release(ahb);
        pendingFrame.copied->set_value();
        if (!copied) continue;

        const auto tCopyDone = State::Clock::now();
        const bool contiguous = g.framesCopied > 0 &&
            g.inputSequences[prevSlot] + 1 == pendingFrame.sequence;
        g.inputSequences[newSlot] = pendingFrame.sequence;
        ++g.framesCopied;

        const int64_t delta = pendingFrame.captureTimestampNs - prevCaptureTimestampNs;
        const auto decision = pendingFrame.animeDecision;
        const bool validGap = pendingFrame.animeMode
            ? canInterpolateAnimePair(delta, decision.allowHeldGap)
            : (delta > 0 && delta <= 250'000'000);
        const bool runFramegen = g.interpolator && contiguous &&
            !g.outputs.empty() &&
            previousAnimeMode == pendingFrame.animeMode &&
            !decision.breakBefore &&
            validGap &&
            !g.bypass.load(std::memory_order_relaxed) &&
            !g.framegenAutoDisabled.load(std::memory_order_relaxed);
        previousAnimeMode = pendingFrame.animeMode;

        bool canDisplayTiming = false;
        {
            std::lock_guard<std::mutex> lock(g.mu);
            canDisplayTiming = g.vk.hasDisplayTiming &&
                g.swap.swapchain != VK_NULL_HANDLE && !g.swap.outOfDate;
        }
        int64_t refreshNs = g.vsyncPeriodNs.load(std::memory_order_relaxed);
        if (refreshNs <= 0) refreshNs = 16'666'667;

        bool anyGenerated = false;
        if (runFramegen && pendingFrame.animeMode) {
            const int64_t lastPresent = static_cast<int64_t>(
                g.lastActualPresentNs.load(std::memory_order_relaxed));
            const int64_t anchor = lastPresent > 0 &&
                std::abs(lastPresent - pendingFrame.captureTimestampNs) < 200'000'000
                ? lastPresent : 0;
            const int animeMult = std::clamp(
                g.targetMultiplier.load(std::memory_order_relaxed), 2, 4);
            const AnimeGridPlan grid = animeGrid.schedule(
                prevCaptureTimestampNs, pendingFrame.captureTimestampNs,
                refreshNs, anchor, decision.allowHeldGap, animeMult);

            if (grid.count > 0) {
                ++prof.gridPairs;
                bool enqueuedAny = false;
                for (size_t i = 0; i < grid.count; ++i) {
                    const auto &sample = grid.frames[i];
                    const int64_t curNowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        State::Clock::now().time_since_epoch()).count();
                    if (sample.isReal()) {
                        enqueuePresentItem(State::PresentItem{
                            .isGenerated = false,
                            .slotIndex = static_cast<size_t>(newSlot),
                            .desiredPresentNs = sample.presentationNs,
                            .latestPresentNs = 0,
                            .captureTimestampNs = pendingFrame.captureTimestampNs,
                            .useDisplayTiming = canDisplayTiming,
                        });
                        ++prof.realPosts;
                        enqueuedAny = true;
                    } else if (sample.isPreviousReal()) {
                        enqueuePresentItem(State::PresentItem{
                            .isGenerated = false,
                            .slotIndex = static_cast<size_t>(prevSlot),
                            .desiredPresentNs = sample.presentationNs,
                            .latestPresentNs = 0,
                            .captureTimestampNs = prevCaptureTimestampNs,
                            .useDisplayTiming = canDisplayTiming,
                        });
                        ++prof.realPosts;
                        enqueuedAny = true;
                    } else {
                        // Do not waste GPU compute on a synthetic slot that is already late.
                        if (curNowNs > sample.presentationNs) {
                            g.lateMidpointsTotal.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        const size_t outIdx = nextOutputSlot++ % g.outputs.size();
                        reclaimSlotIfBusy(true, outIdx);
                        if (generateSingleFrame(prevSlot, newSlot, outIdx, sample.phase)) {
                            consecutiveFramegenErrors = 0;
                            anyGenerated = true;
                            enqueuedAny = true;
                            ++prof.generatedPosts;
                            enqueuePresentItem(State::PresentItem{
                                .isGenerated = true,
                                .slotIndex = outIdx,
                                .desiredPresentNs = sample.presentationNs,
                                .latestPresentNs = sample.presentationNs + refreshNs,
                                .captureTimestampNs = pendingFrame.captureTimestampNs,
                                .useDisplayTiming = canDisplayTiming,
                            });
                        } else {
                            if (++consecutiveFramegenErrors >= 5) {
                                LOGW("ReallyZero failed %u consecutive times — disabling framegen for session",
                                     consecutiveFramegenErrors);
                                g.framegenAutoDisabled.store(true, std::memory_order_relaxed);
                                break;
                            }
                        }
                    }
                }
                if (!enqueuedAny) {
                    enqueuePresentItem(State::PresentItem{
                        .isGenerated = false,
                        .slotIndex = static_cast<size_t>(newSlot),
                        .desiredPresentNs = 0,
                        .captureTimestampNs = pendingFrame.captureTimestampNs,
                        .useDisplayTiming = false,
                    });
                    ++prof.realPosts;
                }
            } else {
                // Fallback inside anime mode if delta < 22ms: emit real frame directly.
                enqueuePresentItem(State::PresentItem{
                    .isGenerated = false,
                    .slotIndex = static_cast<size_t>(newSlot),
                    .desiredPresentNs = 0,
                    .captureTimestampNs = pendingFrame.captureTimestampNs,
                    .useDisplayTiming = false,
                });
                ++prof.realPosts;
            }
        } else if (runFramegen) {
            animeGrid.reset();

            // EMA-smoothed capture interval for normal/game mode pacing.
            State::Clock::duration captureInterval = std::chrono::milliseconds(16);
            const ShizukuTimingSample shizuku = loadShizukuTimingSample();
            const bool haveShizukuCadence =
                shizuku.enabled &&
                shizuku.frameTimeNs > 0 &&
                shizuku.timestampNs >= pendingFrame.captureTimestampNs;
            const bool haveCaptureDelta =
                havePrevCaptureTimestamp &&
                pendingFrame.captureTimestampNs > prevCaptureTimestampNs;
            if (haveShizukuCadence || haveCaptureDelta) {
                constexpr double minNs = 8.0 * 1'000'000.0;
                constexpr double absMaxNs = 1000.0 * 1'000'000.0;
                double rawNs = haveCaptureDelta
                    ? static_cast<double>(pendingFrame.captureTimestampNs - prevCaptureTimestampNs)
                    : static_cast<double>(shizuku.frameTimeNs);
                if (rawNs < minNs) rawNs = minNs;
                if (!emaInit) {
                    emaNs = rawNs;
                    emaInit = true;
                    consecutiveOutliers = 0;
                } else {
                    const double kOutlier = static_cast<double>(
                        g.outlierRatioMicro.load(std::memory_order_relaxed)) / 1'000'000.0;
                    const bool outlier = rawNs > kOutlier * emaNs || rawNs * kOutlier < emaNs;
                    if (outlier) {
                        if (++consecutiveOutliers >= 3) {
                            emaNs = rawNs;
                            consecutiveOutliers = 0;
                        }
                    } else {
                        const double alpha = static_cast<double>(
                            g.emaAlphaMicro.load(std::memory_order_relaxed)) / 1'000'000.0;
                        emaNs = alpha * rawNs + (1.0 - alpha) * emaNs;
                        consecutiveOutliers = 0;
                    }
                }
                double dynMaxNs = std::min(2.0 * emaNs, absMaxNs);
                if (rawNs > dynMaxNs) rawNs = dynMaxNs;
                captureInterval = std::chrono::nanoseconds(static_cast<int64_t>(rawNs));
            }

            int64_t budgetNs = std::chrono::duration_cast<std::chrono::nanoseconds>(captureInterval).count();
            // ZeroFG 1.2.0 x2/x3/x4: cap factor by what the display refresh rate
            // can actually show so a refresh drop never queues unshowable frames.
            const int desiredMult = std::clamp(
                g.targetMultiplier.load(std::memory_order_relaxed), 2, 4);
            const int maxByRefresh = static_cast<int>(std::floor(
                (static_cast<double>(budgetNs) / static_cast<double>(refreshNs)) + 0.15));
            const int effectiveMult = std::clamp(
                desiredMult, 2, std::clamp(maxByRefresh, 2, 4));
            const int capHz = g.targetFpsCapHz.load(std::memory_order_relaxed);
            if (capHz > 0) {
                const int64_t capFloorNs = static_cast<int64_t>(
                    (static_cast<double>(effectiveMult) * 1'000'000'000.0) / static_cast<double>(capHz));
                if (budgetNs < capFloorNs) budgetNs = capFloorNs;
            }

            const int syntheticCount = effectiveMult - 1;
            const int64_t slotStepNs = budgetNs / effectiveMult;
            const int64_t pairStartNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                State::Clock::now().time_since_epoch()).count();
            bool generatedAny = false;
            for (int step = 1; step <= syntheticCount; ++step) {
                const float phase = static_cast<float>(step) / static_cast<float>(effectiveMult);
                const int64_t targetNs = pairStartNs + static_cast<int64_t>(step - 1) * slotStepNs;
                const int64_t latestNs = pairStartNs + static_cast<int64_t>(step) * slotStepNs;
                const int64_t curNowNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    State::Clock::now().time_since_epoch()).count();
                if (step > 1 && curNowNs > latestNs) {
                    g.lateMidpointsTotal.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                const size_t outIdx = nextOutputSlot++ % g.outputs.size();
                reclaimSlotIfBusy(true, outIdx);
                if (generateSingleFrame(prevSlot, newSlot, outIdx, phase)) {
                    consecutiveFramegenErrors = 0;
                    generatedAny = true;
                    anyGenerated = true;
                    enqueuePresentItem(State::PresentItem{
                        .isGenerated = true,
                        .slotIndex = outIdx,
                        .desiredPresentNs = targetNs,
                        .latestPresentNs = latestNs,
                        .captureTimestampNs = pendingFrame.captureTimestampNs,
                        .useDisplayTiming = canDisplayTiming,
                        .fallbackSleepBudgetNs = slotStepNs,
                    });
                    ++prof.generatedPosts;
                } else {
                    if (++consecutiveFramegenErrors >= 5) {
                        LOGW("ReallyZero failed %u consecutive times — disabling framegen for session",
                             consecutiveFramegenErrors);
                        g.framegenAutoDisabled.store(true, std::memory_order_relaxed);
                    }
                    break;
                }
            }

            enqueuePresentItem(State::PresentItem{
                .isGenerated = false,
                .slotIndex = static_cast<size_t>(newSlot),
                .desiredPresentNs = generatedAny ? (pairStartNs + static_cast<int64_t>(syntheticCount) * slotStepNs) : 0,
                .latestPresentNs = 0,
                .captureTimestampNs = pendingFrame.captureTimestampNs,
                .useDisplayTiming = generatedAny && canDisplayTiming,
                .fallbackSleepBudgetNs = generatedAny ? slotStepNs : 0,
            });
            ++prof.realPosts;
        } else {
            animeGrid.reset();
            enqueuePresentItem(State::PresentItem{
                .isGenerated = false,
                .slotIndex = static_cast<size_t>(newSlot),
                .desiredPresentNs = 0,
                .captureTimestampNs = pendingFrame.captureTimestampNs,
                .useDisplayTiming = false,
            });
            ++prof.realPosts;
        }

        prevCaptureTimestampNs = pendingFrame.captureTimestampNs;
        havePrevCaptureTimestamp = prevCaptureTimestampNs > 0;

        const auto tFrameEnd = State::Clock::now();
        using ns = std::chrono::nanoseconds;
        prof.copyNs     += std::chrono::duration_cast<ns>(tCopyDone - frameWorkStartedAt).count();
        prof.waitIdleNs += anyGenerated ? std::chrono::duration_cast<ns>(tFrameEnd - tCopyDone).count() : 0;
        prof.totalNs    += std::chrono::duration_cast<ns>(tFrameEnd - frameWorkStartedAt).count();
        prof.samples++;
        if (prof.samples >= kProfileWindow) {
            const double n = static_cast<double>(prof.samples);
            const int64_t blitNsSum = static_cast<int64_t>(
                g.presentBlitNsAccum.exchange(0, std::memory_order_relaxed));
            const int64_t latNsSum = static_cast<int64_t>(
                g.presentCaptureToDisplayNsAccum.exchange(0, std::memory_order_relaxed));
            const int64_t blitCnt = static_cast<int64_t>(
                g.presentBlitCountAccum.exchange(0, std::memory_order_relaxed));
            const uint64_t lateMid = g.lateMidpointsTotal.exchange(0, std::memory_order_relaxed);
            const uint64_t hits = g.cacheHits.exchange(0, std::memory_order_relaxed);
            const uint64_t misses = g.cacheMisses.exchange(0, std::memory_order_relaxed);
            LOGW("frame profile (avg over %u): copy=%.2fms compute=%.2fms blitWork=%.2fms worker=%.2fms queue=%.2fms latency=%.2fms cache_hits=%llu cache_misses=%llu real=%u generated=%u late=%llu anime_held_total=%llu grid=%u",
                 prof.samples,
                 (prof.copyNs / n) / 1'000'000.0,
                 (prof.waitIdleNs / n) / 1'000'000.0,
                 blitCnt > 0 ? (blitNsSum / static_cast<double>(blitCnt)) / 1'000'000.0 : 0.0,
                 (prof.totalNs / n) / 1'000'000.0,
                 (prof.queueNs / n) / 1'000'000.0,
                 blitCnt > 0 ? (latNsSum / static_cast<double>(blitCnt)) / 1'000'000.0 : 0.0,
                 static_cast<unsigned long long>(hits),
                 static_cast<unsigned long long>(misses),
                 prof.realPosts, prof.generatedPosts,
                 static_cast<unsigned long long>(lateMid),
                 static_cast<unsigned long long>(g.animeDuplicates.load(std::memory_order_relaxed)),
                 prof.gridPairs);
            g.profileSnapshotCopyNs.store(prof.copyNs, std::memory_order_relaxed);
            g.profileSnapshotPresentNs.store(prof.presentNs, std::memory_order_relaxed);
            g.profileSnapshotWaitIdleNs.store(prof.waitIdleNs, std::memory_order_relaxed);
            g.profileSnapshotBlitNs.store(blitNsSum, std::memory_order_relaxed);
            g.profileSnapshotTotalNs.store(prof.totalNs, std::memory_order_relaxed);
            g.profileSnapshotQueueNs.store(prof.queueNs, std::memory_order_relaxed);
            g.profileSnapshotLatencyNs.store(latNsSum, std::memory_order_relaxed);
            g.profileSnapshotBlitCount.store(blitCnt, std::memory_order_relaxed);
            g.profileSnapshotSamples.store(prof.samples, std::memory_order_relaxed);
            prof = ProfileAccum{};
        }
    }
}

} // namespace

int initRenderLoop(const RenderLoopConfig &cfg) {
    std::unique_lock<std::mutex> lock(g.mu);
    if (g.initialized) return kRenderLoopAlreadyInit;

    // multiplier in the prefs is the *total* output rate factor (2x = 60fps from 30fps);
    // framegen's generationCount is how many *extra* frames to interpolate per input
    // pair. So generationCount = multiplier - 1, and we allocate that many output AHBs.
    // (Mirrors lsfg-vk-android/src/context.cpp:80-81,101.)
    const int totalMult = std::clamp(cfg.multiplier, 2, 4);
    g.targetMultiplier.store(totalMult, std::memory_order_relaxed);
    g.multiplier = totalMult - 1;  // generationCount = N extra frames per pair
    g.performanceMode = cfg.performance;
    g.framegenFp16 = cfg.framegenFp16;
    g.hdr = cfg.hdr;
    g.antiArtifacts.store(cfg.antiArtifacts, std::memory_order_relaxed);
    const bool requestedNpu = false;
    g.npuPreset = cfg.npuPreset < 0 ? 0 : (cfg.npuPreset > 4 ? 0 : cfg.npuPreset);
    g.npuUpscaleFactor = cfg.npuUpscaleFactor == 2 ? 2 : 1;
    g.npuAmount = clamp01(cfg.npuAmount);
    g.npuRadius = cfg.npuRadius < 0.5f ? 0.5f : (cfg.npuRadius > 2.0f ? 2.0f : cfg.npuRadius);
    g.npuThreshold = clamp01(cfg.npuThreshold);
    g.npuFp16 = cfg.npuFp16;
    // Default: if the user toggled the NPU category on but left the preset at
    // "Off", fall back to SHARPEN so something visible happens. Without this
    // the UI silently had no effect until the user scrolled down and picked a
    // preset, which read as "toggle does nothing".
    if (requestedNpu && g.npuPreset == 0 && g.npuUpscaleFactor != 2) {
        g.npuPreset = 1; // SHARPEN
    }
    const bool npuWantsWork = g.npuPreset != 0 || g.npuUpscaleFactor == 2;
    const bool npuAvailable = requestedNpu && npuWantsWork &&
        !nnapi_npu_accelerator_devices().empty();
    g.npuPostProcessing = npuAvailable;
    g.cpuPostProcessing = false;
    g.cpuPreset = cfg.cpuPreset < 0 ? 0 : (cfg.cpuPreset > 6 ? 0 : cfg.cpuPreset);
    g.cpuStrength = clamp01(cfg.cpuStrength);
    g.cpuSaturation = clamp01(cfg.cpuSaturation);
    g.cpuVibrance = clamp01(cfg.cpuVibrance);
    g.cpuVignette = clamp01(cfg.cpuVignette);
    g.gpuPostProcessing.store(cfg.gpuPostProcessing, std::memory_order_relaxed);
    g.gpuStage = 1;
    g.gpuMethod.store(std::clamp(cfg.gpuMethod, 0, 16), std::memory_order_relaxed);
    g.gpuUpscaleFactor = 1.0f;
    g.gpuSharpness.store(clamp01(cfg.gpuSharpness), std::memory_order_relaxed);
    g.gpuStrength.store(clamp01(cfg.gpuStrength), std::memory_order_relaxed);
    g.gpuNoopLogged.store(false, std::memory_order_relaxed);
    if (g.gpuPostProcessing.load(std::memory_order_relaxed)) {
        LOGI("GPU SR configured: method=%s sharp=%.2f strength=%.2f",
             gpuMethodName(g.gpuMethod.load(std::memory_order_relaxed)),
             g.gpuSharpness.load(std::memory_order_relaxed),
             g.gpuStrength.load(std::memory_order_relaxed));
    }
    if (requestedNpu && !npuAvailable) {
        LOGW("NPU post-processing requested but no dedicated NNAPI accelerator is available. NPU category disabled.");
    } else if (npuAvailable) {
        LOGI("NPU post-processing enabled: preset=%d upscale=%dx amount=%.2f radius=%.2f fp16=%d devices=%s",
             g.npuPreset, g.npuUpscaleFactor, g.npuAmount, g.npuRadius, (int)g.npuFp16,
             nnapi_npu_summary().c_str());
    }
    if (g.cpuPostProcessing) {
        LOGI("CPU post-processing enabled: preset=%d strength=%.2f sat=%.2f vibrance=%.2f vignette=%.2f",
             g.cpuPreset, g.cpuStrength, g.cpuSaturation, g.cpuVibrance, g.cpuVignette);
    }
    // flowScale on the prefs slider is "0.25..1.0" in user-friendly form, but
    // framegen wants the reciprocal (Linux passes 1.0f / conf.flowScale at
    // src/context.cpp:101). Larger user value = finer flow grid = better quality.
    const float userFlow = (cfg.flowScale >= 0.25f && cfg.flowScale <= 1.0f) ? cfg.flowScale : 1.0f;
    g.flowScale = 1.0f / userFlow;
    applyPacingParamsImpl(cfg.targetFpsCap, cfg.emaAlpha, cfg.outlierRatio,
                          cfg.vsyncSlackMs, cfg.queueDepth);
    LOGI("Pacing: cap=%dHz alpha=%.3f outlier=%.2f slack=%.1fms queue=%d",
         g.targetFpsCapHz.load(std::memory_order_relaxed),
         static_cast<double>(g.emaAlphaMicro.load(std::memory_order_relaxed)) / 1'000'000.0,
         static_cast<double>(g.outlierRatioMicro.load(std::memory_order_relaxed)) / 1'000'000.0,
         static_cast<double>(g.vsyncSlackNs.load(std::memory_order_relaxed)) / 1'000'000.0,
         g.queueDepth.load(std::memory_order_relaxed));
    g.generatedFrames.store(0, std::memory_order_relaxed);
    g.postedFrames.store(0, std::memory_order_relaxed);
    g.uniqueCaptures.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> hashLock(g.captureHashMu);
        g.lastCaptureHash = 0;
        g.lastCaptureHashValid = false;
        g.animeCadence.reset();
        g.animeDuplicates.store(0, std::memory_order_relaxed);
    }
    g.postRingHead.store(0, std::memory_order_relaxed);
    for (auto &slot : g.postRingTimestamps) {
        slot.store(0, std::memory_order_relaxed);
    }
    g.cacheHits.store(0, std::memory_order_relaxed);
    g.cacheMisses.store(0, std::memory_order_relaxed);
    g.pushLogCount.store(0, std::memory_order_relaxed);
    g.blitLogCount.store(0, std::memory_order_relaxed);
    g.lumaGateOpen      = false;
    g.lumaGateDarkCount = 0;
    g.lumaGateStartNs   = 0;
    g.shizukuSampleTimestampNs.store(0, std::memory_order_relaxed);
    g.shizukuFrameTimeNs.store(0, std::memory_order_relaxed);
    g.shizukuPacingJitterNs.store(0, std::memory_order_relaxed);
    g.framesCopied = 0;
    g.nextSequence = 0;
    for (auto &seq : g.inputSequences) seq = 0;
    g.lateMidpointsTotal.store(0, std::memory_order_relaxed);
    g.presentBlitNsAccum.store(0, std::memory_order_relaxed);
    g.presentCaptureToDisplayNsAccum.store(0, std::memory_order_relaxed);
    g.presentBlitCountAccum.store(0, std::memory_order_relaxed);
    g.lastActualPresentNs.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> pLock(g.presentMu);
        g.presentQueue.clear();
        g.inSlotRefCount.fill(0);
        g.outSlotRefCount.fill(0);
        g.presentStopRequested = false;
    }
    // Don't reset g.bypass — the user toggle should persist across re-inits
    // (e.g. when they change multiplier while bypass is on, the new context
    // should also start in bypass).
    // DO reset the framegen auto-disable latch — that flag tracks a driver
    // error tied to the previous device handle, which is being recreated here.
    // Carrying it forward would silently keep framegen off forever after one
    // bad submit, even on a healthy new device.
    g.framegenAutoDisabled.store(false, std::memory_order_relaxed);

    int rc = create_session(g.vk);
    if (rc != kOk) {
        LOGE("create_session failed rc=%d", rc);
        destroy_session(g.vk);
        return kRenderLoopSessionFailed;
    }

    const uint32_t renderW = cfg.width;
    const uint32_t renderH = cfg.height;

    const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
    for (size_t i = 0; i < State::kInputSlotCount; ++i) {
        rc = createAhbImage(g.vk, renderW, renderH, fmt, g.inSlot[i]);
        if (rc != kOk) {
            LOGE("createAhbImage(input %zu) failed rc=%d", i, rc);
            lock.unlock();
            shutdownRenderLoop();
            return kRenderLoopBufferAlloc;
        }
        initInSlotImageLayout(g.inSlot[i].image);
    }
    // Allocate a ring of output buffers so the decoupled present thread can
    // display generated frames while the worker computes the next pair (including
    // multi-phase on-2s / on-3s anime intervals).
    const size_t desiredOutputs = State::kOutputSlotCount;
    for (size_t i = 0; i < desiredOutputs; ++i) {
        AhbImage output{};
        rc = createAhbImage(g.vk, renderW, renderH, fmt, output);
        if (rc != kOk) {
            if (!g.outputs.empty()) break;
            LOGE("createAhbImage(output) failed rc=%d", rc);
            lock.unlock();
            shutdownRenderLoop();
            return kRenderLoopBufferAlloc;
        }
        initInSlotImageLayout(output.image);
        g.outputs.push_back(output);
    }

    const bool framegenReady = createFramegenContext();

    g.initialized = true;

    g.stopRequested = false;
    g.presentWorker = std::thread(presentThread);
    g.worker = std::thread(workerThread);
    // Do NOT build the swapchain here. At initContext time the overlay's
    // Surface is still owned by the mirror VirtualDisplay producer — it's
    // only detached when setLsfgMode() runs (which retargets the VD to the
    // ImageReader). Creating the swapchain before that point races against
    // ANativeWindow's single-producer rule. The first blitOutputToWindow
    // call after LSFG mode is live builds it lazily.
    LOGW("Render loop initialised: capture=%ux%u render=%ux%u totalMult=%dx (gen=%d extra) flowScale=%.2f(internal=%.2f) hdr=%d perf=%d npu=%d cpu=%d gpu=%d ctxId=%d",
         cfg.width, cfg.height, renderW, renderH, totalMult, g.multiplier,
         userFlow, g.flowScale,
         (int)g.hdr, (int)g.performanceMode, (int)g.npuPostProcessing,
         (int)g.cpuPostProcessing, (int)g.gpuPostProcessing.load(std::memory_order_relaxed), framegenReady ? 0 : -1);
    // Tell the caller whether framegen is actually running. If not, Kotlin
    // will keep the overlay up in mirror mode instead of routing the capture
    // through a dead LSFG context.
    return framegenReady ? kOk : kRenderLoopFramegenDisabled;
}

void setOutputSurface(ANativeWindow *win, uint32_t w, uint32_t h) {
    std::lock_guard<std::mutex> lock(g.mu);
    // Always tear down any prior swapchain first — it holds a VkSurfaceKHR
    // which holds an ANativeWindow reference, and the spec requires the
    // surface outlive the swapchain but be destroyed before the window.
    destroySwapchain();
    if (g.outWindow != nullptr) {
        ANativeWindow_release(g.outWindow);
        g.outWindow = nullptr;
    }
    if (win != nullptr) {
        ANativeWindow_acquire(win);
        g.outWindow = win;
        g.outWidth = w;
        g.outHeight = h;
        g.swapWinW = w;
        g.swapWinH = h;
        // Fresh ANativeWindow — its BufferQueue producer slot is empty until
        // the first ANativeWindow_lock or vkCreateAndroidSurfaceKHR succeeds.
        // Reset the taint so a new window can take the WSI fast path even if
        // a previous one was poisoned for CPU.
        g.windowCpuProducerLocked = false;
        // WSI swapchain is only useful when no CPU-side post-process is
        // running. If NPU or CPU filters are on, they need to read/write the
        // pixels on CPU and the CPU blit path handles that; switching to the
        // swapchain would break them because ANativeWindow_lock can't be
        // mixed with Vulkan-side presents on the same surface.
        const bool cpuPostActive = g.npuPostProcessing || g.cpuPostProcessing;
        if (kEnableWsiSwapchain && !cpuPostActive && g.initialized) {
            if (createSwapchain()) {
                LOGW("Output surface attached %ux%u native=%p path=WSI", w, h, static_cast<void *>(win));
            } else {
                LOGW("Output surface attached %ux%u native=%p path=CPU (swapchain unavailable)", w, h, static_cast<void *>(win));
            }
        } else {
            const char *why = !kEnableWsiSwapchain ? "WSI disabled"
                            : cpuPostActive         ? "post-process=on"
                            : !g.initialized        ? "pre-init"
                                                    : "unknown";
            LOGW("Output surface attached %ux%u native=%p path=CPU (%s)",
                 w, h, static_cast<void *>(win), why);
        }
    } else {
        g.outWidth = 0;
        g.outHeight = 0;
        g.swapWinW = 0;
        g.swapWinH = 0;
        LOGW("Output surface detached");
    }
}

void pushFrame(AHardwareBuffer *ahb, int64_t timestampNs) {
    if (ahb == nullptr) return;
    const uint32_t pushLogIndex = g.pushLogCount.load(std::memory_order_relaxed);
    if (pushLogIndex < 30) {
        AHardwareBuffer_Desc desc{};
        AHardwareBuffer_describe(ahb, &desc);
        if (g.pushLogCount.fetch_add(1, std::memory_order_relaxed) < 30) {
            LOGW("pushFrame #%u ahb=%ux%u stride=%u fmt=%u usage=0x%llx ts=%lld",
                 pushLogIndex + 1, desc.width, desc.height, desc.stride, desc.format,
                 static_cast<unsigned long long>(desc.usage),
                 static_cast<long long>(timestampNs));
        }
    }

    AnimeFrameDecision animeDecision;
    bool animeMode;
    {
        std::lock_guard<std::mutex> hashLock(g.captureHashMu);
        animeMode = g.animeMode;
        if (animeMode) {
            animeDecision = g.animeCadence.observe(captureAnimeSignature(ahb), timestampNs);
            if (animeDecision.duplicate) {
                g.animeDuplicates.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        } else {
            const uint32_t hash = captureContentHash(ahb);
            if (hash != 0u) {
                if (g.lastCaptureHashValid && hash == g.lastCaptureHash) return;
                g.lastCaptureHash = hash;
                g.lastCaptureHashValid = true;
            }
        }
        g.uniqueCaptures.fetch_add(1, std::memory_order_relaxed);
    }

    auto copied = std::make_shared<std::promise<void>>();
    auto completion = copied->get_future();
    AHardwareBuffer_acquire(ahb);
    {
        std::lock_guard<std::mutex> lock(g.mu);
        if (!g.initialized || g.stopRequested) {
            AHardwareBuffer_release(ahb);
            return;
        }
        // In framegen mode temporal continuity matters: throwing away older
        // captures widens the pair LSFG sees and causes optical-flow artifacts
        // on fast camera movement. So we keep a short FIFO.
        //
        // However, once the queue is saturated the worker is already too far
        // behind for continuity to hold — at that point dropping the NEW frame
        // pins the user on stale content (hundreds of ms old) and the overlay
        // feels stutter-y. Drop the OLDEST instead to prioritise freshness;
        // the continuity we lose is between two already-doomed frames.
        const size_t maxPending = static_cast<size_t>(
            g.queueDepth.load(std::memory_order_relaxed));
        if (g.pending.size() >= maxPending) {
            AHardwareBuffer_release(g.pending.front().ahb);
            g.pending.front().copied->set_value();
            g.pending.pop_front();
        }
        g.pending.push_back(State::PendingFrame{
            .ahb = ahb,
            .queuedAt = State::Clock::now(),
            .captureTimestampNs = timestampNs,
            .sequence = ++g.nextSequence,
            .animeMode = animeMode,
            .animeDecision = animeDecision,
            .copied = copied,
        });
    }
    g.pendingCv.notify_one();
    completion.wait();
}

void shutdownRenderLoop() {
    {
        std::lock_guard<std::mutex> lock(g.mu);
        g.stopRequested = true;
    }
    g.pendingCv.notify_all();
    {
        std::lock_guard<std::mutex> pLock(g.presentMu);
        g.presentStopRequested = true;
    }
    g.presentCv.notify_all();
    if (g.worker.joinable()) g.worker.join();
    if (g.presentWorker.joinable()) g.presentWorker.join();

    {
        std::lock_guard<std::mutex> pLock(g.presentMu);
        g.presentQueue.clear();
        g.inSlotRefCount.fill(0);
        g.outSlotRefCount.fill(0);
    }

    {
        std::lock_guard<std::mutex> lock(g.mu);
        for (const auto &pendingFrame : g.pending) {
            AHardwareBuffer_release(pendingFrame.ahb);
            pendingFrame.copied->set_value();
        }
        g.pending.clear();
        destroySwapchain();
        g.interpolator.reset();
        for (auto &view : g.frameViews) {
            if (view != VK_NULL_HANDLE) g.vk.fn.vkDestroyImageView(g.vk.device, view, nullptr);
            view = VK_NULL_HANDLE;
        }

        for (auto &o : g.outputs) destroyAhbImage(g.vk, o);
        g.outputs.clear();
        g.gpuPost.reset(g.vk);
        destroyAhbImage(g.vk, g.gpuPostImage);
        for (size_t i = 0; i < State::kInputSlotCount; ++i) destroyAhbImage(g.vk, g.inSlot[i]);



        g.npuPost.reset();
        g.cpuPost.reset();

        // Destroy swapchain (and its surface) before the underlying ANativeWindow
        // is touched — surface destruction drops its internal window ref.
        // g.outWindow is intentionally kept live so the worker thread can still
        // blit during the next session's shader-compilation window (before
        // setOutputSurface is called).  setOutputSurface always releases the
        // old reference before acquiring the new one, so there is no leak.
        g.swapWinW = 0;
        g.swapWinH = 0;

        destroy_session(g.vk);
        g.initialized = false;
        g.generatedFrames.store(0, std::memory_order_relaxed);
        g.postedFrames.store(0, std::memory_order_relaxed);
        g.uniqueCaptures.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> hashLock(g.captureHashMu);
            g.lastCaptureHash = 0;
            g.lastCaptureHashValid = false;
            g.animeCadence.reset();
            g.animeDuplicates.store(0, std::memory_order_relaxed);
        }
        g.postRingHead.store(0, std::memory_order_relaxed);
        for (auto &slot : g.postRingTimestamps) {
            slot.store(0, std::memory_order_relaxed);
        }
        g.shizukuSampleTimestampNs.store(0, std::memory_order_relaxed);
        g.shizukuFrameTimeNs.store(0, std::memory_order_relaxed);
        g.shizukuPacingJitterNs.store(0, std::memory_order_relaxed);
    }
    LOGI("Render loop shut down");
}

uint64_t getGeneratedFrameCount() {
    return g.generatedFrames.load(std::memory_order_relaxed);
}

uint64_t getPostedFrameCount() {
    return g.postedFrames.load(std::memory_order_relaxed);
}

uint64_t getUniqueCaptureCount() {
    return g.uniqueCaptures.load(std::memory_order_relaxed);
}

double getAverageQueueMs() {
    const int64_t samples = g.profileSnapshotSamples.load(std::memory_order_relaxed);
    if (samples <= 0) return 0.0;
    const int64_t queueNs = g.profileSnapshotQueueNs.load(std::memory_order_relaxed);
    return (static_cast<double>(queueNs) / samples) / 1'000'000.0;
}

double getAverageLatencyMs() {
    const int64_t blitCount = g.profileSnapshotBlitCount.load(std::memory_order_relaxed);
    if (blitCount <= 0) return 0.0;
    const int64_t latencyNs = g.profileSnapshotLatencyNs.load(std::memory_order_relaxed);
    return (static_cast<double>(latencyNs) / blitCount) / 1'000'000.0;
}

uint32_t getProfileWindowNs(int64_t *out, uint32_t cap) {
    // out layout: [copyNs, presentNs, waitIdleNs, blitNs, totalNs, samples].
    // Each value is the SUM over the last completed kProfileWindow window;
    // divide by samples for per-frame averages. Returns 6 when populated, 0
    // when no window has closed yet (samples==0) or when out is too small.
    if (out == nullptr || cap < 6) return 0;
    const int64_t samples = g.profileSnapshotSamples.load(std::memory_order_relaxed);
    if (samples <= 0) return 0;
    out[0] = g.profileSnapshotCopyNs.load(std::memory_order_relaxed);
    out[1] = g.profileSnapshotPresentNs.load(std::memory_order_relaxed);
    out[2] = g.profileSnapshotWaitIdleNs.load(std::memory_order_relaxed);
    out[3] = g.profileSnapshotBlitNs.load(std::memory_order_relaxed);
    out[4] = g.profileSnapshotTotalNs.load(std::memory_order_relaxed);
    out[5] = samples;
    return 6;
}

uint32_t getRecentPostIntervalsNs(int64_t *outIntervalsNs, uint32_t cap) {
    if (outIntervalsNs == nullptr || cap == 0) return 0;
    // Snapshot the ring head. Entries from (head - kPostRingSize) to (head-1)
    // are populated (older ones are overwritten by wrap-around). For intervals
    // we need consecutive pairs, so we can produce at most min(cap, N-1)
    // where N is how many valid entries are present.
    const uint64_t head = g.postRingHead.load(std::memory_order_acquire);
    if (head < 2) return 0;  // need at least 2 timestamps for one interval
    const uint64_t validEntries = std::min<uint64_t>(head, State::kPostRingSize);
    const uint64_t available = validEntries - 1;
    const uint32_t want = static_cast<uint32_t>(std::min<uint64_t>(cap, available));
    // Walk the ring newest-first: slot (head-1), (head-2), ... subtracting
    // successive pairs to produce intervals. Skip the pair if either half
    // is zero (race with concurrent write during startup).
    uint32_t written = 0;
    uint64_t prevTs = g.postRingTimestamps[(head - 1) % State::kPostRingSize]
                          .load(std::memory_order_relaxed);
    for (uint32_t i = 1; i <= want && written < cap; ++i) {
        const uint64_t slot = (head - 1 - i) % State::kPostRingSize;
        const uint64_t ts = g.postRingTimestamps[slot].load(std::memory_order_relaxed);
        if (ts == 0 || prevTs == 0 || prevTs < ts) {
            // Either a torn read (zero) or non-monotonic (wraparound race):
            // stop and return what we have.
            break;
        }
        outIntervalsNs[written++] = static_cast<int64_t>(prevTs - ts);
        prevTs = ts;
    }
    return written;
}

void setBypass(bool bypass) {
    g.bypass.store(bypass, std::memory_order_relaxed);
}

void setAnimeMode(bool enabled) {
    std::lock_guard<std::mutex> lock(g.captureHashMu);
    if (g.animeMode == enabled) return;
    g.animeMode = enabled;
    g.animeCadence.reset();
    g.lastCaptureHashValid = false;
    LOGI("Anime frame detection %s", enabled ? "enabled" : "disabled");
}

void setMultiplier(int multiplier) {
    const int clamped = std::clamp(multiplier, 2, 4);
    g.targetMultiplier.store(clamped, std::memory_order_relaxed);
    LOGI("ZeroFG multiplier set to %dx (auto-capped by display refresh)", clamped);
}

void setGpuSrMode(bool enabled, int method, float sharpness, float strength) {
    g.gpuMethod.store(std::clamp(method, 0, 16), std::memory_order_relaxed);
    g.gpuSharpness.store(clamp01(sharpness), std::memory_order_relaxed);
    g.gpuStrength.store(clamp01(strength), std::memory_order_relaxed);
    g.gpuNoopLogged.store(false, std::memory_order_relaxed);
    g.gpuPostProcessing.store(enabled, std::memory_order_relaxed);
    LOGI("GPU SR %s (method=%s sharp=%.2f strength=%.2f)",
         enabled ? "enabled" : "disabled",
         gpuMethodName(g.gpuMethod.load(std::memory_order_relaxed)),
         g.gpuSharpness.load(std::memory_order_relaxed),
         g.gpuStrength.load(std::memory_order_relaxed));
}

void setAntiArtifacts(bool enabled) {
    g.antiArtifacts.store(enabled, std::memory_order_relaxed);
}

void setVsyncPeriodNs(int64_t periodNs) {
    if (periodNs < 0) periodNs = 0;
    g.vsyncPeriodNs.store(periodNs, std::memory_order_relaxed);
}

void setPacingParams(int targetFpsCap, float emaAlpha, float outlierRatio,
                     float vsyncSlackMs, int queueDepth) {
    applyPacingParamsImpl(targetFpsCap, emaAlpha, outlierRatio,
                          vsyncSlackMs, queueDepth);
}

void setShizukuTimingEnabled(bool enabled) {
    g.shizukuTimingEnabled.store(enabled, std::memory_order_relaxed);
    if (!enabled) {
        g.shizukuSampleTimestampNs.store(0, std::memory_order_relaxed);
        g.shizukuFrameTimeNs.store(0, std::memory_order_relaxed);
        g.shizukuPacingJitterNs.store(0, std::memory_order_relaxed);
    }
    LOGI("Shizuku timing %s", enabled ? "enabled" : "disabled");
}

void reportShizukuTiming(int64_t timestampNs,
                         int64_t frameTimeNs,
                         int64_t pacingJitterNs) {
    if (!g.shizukuTimingEnabled.load(std::memory_order_relaxed)) return;
    g.shizukuSampleTimestampNs.store(timestampNs, std::memory_order_relaxed);
    g.shizukuFrameTimeNs.store(frameTimeNs, std::memory_order_relaxed);
    g.shizukuPacingJitterNs.store(
        pacingJitterNs >= 0 ? pacingJitterNs : 0,
        std::memory_order_relaxed);
}

} // namespace lsfg_android
