// JNI entry points for com.lsfg.android.session.NativeBridge.
//
// Phase 3: extractShaders extracts DXBC resources and writes SPIR-V to disk.
// Phase 4: probeShaders validates that every cached SPIR-V blob is accepted by
// the device driver via vkCreateShaderModule.
// Phase 5: initContext / pushFrame / setOutputSurface / destroyContext wire
// the LSFG_3_1 Vulkan pipeline through the AHB bridge.

#include "crash_reporter.hpp"
#include "lsfg_render_loop.hpp"
#include "nnapi_npu.hpp"

#include <android/hardware_buffer_jni.h>
#include <android/native_window_jni.h>
#include <jni.h>

#include <string>

namespace {

constexpr const char *kVersion = "ZeroFG Android 0.1.0";

std::string jstring_to_std(JNIEnv *env, jstring s) {
    if (s == nullptr) {
        return {};
    }
    const char *chars = env->GetStringUTFChars(s, nullptr);
    std::string out = chars ? chars : "";
    if (chars) {
        env->ReleaseStringUTFChars(s, chars);
    }
    return out;
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_lsfg_android_session_NativeBridge_nativeVersion(JNIEnv *env, jobject /*thiz*/) {
    return env->NewStringUTF(kVersion);
}

extern "C" JNIEXPORT void JNICALL
Java_com_lsfg_android_session_NativeBridge_initCrashReporter(
        JNIEnv *env, jobject /*thiz*/, jstring crashPath, jstring logPath) {
    lsfg_android::init_crash_reporter(
        jstring_to_std(env, crashPath),
        jstring_to_std(env, logPath));
}

extern "C" JNIEXPORT jint JNICALL
Java_com_lsfg_android_session_NativeBridge_initContext(
        JNIEnv *env, jobject /*thiz*/,
        jstring cacheDir, jint width, jint height,
        jint multiplier, jfloat flowScale,
        jboolean performance, jboolean hdr,
        jboolean antiArtifacts, jboolean framegenFp16,
        jboolean npuPostProcessing, jint npuPreset,
        jint npuUpscaleFactor, jfloat npuAmount,
        jfloat npuRadius, jfloat npuThreshold, jboolean npuFp16,
        jboolean cpuPostProcessing, jint cpuPreset,
        jfloat cpuStrength, jfloat cpuSaturation,
        jfloat cpuVibrance, jfloat cpuVignette,
        jboolean gpuPostProcessing, jint gpuStage,
        jint gpuMethod, jfloat gpuUpscaleFactor,
        jfloat gpuSharpness, jfloat gpuStrength,
        jint targetFpsCap, jfloat emaAlpha,
        jfloat outlierRatio, jfloat vsyncSlackMs,
        jint queueDepth) {
    if (width <= 0 || height <= 0) return -1;
    const lsfg_android::RenderLoopConfig cfg{
        .width = static_cast<uint32_t>(width),
        .height = static_cast<uint32_t>(height),
        .multiplier = static_cast<int>(multiplier),
        .flowScale = static_cast<float>(flowScale),
        .performance = performance == JNI_TRUE,
        .hdr = hdr == JNI_TRUE,
        .antiArtifacts = antiArtifacts == JNI_TRUE,
        .framegenFp16 = framegenFp16 == JNI_TRUE,
        .npuPostProcessing = npuPostProcessing == JNI_TRUE,
        .npuPreset = static_cast<int>(npuPreset),
        .npuUpscaleFactor = static_cast<int>(npuUpscaleFactor),
        .npuAmount = static_cast<float>(npuAmount),
        .npuRadius = static_cast<float>(npuRadius),
        .npuThreshold = static_cast<float>(npuThreshold),
        .npuFp16 = npuFp16 == JNI_TRUE,
        .cpuPostProcessing = cpuPostProcessing == JNI_TRUE,
        .cpuPreset = static_cast<int>(cpuPreset),
        .cpuStrength = static_cast<float>(cpuStrength),
        .cpuSaturation = static_cast<float>(cpuSaturation),
        .cpuVibrance = static_cast<float>(cpuVibrance),
        .cpuVignette = static_cast<float>(cpuVignette),
        .gpuPostProcessing = gpuPostProcessing == JNI_TRUE,
        .gpuStage = static_cast<int>(gpuStage),
        .gpuMethod = static_cast<int>(gpuMethod),
        .gpuUpscaleFactor = static_cast<float>(gpuUpscaleFactor),
        .gpuSharpness = static_cast<float>(gpuSharpness),
        .gpuStrength = static_cast<float>(gpuStrength),
        .targetFpsCap = static_cast<int>(targetFpsCap),
        .emaAlpha = static_cast<float>(emaAlpha),
        .outlierRatio = static_cast<float>(outlierRatio),
        .vsyncSlackMs = static_cast<float>(vsyncSlackMs),
        .queueDepth = static_cast<int>(queueDepth),
    };
    return lsfg_android::initRenderLoop(cfg);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_lsfg_android_session_NativeBridge_isNpuAvailable(JNIEnv * /*env*/, jobject /*thiz*/) {
    return lsfg_android::nnapi_has_npu_accelerator() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_lsfg_android_session_NativeBridge_isFramegenFp16Supported(
        JNIEnv *env, jobject /*thiz*/, jstring cacheDir) {
    return JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_lsfg_android_session_NativeBridge_getNpuSummary(JNIEnv *env, jobject /*thiz*/) {
    return env->NewStringUTF(lsfg_android::nnapi_npu_summary().c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_lsfg_android_session_NativeBridge_setOutputSurface(
        JNIEnv *env, jobject /*thiz*/, jobject surface, jint w, jint h) {
    ANativeWindow *win = (surface != nullptr)
        ? ANativeWindow_fromSurface(env, surface)
        : nullptr;
    lsfg_android::setOutputSurface(win, static_cast<uint32_t>(w), static_cast<uint32_t>(h));
    if (win != nullptr) {
        // setOutputSurface has acquired its own reference; release ours.
        ANativeWindow_release(win);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_lsfg_android_session_NativeBridge_pushFrame(
        JNIEnv *env, jobject /*thiz*/, jobject hardwareBuffer, jlong timestampNs) {
    if (hardwareBuffer == nullptr) return;
    AHardwareBuffer *ahb = AHardwareBuffer_fromHardwareBuffer(env, hardwareBuffer);
    lsfg_android::pushFrame(ahb, static_cast<int64_t>(timestampNs));
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_lsfg_android_session_NativeBridge_getGeneratedFrameCount(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return static_cast<jlong>(lsfg_android::getGeneratedFrameCount());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_lsfg_android_session_NativeBridge_getPostedFrameCount(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return static_cast<jlong>(lsfg_android::getPostedFrameCount());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_lsfg_android_session_NativeBridge_getUniqueCaptureCount(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return static_cast<jlong>(lsfg_android::getUniqueCaptureCount());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_lsfg_android_session_NativeBridge_getAverageQueueMs(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(lsfg_android::getAverageQueueMs());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_lsfg_android_session_NativeBridge_getAverageLatencyMs(
        JNIEnv * /*env*/, jobject /*thiz*/) {
    return static_cast<jdouble>(lsfg_android::getAverageLatencyMs());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_lsfg_android_session_NativeBridge_getRecentPostIntervalsNs(
        JNIEnv *env, jobject /*thiz*/, jlongArray outArray) {
    if (outArray == nullptr) return 0;
    const jsize cap = env->GetArrayLength(outArray);
    if (cap <= 0) return 0;
    jlong *buf = env->GetLongArrayElements(outArray, nullptr);
    if (buf == nullptr) return 0;
    static_assert(sizeof(jlong) == sizeof(int64_t), "jlong must be int64_t");
    const uint32_t written = lsfg_android::getRecentPostIntervalsNs(
        reinterpret_cast<int64_t *>(buf), static_cast<uint32_t>(cap));
    env->ReleaseLongArrayElements(outArray, buf, 0);
    return static_cast<jint>(written);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_lsfg_android_session_NativeBridge_getProfileWindowNs(
        JNIEnv *env, jobject /*thiz*/, jlongArray outArray) {
    if (outArray == nullptr) return 0;
    const jsize cap = env->GetArrayLength(outArray);
    if (cap < 6) return 0;
    jlong *buf = env->GetLongArrayElements(outArray, nullptr);
    if (buf == nullptr) return 0;
    static_assert(sizeof(jlong) == sizeof(int64_t), "jlong must be int64_t");
    const uint32_t written = lsfg_android::getProfileWindowNs(
        reinterpret_cast<int64_t *>(buf), static_cast<uint32_t>(cap));
    env->ReleaseLongArrayElements(outArray, buf, 0);
    return static_cast<jint>(written);
}

extern "C" JNIEXPORT void JNICALL
Java_com_lsfg_android_session_NativeBridge_setBypass(
        JNIEnv * /*env*/, jobject /*thiz*/, jboolean bypass) {
    lsfg_android::setBypass(bypass == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL
Java_com_lsfg_android_session_NativeBridge_setAnimeMode(
        JNIEnv *, jobject, jboolean enabled) {
    lsfg_android::setAnimeMode(enabled == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL
Java_com_lsfg_android_session_NativeBridge_setMultiplier(
        JNIEnv *, jobject, jint multiplier) {
    lsfg_android::setMultiplier(static_cast<int>(multiplier));
}

extern "C" JNIEXPORT void JNICALL
Java_com_lsfg_android_session_NativeBridge_setGpuSrMode(
        JNIEnv *, jobject, jboolean enabled, jint method,
        jfloat sharpness, jfloat strength) {
    lsfg_android::setGpuSrMode(
        enabled == JNI_TRUE,
        static_cast<int>(method),
        static_cast<float>(sharpness),
        static_cast<float>(strength));
}

extern "C" JNIEXPORT void JNICALL
Java_com_lsfg_android_session_NativeBridge_setAntiArtifacts(
        JNIEnv * /*env*/, jobject /*thiz*/, jboolean enabled) {
    lsfg_android::setAntiArtifacts(enabled == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL
Java_com_lsfg_android_session_NativeBridge_setVsyncPeriodNs(
        JNIEnv * /*env*/, jobject /*thiz*/, jlong periodNs) {
    lsfg_android::setVsyncPeriodNs(static_cast<int64_t>(periodNs));
}

extern "C" JNIEXPORT void JNICALL
Java_com_lsfg_android_session_NativeBridge_setPacingParams(
        JNIEnv * /*env*/, jobject /*thiz*/,
        jint targetFpsCap, jfloat emaAlpha, jfloat outlierRatio,
        jfloat vsyncSlackMs, jint queueDepth) {
    lsfg_android::setPacingParams(
        static_cast<int>(targetFpsCap),
        static_cast<float>(emaAlpha),
        static_cast<float>(outlierRatio),
        static_cast<float>(vsyncSlackMs),
        static_cast<int>(queueDepth));
}

extern "C" JNIEXPORT void JNICALL
Java_com_lsfg_android_session_NativeBridge_setShizukuTimingEnabled(
        JNIEnv * /*env*/, jobject /*thiz*/, jboolean enabled) {
    lsfg_android::setShizukuTimingEnabled(enabled == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL
Java_com_lsfg_android_session_NativeBridge_reportShizukuTiming(
        JNIEnv * /*env*/, jobject /*thiz*/,
        jlong timestampNs, jlong frameTimeNs, jlong pacingJitterNs) {
    lsfg_android::reportShizukuTiming(
        static_cast<int64_t>(timestampNs),
        static_cast<int64_t>(frameTimeNs),
        static_cast<int64_t>(pacingJitterNs));
}

extern "C" JNIEXPORT void JNICALL
Java_com_lsfg_android_session_NativeBridge_destroyContext(JNIEnv * /*env*/, jobject /*thiz*/) {
    lsfg_android::shutdownRenderLoop();
}
