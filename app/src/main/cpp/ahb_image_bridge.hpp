#pragma once

#include "android_vk_session.hpp"

#include <volk.h>
#include <android/hardware_buffer.h>

#include <cstdint>

namespace lsfg_android {

constexpr int kErrAhbAllocate = -30;
constexpr int kErrAhbProperties = -31;
constexpr int kErrAhbImageCreate = -32;
constexpr int kErrAhbMemoryAllocate = -33;
constexpr int kErrAhbBindMemory = -34;

struct AhbImage {
    AHardwareBuffer *ahb = nullptr;       // owned iff ownsAhb
    bool ownsAhb = false;
    VkImage image = VK_NULL_HANDLE;        // owned
    VkDeviceMemory memory = VK_NULL_HANDLE; // owned
    VkExtent2D extent{0, 0};
    VkFormat format = VK_FORMAT_UNDEFINED;
};

// Allocates a fresh AHardwareBuffer of (w, h, fmt) and wraps it in a VkImage.
//
// Format must be one of the standard AHB formats: AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM
// is the safe default for capture; other formats may need a YCbCr conversion.
//
// usage flags: GPU_SAMPLED_IMAGE | GPU_FRAMEBUFFER are set by default.
//
// Returns kOk on success; on failure returns kErrAhb* and `out` is left in
// a destroy()-safe state.
int createAhbImage(VulkanSession &vk, uint32_t w, uint32_t h,
                   VkFormat fmt, AhbImage &out);

// Imports an EXTERNAL AHardwareBuffer (e.g. one acquired from ImageReader).
// Does NOT take ownership of the AHB. The caller must keep it alive (via
// AHardwareBuffer_acquire) for as long as `out` is in use.
//
int importAhbImage(VulkanSession &vk, AHardwareBuffer *ahb, AhbImage &out);

// Releases everything. Safe to call on a partially-initialized AhbImage.
void destroyAhbImage(VulkanSession &vk, AhbImage &img);

} // namespace lsfg_android
