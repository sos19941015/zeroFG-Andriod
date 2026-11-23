#pragma once

// Owns the Vulkan device used for capture, interpolation and presentation.

#include <volk.h>
#include <zerofg/backend_capabilities.h>

#include <array>
#include <cstdint>
#include <string>

namespace lsfg_android {

constexpr int kOk = 0;
constexpr int kSessionNoVulkan = -20;
constexpr int kSessionNoSuitableGpu = -21;
constexpr int kSessionMissingExtension = -22;
constexpr int kSessionDeviceCreateFailed = -23;
constexpr int kSessionAlreadyInitialized = -24;

// Size of the pre-allocated command-buffer / fence ring. The worker thread
// rarely has more than 2-3 submissions in flight (input copy, post-process,
// output blit * multiplier), so 8 gives comfortable headroom to rotate through
// without stalling on the oldest-slot fence.
constexpr uint32_t kCommandRingSize = 8;
constexpr uint32_t kPresentRingSize = 4;

struct VulkanSession {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeFamilyIdx = UINT32_MAX;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandPool presentCommandPool = VK_NULL_HANDLE;

    // Pre-allocated primary command buffers + paired signalling fences. Used
    // round-robin through acquireCommandRing() — waits on the slot's fence
    // (signalled by the previous use), resets CB, hands back {cb, fence}.
    // Replaces the per-frame vkAllocateCommandBuffers + vkQueueWaitIdle +
    // vkFreeCommandBuffers pattern which cost ~0.5-1 ms of driver overhead
    // per submission.
    std::array<VkCommandBuffer, kCommandRingSize> ringCommandBuffers{};
    std::array<VkFence, kCommandRingSize> ringFences{};
    std::array<bool, kCommandRingSize> ringFenceArmed{};
    uint32_t ringNext = 0;

    // Separate command pool + ring exclusively owned by the present thread so
    // presentation never contends with the compute/copy worker on VkCommandPool.
    std::array<VkCommandBuffer, kPresentRingSize> presentCommandBuffers{};
    std::array<VkFence, kPresentRingSize> presentFences{};
    std::array<bool, kPresentRingSize> presentFenceArmed{};
    uint32_t presentRingNext = 0;

    // Per-device function table. Required because framegen creates its own
    // VkDevice with a different extension set and calls volkLoadDevice() on it,
    // which clobbers volk's global function pointers. Anything in this session
    // that touches the Vulkan device MUST go through `fn.*` instead of the
    // globals.
    VolkDeviceTable fn{};

    // Per-instance function pointers we resolved against OUR instance. Same
    // problem as VolkDeviceTable above but at instance scope: framegen's
    // Instance::Instance() creates its own VkInstance without surface
    // extensions and calls volkLoadInstance() on it, which clobbers the global
    // vkCreateAndroidSurfaceKHR pointer (the new instance can't resolve it,
    // so volk overwrites the global with NULL). Save our resolved pointer
    // here at session-init time and use it from the render loop instead of
    // the volk global.
    PFN_vkCreateAndroidSurfaceKHR pfnCreateAndroidSurfaceKHR = nullptr;
    PFN_vkDestroySurfaceKHR pfnDestroySurfaceKHR = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR pfnGetPhysicalDeviceSurfaceCapabilitiesKHR = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR pfnGetPhysicalDeviceSurfaceFormatsKHR = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR pfnGetPhysicalDeviceSurfaceSupportKHR = nullptr;


    // Device UUID packed the way framegen expects: vendorID<<32 | deviceID.
    // Pass this to LSFG_3_1::initialize.
    uint64_t deviceUuid = 0;

    // Whether VK_EXT_queue_family_foreign is enabled. If true, AHB ownership
    // transitions use FOREIGN_EXT; otherwise EXTERNAL.
    bool hasQueueFamilyForeign = false;

    // Whether the instance + device carry the swapchain extension chain
    // (VK_KHR_surface, VK_KHR_android_surface, VK_KHR_swapchain). When true
    // the render loop can present generated frames via the WSI path instead
    // of CPU-blitting through ANativeWindow_lock. Swapchain setup happens
    // lazily in setOutputSurface() once we have the ANativeWindow.
    bool hasSwapchain = false;
    bool hasDisplayTiming = false;
    bool hasGlobalPriority = false;
    uint32_t apiVersion = VK_API_VERSION_1_0;
    bool supportsZeroFG = false;
    zerofg::Capabilities zerofgCaps{};

    bool initialized() const { return device != VK_NULL_HANDLE; }
};

// Acquire a command buffer + fence from the ring. Blocks (via vkWaitForFences)
// until the slot's previous submission is retired, then resets both so the
// caller can record fresh commands. Returns true on success.
//
// Usage:
//   VkCommandBuffer cb; VkFence fence;
//   acquireCommandRing(vk, cb, fence);
//   record...
//   submitCommandRing(vk, cb, fence);
//   waitCommandRing(vk, fence); // or defer the wait past other CPU work
bool acquireCommandRing(VulkanSession &vk, VkCommandBuffer &outCb, VkFence &outFence);
bool acquirePresentRing(VulkanSession &vk, VkCommandBuffer &outCb, VkFence &outFence);
void markPresentFenceArmed(VulkanSession &vk, VkFence fence);

// Submit `cb` on the compute queue, signalling `fence` when it retires.
// Does NOT wait. Use waitCommandRing(fence) when the caller actually needs
// the GPU to be done.
bool submitCommandRing(VulkanSession &vk, VkCommandBuffer cb, VkFence fence);

// Block until `fence` signals. Returns true if the wait succeeded (or the
// fence was never armed). Marks the fence as "pending reset" so the next
// acquireCommandRing() call on that slot resets it.
bool waitCommandRing(VulkanSession &vk, VkFence fence);

// Creates the persistent Vulkan instance + device with all extensions required
// for AHardwareBuffer interop, FD export, and framegen integration.
//
// Returns kOk on success; sets `out` fields. On failure returns one of the
// kSession* error codes and leaves `out` partially-populated — call
// destroy_session() to clean up.
int create_session(VulkanSession &out);

// Tears down everything in the session. Safe to call on a partially-initialized
// session (it will skip null handles).
void destroy_session(VulkanSession &s);

} // namespace lsfg_android
