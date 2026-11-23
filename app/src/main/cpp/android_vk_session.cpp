#include "android_vk_session.hpp"

#include <volk.h>

#include <android/log.h>

#include <cstring>
#include <algorithm>
#include <vector>

#include "crash_reporter.hpp"

#define LOG_TAG "lsfg-vk-session"
#define LOGE(...) ::lsfg_android::ring_logf(LOG_TAG, ANDROID_LOG_ERROR, __VA_ARGS__)
#define LOGW(...) ::lsfg_android::ring_logf(LOG_TAG, ANDROID_LOG_WARN,  __VA_ARGS__)
#define LOGI(...) ::lsfg_android::ring_logf(LOG_TAG, ANDROID_LOG_INFO,  __VA_ARGS__)

namespace lsfg_android {

namespace {

// The remaining AHB dependencies are core in Vulkan 1.1.
constexpr const char *kRequiredDeviceExt[] = {
    VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME,
    VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
};

constexpr const char *kOptionalDeviceExt[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

// Optional instance extensions for the WSI path. Both are required to present
// on an ANativeWindow via Vulkan. If either is missing at instance level we
// disable the WSI path entirely (hasSwapchain stays false).
constexpr const char *kOptionalInstanceExt[] = {
    VK_KHR_SURFACE_EXTENSION_NAME,
    VK_KHR_ANDROID_SURFACE_EXTENSION_NAME,
};

bool has_extension(const std::vector<VkExtensionProperties> &avail, const char *name) {
    for (const auto &e : avail) {
        if (std::strcmp(e.extensionName, name) == 0) return true;
    }
    return false;
}

bool has_instance_extension(const std::vector<VkExtensionProperties> &avail, const char *name) {
    for (const auto &e : avail) {
        if (std::strcmp(e.extensionName, name) == 0) return true;
    }
    return false;
}

// Build the instance with the WSI extensions when they're available. Falling
// back silently is the right move here — the worst case is we miss the
// swapchain fast path, and the CPU blit still works.
VkInstance make_instance(bool &hasSurfaceExts) {
    hasSurfaceExts = false;
    uint32_t loaderVersion = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion) vkEnumerateInstanceVersion(&loaderVersion);
    const VkApplicationInfo appInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "lsfg-android",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "lsfg-vk",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = std::min(loaderVersion, VK_API_VERSION_1_3),
    };

    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> avail(extCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, avail.data());

    std::vector<const char *> enabled;
    bool surfaceAll = true;
    for (const char *opt : kOptionalInstanceExt) {
        if (has_instance_extension(avail, opt)) {
            enabled.push_back(opt);
        } else {
            surfaceAll = false;
        }
    }
    hasSurfaceExts = surfaceAll;

    VkInstanceCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
        .enabledExtensionCount = static_cast<uint32_t>(enabled.size()),
        .ppEnabledExtensionNames = enabled.data(),
    };
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&info, nullptr, &inst) != VK_SUCCESS) return VK_NULL_HANDLE;
    return inst;
}

VkPhysicalDevice pick_physical_device(VkInstance inst) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(inst, &count, nullptr);
    if (count == 0) return VK_NULL_HANDLE;
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(inst, &count, devs.data());
    // First device is the only one on a typical Android phone. If we ever
    // run on a system with a real discrete + integrated split, prefer the
    // one that exposes all required extensions.
    for (auto d : devs) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(d, &props);
        if (props.apiVersion < VK_API_VERSION_1_1) continue;
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(d, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(d, nullptr, &extCount, exts.data());

        bool ok = true;
        for (const char *req : kRequiredDeviceExt) {
            if (!has_extension(exts, req)) {
                LOGW("%s does not advertise %s", props.deviceName, req);
                ok = false;
            }
        }
        if (ok) return d;
    }
    return VK_NULL_HANDLE;
}

uint32_t find_compute_family(VkPhysicalDevice phys) {
    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> q(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qCount, q.data());
    for (uint32_t i = 0; i < qCount; ++i) {
        if (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
    }
    return UINT32_MAX;
}

} // namespace

int create_session(VulkanSession &out) {
    if (out.initialized()) return kSessionAlreadyInitialized;

    if (volkInitialize() != VK_SUCCESS) {
        LOGE("volkInitialize failed");
        return kSessionNoVulkan;
    }

    bool hasSurfaceExts = false;
    out.instance = make_instance(hasSurfaceExts);
    if (out.instance == VK_NULL_HANDLE) {
        LOGE("vkCreateInstance failed");
        return kSessionNoVulkan;
    }
    volkLoadInstance(out.instance);

    // Resolve vkCreateAndroidSurfaceKHR against OUR instance and stash the
    // pointer in the session. Two reasons we can't rely on the volk global:
    //   1. On some drivers (observed on Adreno 840 / Vulkan 1.4 and various
    //      Mali firmwares, also HyperOS Adreno builds) volk's normal load
    //      path doesn't populate the symbol even when VK_KHR_android_surface
    //      was accepted by vkCreateInstance — vkGetInstanceProcAddr resolves
    //      it manually.
    //   2. Even if volkLoadInstance loaded it correctly, framegen's
    //      Instance::Instance() will later call volkLoadInstance() against
    //      its own VkInstance (created without surface extensions), which
    //      overwrites the global pointer to NULL. The render loop reads the
    //      pointer at swapchain-creation time, after framegen init, so the
    //      global is unreliable.
    // The render loop reads out.pfnCreateAndroidSurfaceKHR instead.
    if (hasSurfaceExts) {
        out.pfnCreateAndroidSurfaceKHR = reinterpret_cast<PFN_vkCreateAndroidSurfaceKHR>(
            vkGetInstanceProcAddr(out.instance, "vkCreateAndroidSurfaceKHR"));
        out.pfnDestroySurfaceKHR = reinterpret_cast<PFN_vkDestroySurfaceKHR>(
            vkGetInstanceProcAddr(out.instance, "vkDestroySurfaceKHR"));
        out.pfnGetPhysicalDeviceSurfaceCapabilitiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
            vkGetInstanceProcAddr(out.instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"));
        out.pfnGetPhysicalDeviceSurfaceFormatsKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
            vkGetInstanceProcAddr(out.instance, "vkGetPhysicalDeviceSurfaceFormatsKHR"));
        out.pfnGetPhysicalDeviceSurfaceSupportKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
            vkGetInstanceProcAddr(out.instance, "vkGetPhysicalDeviceSurfaceSupportKHR"));

        if (out.pfnCreateAndroidSurfaceKHR != nullptr &&
            out.pfnDestroySurfaceKHR != nullptr &&
            out.pfnGetPhysicalDeviceSurfaceCapabilitiesKHR != nullptr &&
            out.pfnGetPhysicalDeviceSurfaceFormatsKHR != nullptr &&
            out.pfnGetPhysicalDeviceSurfaceSupportKHR != nullptr) {
            LOGI("vkCreateAndroidSurfaceKHR and related WSI functions resolved and cached on session");
        } else {
            LOGW("WSI surface functions unresolvable on this driver — disabling WSI");
            hasSurfaceExts = false;
        }
    }

    out.physicalDevice = pick_physical_device(out.instance);
    if (out.physicalDevice == VK_NULL_HANDLE) {
        LOGE("No physical device with all required extensions");
        return kSessionMissingExtension;
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(out.physicalDevice, &props);
    out.deviceUuid =
        (static_cast<uint64_t>(props.vendorID) << 32) | props.deviceID;
    LOGI("Using GPU: %s (API %u.%u.%u, vendor=0x%x device=0x%x uuid=0x%llx)",
         props.deviceName,
         VK_VERSION_MAJOR(props.apiVersion),
         VK_VERSION_MINOR(props.apiVersion),
         VK_VERSION_PATCH(props.apiVersion),
         props.vendorID, props.deviceID,
         (unsigned long long)out.deviceUuid);

    out.computeFamilyIdx = find_compute_family(out.physicalDevice);
    if (out.computeFamilyIdx == UINT32_MAX) {
        LOGE("No compute queue family");
        return kSessionNoSuitableGpu;
    }

    // Probe optional extensions and build the final list.
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(out.physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> avail(extCount);
    vkEnumerateDeviceExtensionProperties(out.physicalDevice, nullptr, &extCount, avail.data());

    // Log key extension availability for diagnostics.
    {
        static const char* kDiagExts[] = {
            "VK_KHR_timeline_semaphore",
            "VK_KHR_synchronization2",
            "VK_KHR_vulkan_memory_model",
                    VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME,
            VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
            VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        };
        LOGI("Device extension availability (%u total):", extCount);
        for (const char* ext : kDiagExts) {
            LOGI("  [%s] %s", has_extension(avail, ext) ? "Y" : "N", ext);
        }
    }

    std::vector<const char *> enabledExts;
    for (const char *req : kRequiredDeviceExt) enabledExts.push_back(req);
    out.hasQueueFamilyForeign = true;
    bool hasSwapchainDevExt = false;
    for (const char *opt : kOptionalDeviceExt) {
        if (has_extension(avail, opt)) {
            enabledExts.push_back(opt);
            if (std::strcmp(opt, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
                hasSwapchainDevExt = true;
            }
        } else {
            LOGW("Optional extension %s not available", opt);
        }
    }
    out.hasSwapchain = hasSurfaceExts && hasSwapchainDevExt;
    out.hasDisplayTiming = out.hasSwapchain &&
        has_extension(avail, VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME);
    if (out.hasDisplayTiming) enabledExts.push_back(VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME);

    bool filterCubicEnabled = false;
    if (has_extension(avail, VK_EXT_FILTER_CUBIC_EXTENSION_NAME)) {
        enabledExts.push_back(VK_EXT_FILTER_CUBIC_EXTENSION_NAME);
        filterCubicEnabled = true;
    } else if (has_extension(avail, VK_IMG_FILTER_CUBIC_EXTENSION_NAME)) {
        enabledExts.push_back(VK_IMG_FILTER_CUBIC_EXTENSION_NAME);
        filterCubicEnabled = true;
    }

    const bool hasKhrGlobalPriority = has_extension(avail, VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
    const bool hasExtGlobalPriority = !hasKhrGlobalPriority &&
        has_extension(avail, VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME);
    out.hasGlobalPriority = hasKhrGlobalPriority || hasExtGlobalPriority;
    if (hasKhrGlobalPriority) enabledExts.push_back(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
    else if (hasExtGlobalPriority) enabledExts.push_back(VK_EXT_GLOBAL_PRIORITY_EXTENSION_NAME);

    if (!out.hasSwapchain) {
        LOGW("WSI path disabled (surface=%d, swapchain=%d); falling back to CPU blit for output",
             (int)hasSurfaceExts, (int)hasSwapchainDevExt);
    }

    uint32_t loaderVersion = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion) vkEnumerateInstanceVersion(&loaderVersion);
    out.apiVersion = std::min({loaderVersion, props.apiVersion, VK_API_VERSION_1_3});

    VkPhysicalDeviceVulkan11Features features11{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
    };
    VkPhysicalDeviceVulkan12Features features12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    };
    VkPhysicalDeviceVulkan13Features features13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    };
    VkPhysicalDeviceFeatures2 supported{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    };
    if (out.apiVersion >= VK_API_VERSION_1_3) {
        supported.pNext = &features11;
        features11.pNext = &features12;
        features12.pNext = &features13;
    } else if (out.apiVersion >= VK_API_VERSION_1_2) {
        supported.pNext = &features11;
        features11.pNext = &features12;
    }
    if (vkGetPhysicalDeviceFeatures2) vkGetPhysicalDeviceFeatures2(out.physicalDevice, &supported);

    VkPhysicalDeviceSubgroupProperties subgroupProps{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    };
    VkPhysicalDeviceSubgroupSizeControlProperties subgroupSizeProps{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 props2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &subgroupProps,
    };
    if (out.apiVersion >= VK_API_VERSION_1_3) {
        subgroupProps.pNext = &subgroupSizeProps;
    }
    if (vkGetPhysicalDeviceProperties2) vkGetPhysicalDeviceProperties2(out.physicalDevice, &props2);

    VkPhysicalDeviceVulkan11Features enabled11{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .storageBuffer16BitAccess = features11.storageBuffer16BitAccess,
    };
    VkPhysicalDeviceVulkan12Features enabled12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .shaderFloat16 = features12.shaderFloat16,
        .shaderInt8 = features12.shaderInt8,
    };
    VkPhysicalDeviceVulkan13Features enabled13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .subgroupSizeControl = features13.subgroupSizeControl,
        .computeFullSubgroups = features13.computeFullSubgroups,
        .synchronization2 = features13.synchronization2,
        .maintenance4 = features13.maintenance4,
    };
    out.supportsZeroFG = out.apiVersion >= VK_API_VERSION_1_3 &&
        enabled13.synchronization2 && supported.features.shaderStorageImageExtendedFormats;
    VkPhysicalDeviceFeatures enabled{};
    enabled.shaderStorageImageExtendedFormats = supported.features.shaderStorageImageExtendedFormats;

    out.zerofgCaps = {};
    out.zerofgCaps.effective_api_version = out.apiVersion;
    out.zerofgCaps.synchronization2_enabled = enabled13.synchronization2 == VK_TRUE;
    out.zerofgCaps.shader_storage_image_extended_formats_enabled =
        enabled.shaderStorageImageExtendedFormats == VK_TRUE;
    out.zerofgCaps.shader_float16_supported = features12.shaderFloat16 == VK_TRUE;
    out.zerofgCaps.shader_float16_enabled = enabled12.shaderFloat16 == VK_TRUE;
    out.zerofgCaps.shader_int8_supported = features12.shaderInt8 == VK_TRUE;
    out.zerofgCaps.shader_int8_enabled = enabled12.shaderInt8 == VK_TRUE;
    out.zerofgCaps.storage_buffer_16bit_access_supported = features11.storageBuffer16BitAccess == VK_TRUE;
    out.zerofgCaps.storage_buffer_16bit_access_enabled = enabled11.storageBuffer16BitAccess == VK_TRUE;
    out.zerofgCaps.subgroup_size = subgroupProps.subgroupSize;
    out.zerofgCaps.subgroup_supported_stages = subgroupProps.supportedStages;
    out.zerofgCaps.subgroup_supported_operations = subgroupProps.supportedOperations;
    if (out.apiVersion >= VK_API_VERSION_1_3) {
        out.zerofgCaps.subgroup_min_size = subgroupSizeProps.minSubgroupSize;
        out.zerofgCaps.subgroup_max_size = subgroupSizeProps.maxSubgroupSize;
        out.zerofgCaps.max_compute_workgroup_subgroups = subgroupSizeProps.maxComputeWorkgroupSubgroups;
        out.zerofgCaps.required_subgroup_size_stages = subgroupSizeProps.requiredSubgroupSizeStages;
    }
    out.zerofgCaps.subgroup_size_control_supported = features13.subgroupSizeControl == VK_TRUE;
    out.zerofgCaps.subgroup_size_control_enabled = enabled13.subgroupSizeControl == VK_TRUE;
    out.zerofgCaps.compute_full_subgroups_supported = features13.computeFullSubgroups == VK_TRUE;
    out.zerofgCaps.compute_full_subgroups_enabled = enabled13.computeFullSubgroups == VK_TRUE;
    out.zerofgCaps.maintenance4_supported = features13.maintenance4 == VK_TRUE;
    out.zerofgCaps.maintenance4_enabled = enabled13.maintenance4 == VK_TRUE;
    out.zerofgCaps.filter_cubic_supported = filterCubicEnabled;
    out.zerofgCaps.filter_cubic_enabled = filterCubicEnabled;

    void *devPNext = nullptr;
    if (out.apiVersion >= VK_API_VERSION_1_3) {
        enabled11.pNext = &enabled12;
        enabled12.pNext = &enabled13;
        devPNext = &enabled11;
    } else if (out.apiVersion >= VK_API_VERSION_1_2) {
        enabled11.pNext = &enabled12;
        devPNext = &enabled11;
    }

    const float prio = 1.0f;
    VkDeviceQueueGlobalPriorityCreateInfoKHR globalPrioInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
        .globalPriority = VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR,
    };
    VkDeviceQueueCreateInfo qInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = out.hasGlobalPriority ? &globalPrioInfo : nullptr,
        .queueFamilyIndex = out.computeFamilyIdx,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    const VkDeviceCreateInfo devInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = devPNext,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qInfo,
        .enabledExtensionCount = static_cast<uint32_t>(enabledExts.size()),
        .ppEnabledExtensionNames = enabledExts.data(),
        .pEnabledFeatures = &enabled,
    };
    VkResult devRes = vkCreateDevice(out.physicalDevice, &devInfo, nullptr, &out.device);
    if (devRes != VK_SUCCESS && out.hasGlobalPriority) {
        LOGW("vkCreateDevice with HIGH global priority returned %d; retrying with default priority",
             static_cast<int>(devRes));
        qInfo.pNext = nullptr;
        out.hasGlobalPriority = false;
        devRes = vkCreateDevice(out.physicalDevice, &devInfo, nullptr, &out.device);
    }
    if (devRes != VK_SUCCESS) {
        LOGE("vkCreateDevice failed rc=%d", static_cast<int>(devRes));
        return kSessionDeviceCreateFailed;
    }
    // Populate per-device table — these pointers stay valid for our device
    // even after framegen's volkLoadDevice() clobbers the globals.
    volkLoadDeviceTable(&out.fn, out.device);

    out.fn.vkGetDeviceQueue(out.device, out.computeFamilyIdx, 0, &out.computeQueue);

    const VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = out.computeFamilyIdx,
    };
    if (out.fn.vkCreateCommandPool(out.device, &poolInfo, nullptr, &out.commandPool) != VK_SUCCESS) {
        LOGE("vkCreateCommandPool failed");
        return kSessionDeviceCreateFailed;
    }

    // Pre-allocate the ring. A single vkAllocateCommandBuffers for all N slots
    // is cheaper than N individual calls, and since the pool has
    // RESET_COMMAND_BUFFER_BIT we can reset each CB independently in
    // acquireCommandRing().
    VkCommandBufferAllocateInfo cbai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = out.commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = kCommandRingSize,
    };
    if (out.fn.vkAllocateCommandBuffers(out.device, &cbai,
            out.ringCommandBuffers.data()) != VK_SUCCESS) {
        LOGE("vkAllocateCommandBuffers(ring) failed");
        return kSessionDeviceCreateFailed;
    }
    for (uint32_t i = 0; i < kCommandRingSize; ++i) {
        const VkFenceCreateInfo fi{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            // Unsignaled: first acquireCommandRing on this slot skips the wait.
        };
        if (out.fn.vkCreateFence(out.device, &fi, nullptr, &out.ringFences[i]) != VK_SUCCESS) {
            LOGE("vkCreateFence(ring %u) failed", i);
            return kSessionDeviceCreateFailed;
        }
        out.ringFenceArmed[i] = false;
    }
    out.ringNext = 0;

    if (out.fn.vkCreateCommandPool(out.device, &poolInfo, nullptr, &out.presentCommandPool) != VK_SUCCESS) {
        LOGE("vkCreateCommandPool(present) failed");
        return kSessionDeviceCreateFailed;
    }
    VkCommandBufferAllocateInfo pcbai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = out.presentCommandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = kPresentRingSize,
    };
    if (out.fn.vkAllocateCommandBuffers(out.device, &pcbai,
            out.presentCommandBuffers.data()) != VK_SUCCESS) {
        LOGE("vkAllocateCommandBuffers(present ring) failed");
        return kSessionDeviceCreateFailed;
    }
    for (uint32_t i = 0; i < kPresentRingSize; ++i) {
        const VkFenceCreateInfo fi{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        };
        if (out.fn.vkCreateFence(out.device, &fi, nullptr, &out.presentFences[i]) != VK_SUCCESS) {
            LOGE("vkCreateFence(present ring %u) failed", i);
            return kSessionDeviceCreateFailed;
        }
        out.presentFenceArmed[i] = false;
    }
    out.presentRingNext = 0;

    LOGI("Vulkan session ready (compute family=%u, %zu extensions, ReallyZero=%s, fp16=%d, subgroup=%u, cubic=%d, highPrio=%d, swapchain=%s, displayTiming=%s)",
         out.computeFamilyIdx, enabledExts.size(),
         out.supportsZeroFG ? "yes" : "no",
         (int)out.zerofgCaps.shader_float16_enabled,
         out.zerofgCaps.subgroup_size,
         (int)out.zerofgCaps.filter_cubic_enabled,
         (int)out.hasGlobalPriority,
         out.hasSwapchain ? "yes" : "no",
         out.hasDisplayTiming ? "yes" : "no");
    return kOk;
}

bool acquireCommandRing(VulkanSession &vk, VkCommandBuffer &outCb, VkFence &outFence) {
    if (vk.device == VK_NULL_HANDLE) return false;
    const uint32_t idx = vk.ringNext;
    vk.ringNext = (vk.ringNext + 1) % kCommandRingSize;

    VkFence fence = vk.ringFences[idx];
    VkCommandBuffer cb = vk.ringCommandBuffers[idx];

    // Wait for the previous submission on this slot to retire — only if it
    // was actually armed. First use of each slot skips the wait because the
    // fence starts unsignaled and no previous submit targeted it.
    //
    // Don't proceed past a timeout: vkResetFences on an unsignaled (still
    // pending) fence is undefined behaviour per the Vulkan spec and has been
    // observed to corrupt driver state on Adreno, killing the process on a
    // subsequent submit. Better to fail the acquire and let the caller skip
    // this frame.
    if (vk.ringFenceArmed[idx]) {
        // 500 ms is a generous upper bound: a frame that takes this long on
        // the compute queue is catastrophic and we want to surface it as a
        // warning rather than deadlock forever.
        const VkResult r = vk.fn.vkWaitForFences(vk.device, 1, &fence,
                                                 VK_TRUE, 500ULL * 1'000'000ULL);
        if (r != VK_SUCCESS) {
            LOGW("acquireCommandRing: vkWaitForFences slot=%u returned %d — skipping frame", idx, (int)r);
            return false;
        }
        if (vk.fn.vkResetFences(vk.device, 1, &fence) != VK_SUCCESS) {
            LOGW("acquireCommandRing: vkResetFences slot=%u failed", idx);
            return false;
        }
        vk.ringFenceArmed[idx] = false;
    }

    if (vk.fn.vkResetCommandBuffer(cb, 0) != VK_SUCCESS) {
        LOGW("vkResetCommandBuffer failed on ring slot %u", idx);
        return false;
    }
    outCb = cb;
    outFence = fence;
    return true;
}

bool acquirePresentRing(VulkanSession &vk, VkCommandBuffer &outCb, VkFence &outFence) {
    if (vk.device == VK_NULL_HANDLE) return false;
    const uint32_t idx = vk.presentRingNext;
    vk.presentRingNext = (vk.presentRingNext + 1) % kPresentRingSize;

    VkFence fence = vk.presentFences[idx];
    VkCommandBuffer cb = vk.presentCommandBuffers[idx];

    if (vk.presentFenceArmed[idx]) {
        const VkResult r = vk.fn.vkWaitForFences(vk.device, 1, &fence,
                                                 VK_TRUE, 500ULL * 1'000'000ULL);
        if (r != VK_SUCCESS) {
            LOGW("acquirePresentRing: vkWaitForFences slot=%u returned %d — skipping frame", idx, (int)r);
            return false;
        }
        if (vk.fn.vkResetFences(vk.device, 1, &fence) != VK_SUCCESS) {
            LOGW("acquirePresentRing: vkResetFences slot=%u failed", idx);
            return false;
        }
        vk.presentFenceArmed[idx] = false;
    }

    if (vk.fn.vkResetCommandBuffer(cb, 0) != VK_SUCCESS) {
        LOGW("vkResetCommandBuffer failed on present ring slot %u", idx);
        return false;
    }
    outCb = cb;
    outFence = fence;
    return true;
}

void markPresentFenceArmed(VulkanSession &vk, VkFence fence) {
    for (uint32_t i = 0; i < kPresentRingSize; ++i) {
        if (vk.presentFences[i] == fence) {
            vk.presentFenceArmed[i] = true;
            break;
        }
    }
}

bool submitCommandRing(VulkanSession &vk, VkCommandBuffer cb, VkFence fence) {
    if (vk.device == VK_NULL_HANDLE || cb == VK_NULL_HANDLE) return false;
    const VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    const VkResult r = vk.fn.vkQueueSubmit(vk.computeQueue, 1, &si, fence);
    if (r != VK_SUCCESS) {
        LOGW("vkQueueSubmit failed rc=%d", (int)r);
        return false;
    }
    // Mark the slot whose fence we just armed. Find it by scanning — ring is
    // tiny (8 slots) so linear search is noise vs the submit cost itself.
    for (uint32_t i = 0; i < kCommandRingSize; ++i) {
        if (vk.ringFences[i] == fence) {
            vk.ringFenceArmed[i] = true;
            break;
        }
    }
    return true;
}

bool waitCommandRing(VulkanSession &vk, VkFence fence) {
    if (vk.device == VK_NULL_HANDLE || fence == VK_NULL_HANDLE) return true;
    // Look up whether this fence is actually armed; if not, the caller is
    // waiting on something that never got submitted (e.g. record failed and
    // no submit happened) — treat as trivially done.
    bool armed = false;
    for (uint32_t i = 0; i < kCommandRingSize; ++i) {
        if (vk.ringFences[i] == fence && vk.ringFenceArmed[i]) {
            armed = true;
            break;
        }
    }
    if (!armed) return true;

    const VkResult r = vk.fn.vkWaitForFences(vk.device, 1, &fence,
                                             VK_TRUE, 500ULL * 1'000'000ULL);
    if (r != VK_SUCCESS) {
        LOGW("waitCommandRing: vkWaitForFences returned %d", (int)r);
        return false;
    }
    return true;
}

void destroy_session(VulkanSession &s) {
    if (s.device != VK_NULL_HANDLE) {
        // Drain any in-flight ring work before freeing fences. Safe even if
        // no CBs were ever submitted: unarmed fences aren't waited on.
        if (s.fn.vkWaitForFences != nullptr) {
            for (uint32_t i = 0; i < kCommandRingSize; ++i) {
                if (s.ringFences[i] != VK_NULL_HANDLE && s.ringFenceArmed[i]) {
                    s.fn.vkWaitForFences(s.device, 1, &s.ringFences[i],
                                         VK_TRUE, 500ULL * 1'000'000ULL);
                    s.ringFenceArmed[i] = false;
                }
            }
            for (uint32_t i = 0; i < kPresentRingSize; ++i) {
                if (s.presentFences[i] != VK_NULL_HANDLE && s.presentFenceArmed[i]) {
                    s.fn.vkWaitForFences(s.device, 1, &s.presentFences[i],
                                         VK_TRUE, 500ULL * 1'000'000ULL);
                    s.presentFenceArmed[i] = false;
                }
            }
        }
        if (s.fn.vkDestroyFence != nullptr) {
            for (uint32_t i = 0; i < kCommandRingSize; ++i) {
                if (s.ringFences[i] != VK_NULL_HANDLE) {
                    s.fn.vkDestroyFence(s.device, s.ringFences[i], nullptr);
                    s.ringFences[i] = VK_NULL_HANDLE;
                }
                s.ringCommandBuffers[i] = VK_NULL_HANDLE; // owned by the pool
            }
            for (uint32_t i = 0; i < kPresentRingSize; ++i) {
                if (s.presentFences[i] != VK_NULL_HANDLE) {
                    s.fn.vkDestroyFence(s.device, s.presentFences[i], nullptr);
                    s.presentFences[i] = VK_NULL_HANDLE;
                }
                s.presentCommandBuffers[i] = VK_NULL_HANDLE;
            }
        }
        if (s.commandPool != VK_NULL_HANDLE && s.fn.vkDestroyCommandPool != nullptr) {
            s.fn.vkDestroyCommandPool(s.device, s.commandPool, nullptr);
            s.commandPool = VK_NULL_HANDLE;
        }
        if (s.presentCommandPool != VK_NULL_HANDLE && s.fn.vkDestroyCommandPool != nullptr) {
            s.fn.vkDestroyCommandPool(s.device, s.presentCommandPool, nullptr);
            s.presentCommandPool = VK_NULL_HANDLE;
        }
        if (s.fn.vkDeviceWaitIdle != nullptr) s.fn.vkDeviceWaitIdle(s.device);
        if (s.fn.vkDestroyDevice != nullptr) {
            s.fn.vkDestroyDevice(s.device, nullptr);
        } else {
            // Fallback to global if our table was never populated.
            vkDestroyDevice(s.device, nullptr);
        }
        s.device = VK_NULL_HANDLE;
    }
    if (s.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(s.instance, nullptr);
        s.instance = VK_NULL_HANDLE;
    }
    s.physicalDevice = VK_NULL_HANDLE;
    s.computeQueue = VK_NULL_HANDLE;
    s.computeFamilyIdx = UINT32_MAX;
    s.deviceUuid = 0;
    s.supportsZeroFG = false;
    s.apiVersion = VK_API_VERSION_1_0;
    s.hasQueueFamilyForeign = false;
    s.hasSwapchain = false;
    s.hasDisplayTiming = false;
    s.hasGlobalPriority = false;
    s.zerofgCaps = {};
    s.ringNext = 0;
    s.ringCommandBuffers.fill(VK_NULL_HANDLE);
    s.ringFences.fill(VK_NULL_HANDLE);
    s.ringFenceArmed.fill(false);
    s.presentRingNext = 0;
    s.presentCommandBuffers.fill(VK_NULL_HANDLE);
    s.presentFences.fill(VK_NULL_HANDLE);
    s.presentFenceArmed.fill(false);
    s.fn = {};
}

} // namespace lsfg_android
