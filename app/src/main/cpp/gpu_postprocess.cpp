#include "gpu_postprocess.hpp"
#include "gpu_postprocess_spv.hpp"

#include <algorithm>

namespace lsfg_android {

namespace {

struct PushConstants {
    int32_t srcW;
    int32_t srcH;
    int32_t dstW;
    int32_t dstH;
    int32_t method;
    float sharpness;
    float strength;
    float pad0;
};

VkImageView makeImageView(VulkanSession &vk, const AhbImage &img) {
    const VkImageViewCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = img.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = img.format,
        .components = {
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageView view = VK_NULL_HANDLE;
    if (vk.fn.vkCreateImageView(vk.device, &info, nullptr, &view) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return view;
}

} // namespace

bool GpuPostProcessor::ensurePipeline(VulkanSession &vk) {
    if (pipeline != VK_NULL_HANDLE) return true;

    VkDescriptorSetLayoutBinding bindings[2]{
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dslInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = bindings,
    };
    if (vk.fn.vkCreateDescriptorSetLayout(vk.device, &dslInfo, nullptr,
            &descriptorSetLayout) != VK_SUCCESS) {
        reset(vk);
        return false;
    }

    const VkPushConstantRange pushRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };
    const VkPipelineLayoutCreateInfo plInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushRange,
    };
    if (vk.fn.vkCreatePipelineLayout(vk.device, &plInfo, nullptr,
            &pipelineLayout) != VK_SUCCESS) {
        reset(vk);
        return false;
    }

    const VkShaderModuleCreateInfo smInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = kGpuPostprocessCompSpvSize,
        .pCode = kGpuPostprocessCompSpv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    if (vk.fn.vkCreateShaderModule(vk.device, &smInfo, nullptr, &shader) != VK_SUCCESS) {
        reset(vk);
        return false;
    }
    const VkPipelineShaderStageCreateInfo stage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader,
        .pName = "main",
    };
    const VkComputePipelineCreateInfo pipeInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage,
        .layout = pipelineLayout,
    };
    const bool pipeOk = vk.fn.vkCreateComputePipelines(
        vk.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline) == VK_SUCCESS;
    vk.fn.vkDestroyShaderModule(vk.device, shader, nullptr);
    if (!pipeOk) {
        reset(vk);
        return false;
    }

    const VkDescriptorPoolSize poolSize{
        .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 2,
    };
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    if (vk.fn.vkCreateDescriptorPool(vk.device, &poolInfo, nullptr,
            &descriptorPool) != VK_SUCCESS) {
        reset(vk);
        return false;
    }

    const VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptorSetLayout,
    };
    if (vk.fn.vkAllocateDescriptorSets(vk.device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        reset(vk);
        return false;
    }
    return true;
}

bool GpuPostProcessor::process(VulkanSession &vk,
                               std::mutex &queueMu,
                               const AhbImage &src,
                               const AhbImage &dst,
                               const GpuPostProcessConfig &config) {
    if (src.image == VK_NULL_HANDLE || dst.image == VK_NULL_HANDLE) return false;
    if (!ensurePipeline(vk)) return false;

    VkImageView srcView = makeImageView(vk, src);
    VkImageView dstView = makeImageView(vk, dst);
    if (srcView == VK_NULL_HANDLE || dstView == VK_NULL_HANDLE) {
        if (srcView != VK_NULL_HANDLE) vk.fn.vkDestroyImageView(vk.device, srcView, nullptr);
        if (dstView != VK_NULL_HANDLE) vk.fn.vkDestroyImageView(vk.device, dstView, nullptr);
        return false;
    }

    const VkDescriptorImageInfo srcInfo{
        .imageView = srcView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo dstInfo{
        .imageView = dstView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    VkWriteDescriptorSet writes[2]{
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &srcInfo,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &dstInfo,
        },
    };
    vk.fn.vkUpdateDescriptorSets(vk.device, 2, writes, 0, nullptr);

    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    if (!acquirePresentRing(vk, cb, fence)) {
        vk.fn.vkDestroyImageView(vk.device, srcView, nullptr);
        vk.fn.vkDestroyImageView(vk.device, dstView, nullptr);
        return false;
    }

    const VkCommandBufferBeginInfo bi{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vk.fn.vkBeginCommandBuffer(cb, &bi);

    // Preserve GENERAL layout on src and dst (CRITICAL on Mali AFBC/AFRC:
    // oldLayout=UNDEFINED on src would discard/corrupt image contents).
    VkImageMemoryBarrier toSrc{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = src.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageMemoryBarrier toDst{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = dst.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkImageMemoryBarrier pre[2] = {toSrc, toDst};
    vk.fn.vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 2, pre);

    const PushConstants pc{
        .srcW = static_cast<int32_t>(src.extent.width),
        .srcH = static_cast<int32_t>(src.extent.height),
        .dstW = static_cast<int32_t>(dst.extent.width),
        .dstH = static_cast<int32_t>(dst.extent.height),
        .method = std::clamp(config.method, 0, 16),
        .sharpness = std::clamp(config.sharpness, 0.0f, 1.0f),
        .strength = std::clamp(config.strength, 0.0f, 1.0f),
        .pad0 = 0.0f,
    };
    vk.fn.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vk.fn.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                                  0, 1, &descriptorSet, 0, nullptr);
    vk.fn.vkCmdPushConstants(cb, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                             0, sizeof(PushConstants), &pc);
    vk.fn.vkCmdDispatch(cb,
        (dst.extent.width  + 15) / 16,
        (dst.extent.height +  7) /  8,
        1);

    VkImageMemoryBarrier postDst{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = dst.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vk.fn.vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &postDst);

    vk.fn.vkEndCommandBuffer(cb);
    const VkSubmitInfo si{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    bool ok = false;
    {
        std::lock_guard<std::mutex> qLock(queueMu);
        ok = vk.fn.vkQueueSubmit(vk.computeQueue, 1, &si, fence) == VK_SUCCESS;
    }
    if (ok) {
        markPresentFenceArmed(vk, fence);
        ok = vk.fn.vkWaitForFences(vk.device, 1, &fence, VK_TRUE, 500ULL * 1'000'000ULL) == VK_SUCCESS;
    }
    vk.fn.vkDestroyImageView(vk.device, srcView, nullptr);
    vk.fn.vkDestroyImageView(vk.device, dstView, nullptr);
    return ok;
}

void GpuPostProcessor::reset(VulkanSession &vk) {
    if (vk.device == VK_NULL_HANDLE) {
        descriptorSetLayout = VK_NULL_HANDLE;
        pipelineLayout = VK_NULL_HANDLE;
        pipeline = VK_NULL_HANDLE;
        descriptorPool = VK_NULL_HANDLE;
        descriptorSet = VK_NULL_HANDLE;
        return;
    }
    if (descriptorPool != VK_NULL_HANDLE && vk.fn.vkDestroyDescriptorPool != nullptr) {
        vk.fn.vkDestroyDescriptorPool(vk.device, descriptorPool, nullptr);
    }
    descriptorPool = VK_NULL_HANDLE;
    descriptorSet = VK_NULL_HANDLE;
    if (pipeline != VK_NULL_HANDLE && vk.fn.vkDestroyPipeline != nullptr) {
        vk.fn.vkDestroyPipeline(vk.device, pipeline, nullptr);
    }
    pipeline = VK_NULL_HANDLE;
    if (pipelineLayout != VK_NULL_HANDLE && vk.fn.vkDestroyPipelineLayout != nullptr) {
        vk.fn.vkDestroyPipelineLayout(vk.device, pipelineLayout, nullptr);
    }
    pipelineLayout = VK_NULL_HANDLE;
    if (descriptorSetLayout != VK_NULL_HANDLE && vk.fn.vkDestroyDescriptorSetLayout != nullptr) {
        vk.fn.vkDestroyDescriptorSetLayout(vk.device, descriptorSetLayout, nullptr);
    }
    descriptorSetLayout = VK_NULL_HANDLE;
}

} // namespace lsfg_android
