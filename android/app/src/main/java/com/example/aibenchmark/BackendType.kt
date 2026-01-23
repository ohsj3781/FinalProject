package com.example.aibenchmark

import android.os.Build
import java.io.File

/**
 * Supported compute backends for model inference
 */
enum class BackendType(
    val displayName: String,
    val description: String,
    val modelSuffix: String
) {
    /**
     * CPU backend using XNNPACK for optimized CPU inference
     */
    XNNPACK(
        displayName = "CPU (XNNPACK)",
        description = "Optimized CPU inference using XNNPACK",
        modelSuffix = "_xnnpack"
    ),

    /**
     * Qualcomm NPU backend using QNN (Qualcomm Neural Network)
     * Requires Snapdragon chipset with HTP (Hexagon Tensor Processor)
     */
    QNN(
        displayName = "NPU (Qualcomm QNN)",
        description = "Qualcomm NPU acceleration using QNN SDK",
        modelSuffix = "_qnn"
    ),

    /**
     * Android NNAPI backend - universal NPU/DSP/GPU acceleration
     * Supports Samsung Exynos, Qualcomm, MediaTek, and other Android NPUs
     */
    NNAPI(
        displayName = "NPU (NNAPI)",
        description = "Android Neural Networks API - universal hardware acceleration",
        modelSuffix = "_nnapi"
    ),

    /**
     * Samsung Exynos NPU backend using ENN (Exynos Neural Network)
     * Requires Exynos chipset with NPU
     */
    SAMSUNG_ENN(
        displayName = "NPU (Samsung ENN)",
        description = "Samsung NPU acceleration using ENN SDK",
        modelSuffix = "_enn"
    ),

    /**
     * Qualcomm GPU backend using QNN
     */
    QNN_GPU(
        displayName = "GPU (Qualcomm)",
        description = "Qualcomm GPU acceleration using QNN SDK",
        modelSuffix = "_qnn_gpu"
    ),

    /**
     * Samsung GPU backend using OpenCL
     */
    SAMSUNG_GPU(
        displayName = "GPU (Mali/Samsung)",
        description = "Samsung GPU acceleration using OpenCL",
        modelSuffix = "_opencl"
    ),

    /**
     * Vulkan GPU backend
     */
    VULKAN(
        displayName = "GPU (Vulkan)",
        description = "Cross-platform GPU acceleration using Vulkan",
        modelSuffix = "_vulkan"
    ),

    /**
     * Default portable backend (no optimization)
     */
    PORTABLE(
        displayName = "Portable (Default)",
        description = "Default portable backend without hardware acceleration",
        modelSuffix = ""
    );

    companion object {
        /**
         * Detect backend from model filename
         */
        fun fromModelName(modelName: String): BackendType {
            val lowerName = modelName.lowercase()
            return when {
                lowerName.contains("nnapi") -> NNAPI
                lowerName.contains("qnn_gpu") || lowerName.contains("qnn-gpu") -> QNN_GPU
                lowerName.contains("qnn") || lowerName.contains("htp") -> QNN
                lowerName.contains("enn") || lowerName.contains("samsung_npu") -> SAMSUNG_ENN
                lowerName.contains("opencl") || lowerName.contains("mali") -> SAMSUNG_GPU
                lowerName.contains("xnnpack") -> XNNPACK
                lowerName.contains("vulkan") -> VULKAN
                else -> PORTABLE
            }
        }

        /**
         * Get all backends available on this device
         */
        fun getAvailableBackends(): List<BackendType> {
            val available = mutableListOf<BackendType>()

            // XNNPACK (CPU) is always available
            available.add(XNNPACK)
            available.add(PORTABLE)

            // NNAPI is available on Android 8.1+ (API 27+)
            // It automatically uses the best available accelerator (NPU/DSP/GPU)
            if (Build.VERSION.SDK_INT >= 27) {
                available.add(NNAPI)
            }

            // Check for Qualcomm chipset (QNN support)
            if (isQualcommDevice()) {
                available.add(QNN)
                available.add(QNN_GPU)
            }

            // Check for Samsung Exynos chipset (ENN support)
            if (isSamsungExynosDevice()) {
                available.add(SAMSUNG_ENN)
                available.add(SAMSUNG_GPU)
            }

            // Vulkan is available on most modern Android devices
            available.add(VULKAN)

            return available
        }

        /**
         * Get detailed device info for debugging
         */
        fun getDeviceChipsetInfo(): ChipsetInfo {
            return ChipsetInfo(
                manufacturer = Build.MANUFACTURER,
                model = Build.MODEL,
                hardware = Build.HARDWARE,
                board = Build.BOARD,
                socManufacturer = getSocManufacturer(),
                socModel = getSocModel(),
                hasNpu = hasNpuSupport(),
                npuType = detectNpuType()
            )
        }

        private fun getSocManufacturer(): String {
            return try {
                File("/sys/devices/soc0/vendor").readText().trim()
            } catch (e: Exception) {
                try {
                    // Alternative path for some devices
                    File("/sys/firmware/devicetree/base/model").readText().trim()
                } catch (e2: Exception) {
                    "Unknown"
                }
            }
        }

        private fun getSocModel(): String {
            // Try multiple methods to get SoC model
            val methods = listOf(
                { File("/sys/devices/soc0/soc_id").readText().trim() },
                { File("/sys/devices/soc0/machine").readText().trim() },
                { Build.SOC_MODEL }, // Android 12+
                { Build.HARDWARE }
            )

            for (method in methods) {
                try {
                    val result = method()
                    if (result.isNotEmpty() && result != "unknown") {
                        return result
                    }
                } catch (e: Exception) {
                    continue
                }
            }
            return "Unknown"
        }

        private fun hasNpuSupport(): Boolean {
            return isQualcommDevice() || isSamsungExynosDevice() || isMediaTekDevice()
        }

        private fun detectNpuType(): String {
            return when {
                isQualcommDevice() -> "Qualcomm Hexagon NPU"
                isSamsungExynosDevice() -> "Samsung Exynos NPU"
                isMediaTekDevice() -> "MediaTek APU"
                else -> "None detected"
            }
        }

        /**
         * Check if device has Qualcomm Snapdragon chipset
         */
        private fun isQualcommDevice(): Boolean {
            val hardware = Build.HARDWARE.lowercase()
            val board = Build.BOARD.lowercase()

            // Check common Qualcomm identifiers
            if (hardware.contains("qcom") || hardware.contains("snapdragon")) return true
            if (board.contains("msm") || board.contains("sdm") || board.contains("sm")) return true

            // Check SOC info
            try {
                val socVendor = File("/sys/devices/soc0/vendor").readText().trim().lowercase()
                if (socVendor.contains("qualcomm")) return true
            } catch (e: Exception) { }

            return false
        }

        /**
         * Check if device has Samsung Exynos chipset
         */
        fun isSamsungExynosDevice(): Boolean {
            val hardware = Build.HARDWARE.lowercase()
            val board = Build.BOARD.lowercase()
            val manufacturer = Build.MANUFACTURER.lowercase()

            // Check common Exynos identifiers
            if (hardware.contains("exynos") || hardware.contains("samsungexynos")) return true
            if (board.contains("exynos") || board.contains("universal")) return true
            if (hardware.contains("mali") && manufacturer.contains("samsung")) return true

            // Check for specific Exynos hardware names
            val exynosPatterns = listOf("s5e", "exynos", "universal", "erd")
            if (exynosPatterns.any { board.contains(it) || hardware.contains(it) }) return true

            // Additional check using Build.SOC_MODEL (Android 12+)
            try {
                val socModel = Build.SOC_MODEL.lowercase()
                if (socModel.contains("exynos") || socModel.contains("s5e")) return true
            } catch (e: Exception) { }

            // Check via /proc/cpuinfo for Exynos
            try {
                val cpuInfo = File("/proc/cpuinfo").readText().lowercase()
                if (cpuInfo.contains("exynos") || cpuInfo.contains("samsung")) return true
            } catch (e: Exception) { }

            return false
        }

        /**
         * Check if device has MediaTek chipset
         */
        private fun isMediaTekDevice(): Boolean {
            val hardware = Build.HARDWARE.lowercase()
            val board = Build.BOARD.lowercase()

            return hardware.contains("mt") ||
                    hardware.contains("mediatek") ||
                    board.contains("mt") ||
                    board.contains("mediatek")
        }
    }

    /**
     * Data class to hold chipset information
     */
    data class ChipsetInfo(
        val manufacturer: String,
        val model: String,
        val hardware: String,
        val board: String,
        val socManufacturer: String,
        val socModel: String,
        val hasNpu: Boolean,
        val npuType: String
    ) {
        override fun toString(): String {
            return buildString {
                appendLine("Manufacturer: $manufacturer")
                appendLine("Model: $model")
                appendLine("Hardware: $hardware")
                appendLine("Board: $board")
                appendLine("SoC: $socManufacturer $socModel")
                appendLine("NPU: $npuType")
                appendLine("NPU Available: $hasNpu")
            }
        }
    }
}
