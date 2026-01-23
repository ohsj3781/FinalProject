package com.example.aibenchmark

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

/**
 * Model Benchmark class for ExecuTorch inference
 *
 * Handles model loading, image preprocessing, and inference timing.
 * Supports multiple backends: CPU (XNNPACK), NPU (QNN), GPU (Vulkan)
 *
 * @param context Application context
 * @param modelPath Path to the model file (asset name or absolute file path)
 * @param isAssetModel If true, modelPath is treated as an asset name; otherwise as an external file path
 */
class ModelBenchmark(
    private val context: Context,
    private val modelPath: String,
    private val isAssetModel: Boolean = true
) : Closeable {

    companion object {
        private const val TAG = "ModelBenchmark"

        /**
         * Load NNAPI backend libraries
         * NNAPI is the recommended backend for Samsung Exynos and other Android NPUs
         */
        fun loadNnapiBackend(): Boolean {
            return try {
                // NNAPI backend library
                System.loadLibrary("nnapi_backend")
                Log.i(TAG, "NNAPI backend loaded successfully")
                true
            } catch (e: UnsatisfiedLinkError) {
                Log.w(TAG, "NNAPI backend not available: ${e.message}")
                // Try alternative library name
                try {
                    System.loadLibrary("executorch_nnapi")
                    Log.i(TAG, "NNAPI backend loaded (alternative)")
                    true
                } catch (e2: UnsatisfiedLinkError) {
                    Log.w(TAG, "NNAPI backend not available (alternative): ${e2.message}")
                    false
                }
            }
        }

        /**
         * Load QNN backend libraries if available
         * Call this before creating ModelBenchmark for QNN models
         */
        fun loadQnnBackend(): Boolean {
            return try {
                // Try to load QNN delegate library
                System.loadLibrary("qnn_executorch_backend")
                Log.i(TAG, "QNN backend loaded successfully")
                true
            } catch (e: UnsatisfiedLinkError) {
                Log.w(TAG, "QNN backend not available: ${e.message}")
                false
            }
        }

        /**
         * Load Vulkan backend libraries if available
         */
        fun loadVulkanBackend(): Boolean {
            return try {
                System.loadLibrary("vulkan_executorch_backend")
                Log.i(TAG, "Vulkan backend loaded successfully")
                true
            } catch (e: UnsatisfiedLinkError) {
                Log.w(TAG, "Vulkan backend not available: ${e.message}")
                false
            }
        }

        /**
         * Load all available backends
         * Returns a map of backend type to load status
         */
        fun loadAllBackends(): Map<String, Boolean> {
            return mapOf(
                "NNAPI" to loadNnapiBackend(),
                "QNN" to loadQnnBackend(),
                "Vulkan" to loadVulkanBackend()
            )
        }

        /**
         * Get device information for debugging
         */
        fun getDeviceInfo(): String {
            return buildString {
                appendLine("Device: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}")
                appendLine("Hardware: ${android.os.Build.HARDWARE}")
                appendLine("Board: ${android.os.Build.BOARD}")
                appendLine("SOC: ${getSocInfo()}")
                appendLine("Android: ${android.os.Build.VERSION.SDK_INT}")
            }
        }

        private fun getSocInfo(): String {
            return try {
                val socId = File("/sys/devices/soc0/soc_id").readText().trim()
                val family = try {
                    File("/sys/devices/soc0/family").readText().trim()
                } catch (e: Exception) { "Unknown" }
                "$family (ID: $socId)"
            } catch (e: Exception) {
                "Unknown"
            }
        }
    }

    private var module: Module? = null

    // Detected backend based on model name
    val detectedBackend: BackendType = BackendType.fromModelName(modelPath)

    // Image preprocessing constants (ImageNet normalization)
    private val inputSize = 224
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    // COCO categories (80 classes)
    private val cocoCategories = arrayOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    )

    init {
        loadModel()
    }

    /**
     * Load the ExecuTorch model from assets or external storage
     */
    private fun loadModel() {
        val modelFilePath: String

        if (isAssetModel) {
            // Load from assets - copy to cache first
            val modelFileName = File(modelPath).name
            val modelFile = File(context.cacheDir, modelFileName)

            // Copy model from assets to cache if not exists or if size differs
            val needsCopy = !modelFile.exists() || modelFile.length() == 0L
            if (needsCopy) {
                context.assets.open(modelPath).use { input ->
                    FileOutputStream(modelFile).use { output ->
                        input.copyTo(output)
                    }
                }
            }
            modelFilePath = modelFile.absolutePath
        } else {
            // Load from external storage directly
            val externalFile = File(modelPath)
            if (!externalFile.exists()) {
                throw IllegalArgumentException("Model file not found: $modelPath")
            }
            modelFilePath = externalFile.absolutePath
        }

        module = Module.load(modelFilePath)
    }

    /**
     * Run inference on an image and return inference time in milliseconds
     *
     * @param imagePath Path to the image file
     * @return Inference time in milliseconds
     */
    fun runInference(imagePath: String): Long {
        // Load and preprocess image
        val bitmap = loadAndResizeBitmap(imagePath)
        val inputTensor = preprocessBitmap(bitmap)

        // Measure inference time
        val startTime = System.nanoTime()
        val outputEValue = module?.forward(EValue.from(inputTensor))
        val endTime = System.nanoTime()

        // Calculate inference time in milliseconds
        val inferenceTimeMs = (endTime - startTime) / 1_000_000

        return inferenceTimeMs
    }

    /**
     * Run inference and return predictions
     *
     * @param imagePath Path to the image file
     * @param threshold Confidence threshold for predictions
     * @return Pair of (inference time ms, list of predicted labels with confidence)
     */
    fun runInferenceWithResults(
        imagePath: String,
        threshold: Float = 0.5f
    ): Pair<Long, List<Pair<String, Float>>> {
        // Load and preprocess image
        val bitmap = loadAndResizeBitmap(imagePath)
        val inputTensor = preprocessBitmap(bitmap)

        // Measure inference time
        val startTime = System.nanoTime()
        val outputEValue = module?.forward(EValue.from(inputTensor))
        val endTime = System.nanoTime()

        val inferenceTimeMs = (endTime - startTime) / 1_000_000

        // Process output
        val predictions = mutableListOf<Pair<String, Float>>()

        outputEValue?.firstOrNull()?.let { output ->
            if (output.isTensor()) {
                val outputTensor = output.toTensor()
                val scores = outputTensor.dataAsFloatArray

                // Apply sigmoid and threshold
                for (i in scores.indices) {
                    val probability = sigmoid(scores[i])
                    if (probability >= threshold && i < cocoCategories.size) {
                        predictions.add(cocoCategories[i] to probability)
                    }
                }

                // Sort by confidence
                predictions.sortByDescending { it.second }
            }
        }

        return inferenceTimeMs to predictions
    }

    /**
     * Load and resize bitmap to input size
     */
    private fun loadAndResizeBitmap(imagePath: String): Bitmap {
        val options = BitmapFactory.Options().apply {
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }

        val originalBitmap = BitmapFactory.decodeFile(imagePath, options)
            ?: throw IllegalArgumentException("Cannot load image: $imagePath")

        // Resize to inputSize x inputSize
        return Bitmap.createScaledBitmap(originalBitmap, inputSize, inputSize, true).also {
            if (it != originalBitmap) {
                originalBitmap.recycle()
            }
        }
    }

    /**
     * Preprocess bitmap for model input
     * - Normalize with ImageNet mean and std
     * - Convert to CHW format (Channel, Height, Width)
     */
    private fun preprocessBitmap(bitmap: Bitmap): Tensor {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Create float buffer for tensor data (CHW format)
        val floatBuffer = FloatBuffer.allocate(3 * height * width)

        // Convert to normalized float tensor in CHW format
        for (c in 0 until 3) {
            for (h in 0 until height) {
                for (w in 0 until width) {
                    val pixel = pixels[h * width + w]

                    val value = when (c) {
                        0 -> ((pixel shr 16) and 0xFF) / 255.0f  // R
                        1 -> ((pixel shr 8) and 0xFF) / 255.0f   // G
                        2 -> (pixel and 0xFF) / 255.0f           // B
                        else -> 0f
                    }

                    // Normalize with ImageNet mean and std
                    val normalizedValue = (value - mean[c]) / std[c]
                    floatBuffer.put(normalizedValue)
                }
            }
        }

        floatBuffer.rewind()

        // Create tensor with shape [1, 3, 224, 224]
        return Tensor.fromBlob(
            floatBuffer.array(),
            longArrayOf(1, 3, height.toLong(), width.toLong())
        )
    }

    /**
     * Sigmoid activation function
     */
    private fun sigmoid(x: Float): Float {
        return (1.0 / (1.0 + Math.exp(-x.toDouble()))).toFloat()
    }

    /**
     * Get model information including backend
     */
    fun getModelInfo(): String {
        return buildString {
            appendLine("ExecuTorch Model")
            appendLine("File: ${File(modelPath).name}")
            appendLine("Backend: ${detectedBackend.displayName}")
            appendLine("Input: ${inputSize}x${inputSize}x3")
            appendLine("Classes: ${cocoCategories.size}")
        }
    }

    /**
     * Get backend display name
     */
    fun getBackendName(): String = detectedBackend.displayName

    override fun close() {
        module?.destroy()
        module = null
    }
}
