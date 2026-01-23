package com.example.aibenchmark

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.content.Intent
import android.net.Uri
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.aibenchmark.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Main Activity for AI Model Benchmark Application
 *
 * This application measures inference performance of ExecuTorch models
 * using COCO 2017 validation dataset.
 */
/**
 * Represents a model option for selection
 */
sealed class ModelOption(val displayName: String) {
    data class AssetModel(val assetPath: String, val name: String) : ModelOption(name)
    data class ExternalModel(val filePath: String) : ModelOption("External: ${File(filePath).name}")
    object CustomPath : ModelOption("Custom path...")
}

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var modelBenchmark: ModelBenchmark? = null
    private var isBenchmarkRunning = false

    // Model selection
    private val availableModels = mutableListOf<ModelOption>()
    private var selectedModel: ModelOption? = null

    // Default paths - Internal storage Pictures folder
    private val defaultCocoPath: String by lazy {
        "${Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)}/coco/val2017"
    }
    private val defaultExternalModelPath: String by lazy {
        "${Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)}/models"
    }

    // Permission launcher
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (allGranted) {
            showStatus("Permissions granted")
        } else {
            showStatus("Storage permission required")
        }
    }

    // Manage storage permission for Android 11+
    private val manageStorageLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (Environment.isExternalStorageManager()) {
                showStatus("Storage access granted")
            } else {
                showStatus("Storage access denied")
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupUI()
        checkPermissions()
    }

    private fun setupUI() {
        // Set default paths
        binding.editTextCocoPath.setText(defaultCocoPath)

        // Setup device info display
        setupDeviceInfo()

        // Setup model selection
        setupModelSelection()

        // Start benchmark button
        binding.buttonStartBenchmark.setOnClickListener {
            if (!isBenchmarkRunning) {
                startBenchmark()
            } else {
                showStatus("Benchmark already running...")
            }
        }

        // Stop benchmark button
        binding.buttonStopBenchmark.setOnClickListener {
            stopBenchmark()
        }

        // Initially disable stop button
        binding.buttonStopBenchmark.isEnabled = false
    }

    private fun setupDeviceInfo() {
        // Get detailed chipset information
        val chipsetInfo = BackendType.getDeviceChipsetInfo()
        binding.textViewDeviceInfo.text = chipsetInfo.toString()

        // Check NPU availability
        val availableBackends = BackendType.getAvailableBackends()
        val hasQualcommNpu = availableBackends.contains(BackendType.QNN)
        val hasSamsungNpu = availableBackends.contains(BackendType.SAMSUNG_ENN)

        when {
            hasSamsungNpu -> {
                binding.textViewNpuStatus.text = "NPU: ${chipsetInfo.npuType}"
                binding.textViewNpuStatus.setTextColor(getColor(R.color.accent))
                showStatus("Samsung Exynos NPU detected. Use _enn models for NPU acceleration.")
            }
            hasQualcommNpu -> {
                binding.textViewNpuStatus.text = "NPU: ${chipsetInfo.npuType}"
                binding.textViewNpuStatus.setTextColor(getColor(R.color.accent))
            }
            else -> {
                binding.textViewNpuStatus.text = "NPU: Not detected (CPU only)"
                binding.textViewNpuStatus.setTextColor(getColor(android.R.color.darker_gray))
            }
        }

        // Try to load appropriate backend
        lifecycleScope.launch(Dispatchers.IO) {
            // Load all available backends
            val backendStatus = ModelBenchmark.loadAllBackends()

            // Determine which backend to use
            val nnapiLoaded = backendStatus["NNAPI"] == true
            val qnnLoaded = backendStatus["QNN"] == true

            withContext(Dispatchers.Main) {
                when {
                    nnapiLoaded && hasSamsungNpu -> {
                        binding.textViewNpuStatus.text = "NPU: Ready (NNAPI - Samsung)"
                        binding.textViewNpuStatus.setTextColor(getColor(R.color.accent))
                        showStatus("NNAPI loaded. Use *_nnapi.pte models for NPU acceleration.")
                    }
                    nnapiLoaded -> {
                        binding.textViewNpuStatus.text = "NPU: Ready (NNAPI)"
                        binding.textViewNpuStatus.setTextColor(getColor(R.color.accent))
                    }
                    qnnLoaded && hasQualcommNpu -> {
                        binding.textViewNpuStatus.text = "NPU: Ready (QNN - Qualcomm)"
                        binding.textViewNpuStatus.setTextColor(getColor(R.color.accent))
                    }
                    hasSamsungNpu || hasQualcommNpu -> {
                        binding.textViewNpuStatus.text = "NPU: Detected (Backend not loaded)"
                        binding.textViewNpuStatus.setTextColor(getColor(R.color.primary))
                    }
                }
            }
        }
    }

    private fun setupModelSelection() {
        // Scan for available models
        loadAvailableModels()

        // Setup spinner adapter
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            availableModels.map { it.displayName }
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.spinnerModelSelection.adapter = adapter

        // Handle model selection
        binding.spinnerModelSelection.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selected = availableModels.getOrNull(position) ?: return
                handleModelSelection(selected)
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                // Keep current selection
            }
        }

        // Browse button for custom model
        binding.buttonBrowseModel.setOnClickListener {
            // Open file picker or show path input
            val customPath = binding.editTextCustomModelPath.text.toString().trim()
            if (customPath.isNotEmpty()) {
                validateAndSelectCustomModel(customPath)
            } else {
                showStatus("Please enter a model path")
            }
        }

        // Select first model by default
        if (availableModels.isNotEmpty()) {
            selectedModel = availableModels[0]
            updateSelectedModelDisplay()
        }
    }

    private fun loadAvailableModels() {
        availableModels.clear()

        // 1. Scan assets folder for .pte files
        try {
            val assetFiles = assets.list("") ?: emptyArray()
            assetFiles.filter { it.endsWith(".pte") }
                .forEach { fileName ->
                    availableModels.add(ModelOption.AssetModel(fileName, "Asset: $fileName"))
                }
        } catch (e: Exception) {
            showStatus("Warning: Could not scan assets")
        }

        // 2. Scan external models folder
        val externalModelsDir = File(defaultExternalModelPath)
        if (externalModelsDir.exists() && externalModelsDir.isDirectory) {
            externalModelsDir.listFiles { file ->
                file.isFile && file.extension.lowercase() == "pte"
            }?.forEach { modelFile ->
                availableModels.add(ModelOption.ExternalModel(modelFile.absolutePath))
            }
        }

        // 3. Add custom path option
        availableModels.add(ModelOption.CustomPath)

        // If no models found, show warning
        if (availableModels.size == 1) { // Only CustomPath exists
            showStatus("Warning: No models found. Add .pte files to assets or ${defaultExternalModelPath}")
        }
    }

    private fun handleModelSelection(model: ModelOption) {
        when (model) {
            is ModelOption.AssetModel -> {
                selectedModel = model
                binding.layoutCustomModelPath.visibility = View.GONE
                updateSelectedModelDisplay()
            }
            is ModelOption.ExternalModel -> {
                selectedModel = model
                binding.layoutCustomModelPath.visibility = View.GONE
                updateSelectedModelDisplay()
            }
            is ModelOption.CustomPath -> {
                binding.layoutCustomModelPath.visibility = View.VISIBLE
                binding.textViewSelectedModel.text = "Selected: Enter custom path below"
            }
        }
    }

    private fun validateAndSelectCustomModel(path: String) {
        val modelFile = File(path)
        if (modelFile.exists() && modelFile.isFile && modelFile.extension.lowercase() == "pte") {
            selectedModel = ModelOption.ExternalModel(path)
            updateSelectedModelDisplay()
            showStatus("Custom model selected: ${modelFile.name}")
        } else {
            showStatus("Error: Invalid model file. Must be a .pte file")
            Toast.makeText(this, "Invalid model file", Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateSelectedModelDisplay() {
        val modelName = when (val model = selectedModel) {
            is ModelOption.AssetModel -> model.assetPath
            is ModelOption.ExternalModel -> File(model.filePath).name
            is ModelOption.CustomPath -> "Custom"
            null -> "None"
        }
        binding.textViewSelectedModel.text = "Selected: $modelName"

        // Update backend type based on model name
        val backend = BackendType.fromModelName(modelName)
        binding.textViewBackendType.text = backend.displayName

        // Update color based on backend type
        val backendColor = when (backend) {
            BackendType.QNN, BackendType.QNN_GPU -> R.color.accent
            BackendType.VULKAN -> R.color.primary
            else -> android.R.color.darker_gray
        }
        binding.textViewBackendType.setTextColor(getColor(backendColor))
    }

    private fun getSelectedModelPath(): String? {
        return when (val model = selectedModel) {
            is ModelOption.AssetModel -> model.assetPath
            is ModelOption.ExternalModel -> model.filePath
            is ModelOption.CustomPath -> binding.editTextCustomModelPath.text.toString().trim().ifEmpty { null }
            null -> null
        }
    }

    private fun isAssetModel(): Boolean {
        return selectedModel is ModelOption.AssetModel
    }

    private fun checkPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            // Android 11+ requires MANAGE_EXTERNAL_STORAGE
            if (!Environment.isExternalStorageManager()) {
                showStatus("Please grant storage access")
                val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
                intent.data = Uri.parse("package:$packageName")
                manageStorageLauncher.launch(intent)
            }
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            // Android 6-10
            val permissions = arrayOf(
                Manifest.permission.READ_EXTERNAL_STORAGE
            )
            val notGranted = permissions.filter {
                ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
            }
            if (notGranted.isNotEmpty()) {
                permissionLauncher.launch(notGranted.toTypedArray())
            }
        }
    }

    private fun startBenchmark() {
        // Validate model selection
        val modelPath = getSelectedModelPath()
        if (modelPath.isNullOrEmpty()) {
            showStatus("Error: Please select a model")
            Toast.makeText(this, "No model selected", Toast.LENGTH_SHORT).show()
            return
        }

        // For custom path, validate the file exists
        if (selectedModel is ModelOption.CustomPath || selectedModel is ModelOption.ExternalModel) {
            val modelFile = File(modelPath)
            if (!modelFile.exists()) {
                showStatus("Error: Model file not found at $modelPath")
                Toast.makeText(this, "Model file not found", Toast.LENGTH_SHORT).show()
                return
            }
        }

        val cocoPath = binding.editTextCocoPath.text.toString().trim()

        // Validate COCO path
        val cocoDir = File(cocoPath)
        if (!cocoDir.exists() || !cocoDir.isDirectory) {
            showStatus("Error: COCO directory not found at $cocoPath")
            Toast.makeText(this, "COCO directory not found", Toast.LENGTH_SHORT).show()
            return
        }

        // Count images
        val imageFiles = cocoDir.listFiles { file ->
            file.isFile && file.extension.lowercase() in listOf("jpg", "jpeg", "png")
        }

        if (imageFiles.isNullOrEmpty()) {
            showStatus("Error: No images found in $cocoPath")
            Toast.makeText(this, "No images found", Toast.LENGTH_SHORT).show()
            return
        }

        showStatus("Found ${imageFiles.size} images, Model: ${File(modelPath).name}")
        binding.textViewTotalImages.text = "${imageFiles.size}"

        // Start benchmark
        isBenchmarkRunning = true
        binding.buttonStartBenchmark.isEnabled = false
        binding.buttonStopBenchmark.isEnabled = true
        binding.progressBar.progress = 0

        lifecycleScope.launch {
            runBenchmark(modelPath, imageFiles.toList())
        }
    }

    private suspend fun runBenchmark(modelPath: String, imageFiles: List<File>) {
        withContext(Dispatchers.IO) {
            try {
                // Initialize model
                withContext(Dispatchers.Main) {
                    showStatus("Loading model: ${File(modelPath).name}...")
                }

                // Determine if model is from assets or external storage
                val isAsset = selectedModel is ModelOption.AssetModel
                modelBenchmark = ModelBenchmark(this@MainActivity, modelPath, isAsset)

                val backend = modelBenchmark?.detectedBackend ?: BackendType.PORTABLE
                withContext(Dispatchers.Main) {
                    showStatus("Model loaded (${backend.displayName}). Starting benchmark...")
                    binding.textViewBackendType.text = backend.displayName
                }

                val totalImages = imageFiles.size
                var processedImages = 0
                var totalInferenceTime = 0L
                val inferenceTimes = mutableListOf<Long>()

                val startTime = System.currentTimeMillis()

                for ((index, imageFile) in imageFiles.withIndex()) {
                    if (!isBenchmarkRunning) {
                        break
                    }

                    try {
                        // Run inference and measure time
                        val inferenceTime = modelBenchmark?.runInference(imageFile.absolutePath) ?: 0L
                        totalInferenceTime += inferenceTime
                        inferenceTimes.add(inferenceTime)
                        processedImages++

                        // Update UI every 10 images or at the end
                        if (processedImages % 10 == 0 || processedImages == totalImages) {
                            val progress = (processedImages * 100) / totalImages
                            val avgTime = if (processedImages > 0) totalInferenceTime / processedImages else 0
                            val elapsedTime = System.currentTimeMillis() - startTime

                            withContext(Dispatchers.Main) {
                                updateProgress(
                                    progress = progress,
                                    processed = processedImages,
                                    total = totalImages,
                                    avgInferenceTime = avgTime,
                                    totalElapsedTime = elapsedTime,
                                    lastInferenceTime = inferenceTime
                                )
                            }
                        }
                    } catch (e: Exception) {
                        // Skip failed images
                        withContext(Dispatchers.Main) {
                            showStatus("Warning: Failed to process ${imageFile.name}")
                        }
                    }
                }

                val endTime = System.currentTimeMillis()
                val totalTime = endTime - startTime

                // Calculate statistics
                val avgTime = if (processedImages > 0) totalInferenceTime / processedImages else 0
                val minTime = inferenceTimes.minOrNull() ?: 0
                val maxTime = inferenceTimes.maxOrNull() ?: 0

                withContext(Dispatchers.Main) {
                    showFinalResults(
                        processedImages = processedImages,
                        totalImages = totalImages,
                        totalTime = totalTime,
                        totalInferenceTime = totalInferenceTime,
                        avgInferenceTime = avgTime,
                        minInferenceTime = minTime,
                        maxInferenceTime = maxTime
                    )
                }

            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    showStatus("Error: ${e.message}")
                    Toast.makeText(this@MainActivity, "Benchmark failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            } finally {
                withContext(Dispatchers.Main) {
                    isBenchmarkRunning = false
                    binding.buttonStartBenchmark.isEnabled = true
                    binding.buttonStopBenchmark.isEnabled = false
                }
            }
        }
    }

    private fun updateProgress(
        progress: Int,
        processed: Int,
        total: Int,
        avgInferenceTime: Long,
        totalElapsedTime: Long,
        lastInferenceTime: Long
    ) {
        binding.progressBar.progress = progress
        binding.textViewProgress.text = "$processed / $total images"
        binding.textViewAvgInferenceTime.text = "${avgInferenceTime} ms"
        binding.textViewLastInferenceTime.text = "${lastInferenceTime} ms"
        binding.textViewTotalElapsedTime.text = formatTime(totalElapsedTime)

        // Calculate estimated remaining time
        if (processed > 0) {
            val remainingImages = total - processed
            val estimatedRemaining = (avgInferenceTime * remainingImages)
            binding.textViewEstimatedRemaining.text = formatTime(estimatedRemaining)
        }

        showStatus("Processing: $processed / $total")
    }

    private fun showFinalResults(
        processedImages: Int,
        totalImages: Int,
        totalTime: Long,
        totalInferenceTime: Long,
        avgInferenceTime: Long,
        minInferenceTime: Long,
        maxInferenceTime: Long
    ) {
        binding.progressBar.progress = 100
        binding.textViewProgress.text = "$processedImages / $totalImages images (Complete)"
        binding.textViewAvgInferenceTime.text = "${avgInferenceTime} ms"
        binding.textViewTotalElapsedTime.text = formatTime(totalTime)
        binding.textViewEstimatedRemaining.text = "Done"

        // Update results section
        binding.textViewResultTotalTime.text = formatTime(totalTime)
        binding.textViewResultInferenceTime.text = formatTime(totalInferenceTime)
        binding.textViewResultAvgTime.text = "${avgInferenceTime} ms"
        binding.textViewResultMinTime.text = "${minInferenceTime} ms"
        binding.textViewResultMaxTime.text = "${maxInferenceTime} ms"
        binding.textViewResultFPS.text = String.format("%.2f", 1000.0 / avgInferenceTime)

        val status = """
            Benchmark Complete!
            Processed: $processedImages images
            Total Time: ${formatTime(totalTime)}
            Avg Inference: ${avgInferenceTime} ms/image
            FPS: ${String.format("%.2f", 1000.0 / avgInferenceTime)}
        """.trimIndent()

        showStatus(status)
        Toast.makeText(this, "Benchmark completed!", Toast.LENGTH_SHORT).show()
    }

    private fun stopBenchmark() {
        isBenchmarkRunning = false
        showStatus("Stopping benchmark...")
        binding.buttonStopBenchmark.isEnabled = false
    }

    private fun showStatus(message: String) {
        binding.textViewStatus.text = message
    }

    private fun formatTime(millis: Long): String {
        return when {
            millis < 1000 -> "${millis} ms"
            millis < 60000 -> String.format("%.2f sec", millis / 1000.0)
            else -> String.format("%.2f min", millis / 60000.0)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        modelBenchmark?.close()
    }
}
