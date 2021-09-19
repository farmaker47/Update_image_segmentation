/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imagesegmentation.camera

import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.media.Image
import android.os.Bundle
import androidx.fragment.app.Fragment
import android.util.Log
import android.util.Size
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.*
import org.tensorflow.lite.examples.imagesegmentation.databinding.TfeCameraFragmentBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.io.*

class CameraFragment : Fragment() {

    /**
     * Interface to interact with the hosting activity
     */
    interface OnCaptureFinished {
        fun onCaptureFinished(file: File)
    }

    private var preview: Preview? = null
    private var imageCapture: ImageCapture? = null
    private var camera: Camera? = null
    private var bitmap: Bitmap? = null
    private var lensFacing: Int = 0
    private lateinit var filePath: File

    internal lateinit var callback: OnCaptureFinished

    private lateinit var binding: TfeCameraFragmentBinding

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        binding = TfeCameraFragmentBinding.inflate(inflater)
        return binding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Start the camera
        startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())

        val screenAspectRatio = 1.0 / 1.0
        Log.d(TAG, "Preview aspect ratio: $screenAspectRatio")

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            // set up Preview
            preview = Preview.Builder().build()

            // set up Capture
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .setTargetResolution(Size(512, 512)) // Target resolution to 512x512
                .build()

            // Select front camera as default for selfie
            val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing)
                .build() // Change camera to front facing

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture
                )
                preview?.setSurfaceProvider(binding.previewView.surfaceProvider)
            } catch (exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(requireContext()))
    }

    @SuppressLint("UnsafeOptInUsageError")
    fun takePicture() {

        lifecycleScope.launch(Dispatchers.Default) {

            // Get a stable reference of the modifiable image capture use case
            val imageCapture = imageCapture ?: return@launch

            // Create timestamped output file to hold the image
            val photoFile = createFile(requireActivity(), "jpg")

            // Create output options object which contains file + metadata
            val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

            imageCapture.takePicture(
                outputOptions,
                ContextCompat.getMainExecutor(requireActivity()),
                object : ImageCapture.OnImageSavedCallback {
                    override fun onError(exc: ImageCaptureException) {
                        Log.e(TAG, "Photo capture failed: ${exc.message}", exc)
                    }

                    override fun onImageSaved(output: ImageCapture.OutputFileResults) {

                        // Get rotation degree
                        val degrees: Int = rotationDegrees(photoFile)

                        // Create a bitmap from the .jpg image
                        bitmap = BitmapFactory.decodeFile(photoFile.absolutePath)

                        // Rotate image if needed
                        if (degrees != 0) {
                            bitmap = rotateBitmap(bitmap!!, degrees)
                        }

                        // Save bitmap image
                        filePath = saveBitmap(bitmap, photoFile)
                        Log.v(TAG, filePath.toString())

                        // Trigger the callback
                        callback.onCaptureFinished(filePath)

                    }
                })
        }
    }

    fun saveBitmap(bitmap: Bitmap?, file: File): File {
        try {
            val stream: OutputStream = FileOutputStream(file)
            bitmap?.compress(Bitmap.CompressFormat.JPEG, 100, stream)
            stream.flush()
            stream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }

        return file
    }

    private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        val rotationMatrix = Matrix()
        rotationMatrix.postRotate(rotationDegrees.toFloat())
        val rotatedBitmap =
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, rotationMatrix, true)
        bitmap.recycle()
        return rotatedBitmap
    }

    /**
     * Get rotation degree from image exif
     */
    private fun rotationDegrees(file: File): Int {
        val ei = ExifInterface(file.absolutePath)
        // Return rotation degree based on orientation from exif
        return when (ei.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        )) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90
            ExifInterface.ORIENTATION_ROTATE_180 -> 180
            ExifInterface.ORIENTATION_ROTATE_270 -> 270
            else -> 0
        }
    }

    /**
    Keeping a reference to the activity to make communication between it and this fragment
    easier.
     */
    override fun onAttach(context: Context) {
        super.onAttach(context)
        callback = context as OnCaptureFinished
    }

    fun setFacingCamera(lensFacingImport: Int) {
        lensFacing = lensFacingImport
    }

    companion object {
        private val TAG = CameraFragment::class.java.simpleName

        /**
         * Create a [File] named a using formatted timestamp with the current date and time.
         *
         * @return [File] created.
         */
        private fun createFile(context: Context, extension: String): File {
            val sdf = SimpleDateFormat("yyyy_MM_dd_HH_mm_ss_SSS", Locale.US)
            /**
             * Returns the absolute path to the directory on the filesystem where files created with openFileOutput are stored.
             * The returned path may change over time if the calling app is moved to an adopted storage device, so only relative paths should be persisted.
             * No additional permissions are required for the calling app to read or write files under the returned path.
             * Returns:
             * The path of the directory holding application files.
             */
            return File(context.filesDir, "IMG_${sdf.format(Date())}.$extension")
        }

        @JvmStatic
        fun newInstance(): CameraFragment =
            CameraFragment()
    }
}
