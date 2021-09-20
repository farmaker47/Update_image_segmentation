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
import android.content.Context
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.*
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import org.tensorflow.lite.examples.imagesegmentation.databinding.TfeCameraFragmentBinding
import java.io.*
import java.util.concurrent.ExecutionException

class CameraFragment : Fragment() {

    /**
     * Interface to interact with the hosting activity
     */
    interface OnCaptureFinished {
        fun onCaptureFinished(image: Image, imageRotation: Int)
    }

    private var imageAnalysis: ImageAnalysis? = null
    private var lensFacing: Int = 0
    private lateinit var filePath: File
    private val DESIRED_PREVIEW_SIZE = Size(640, 480)
    private var isProcessingFrame = true
    var copyImage: ImageProxy? = null
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

    @SuppressLint("UnsafeOptInUsageError")
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireActivity())

        cameraProviderFuture.addListener({
            // Camera provider is now guaranteed to be available
            try {
                val cameraProvider = cameraProviderFuture.get()

                // Set up the view finder use case to display camera preview
                val preview =
                    Preview.Builder().build()

                // Choose the camera by requiring a lens facing
                val cameraSelector = CameraSelector.Builder()
                    .requireLensFacing(lensFacing)
                    .build()

                // Image Analysis
                imageAnalysis = ImageAnalysis.Builder()
                    .setTargetResolution(DESIRED_PREVIEW_SIZE)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                imageAnalysis?.setAnalyzer(
                    ContextCompat.getMainExecutor(requireActivity()),
                    { image ->
                        copyImage = image
                        // Define rotation Degrees of the imageProxy
                        val rotationDegrees = image.imageInfo.rotationDegrees
                        Log.v("Image_degrees", rotationDegrees.toString())

                        // Trigger the callback
                        if (!isProcessingFrame) {
                            //image.image?.let { copyImage = image }
                            copyImage?.image?.let {
                                callback.onCaptureFinished(
                                    it,
                                    rotationDegrees
                                )
                            }
                            isProcessingFrame = true

                        }
                    })

                // Connect the preview use case to the previewView
                preview.setSurfaceProvider(
                    binding.previewView.surfaceProvider
                )

                // Attach use cases to the camera with the same lifecycle owner
                if (cameraProvider != null) {
                    val camera = cameraProvider.bindToLifecycle(
                        this,
                        cameraSelector,
                        imageAnalysis,
                        preview
                    )
                }
            } catch (e: ExecutionException) {
                e.printStackTrace()
            } catch (e: InterruptedException) {
                e.printStackTrace()
            }
        }, ContextCompat.getMainExecutor(requireActivity()))
    }

    fun processingImage() {
        isProcessingFrame = false
        copyImage?.close()
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
        @JvmStatic
        fun newInstance(): CameraFragment =
            CameraFragment()
    }
}
