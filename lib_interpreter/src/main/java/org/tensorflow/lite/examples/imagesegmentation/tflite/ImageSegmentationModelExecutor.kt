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

package org.tensorflow.lite.examples.imagesegmentation.tflite

import android.R.attr
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.media.Image
import android.os.SystemClock
import androidx.core.graphics.ColorUtils
import android.util.Log
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.random.Random
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.imagesegmentation.utils.ImageUtils
import org.tensorflow.lite.gpu.GpuDelegate
import android.R.attr.bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp

import org.tensorflow.lite.support.image.TensorImage

import org.tensorflow.lite.support.image.ops.ResizeOp

import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.Rot90Op


/**
 * Class responsible to run the Image Segmentation model.
 * more information about the DeepLab model being used can
 * be found here:
 * https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html
 * https://www.tensorflow.org/lite/models/segmentation/overview
 * https://github.com/tensorflow/models/tree/master/research/deeplab
 *
 * Label names: 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
 * 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
 * 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
 */
class ImageSegmentationModelExecutor(
    context: Context,
    private var useGPU: Boolean = false
) {
    private var gpuDelegate: GpuDelegate? = null

    private val segmentationMasks: ByteBuffer
    private val interpreter: Interpreter

    private var fullTimeExecutionTime = 0L
    private var preprocessTime = 0L
    private var imageSegmentationTime = 0L
    private var maskFlatteningTime = 0L

    private var numberThreads = 4

    private val yuvBytes = arrayOfNulls<ByteArray>(3)
    private var rgbBytes: IntArray? = null
    private var yRowStride = 0
    val kMaxChannelValue = 262143

    init {

        interpreter = getInterpreter(context, imageSegmentationModel, useGPU)
        segmentationMasks = ByteBuffer.allocateDirect(1 * imageSize * imageSize * NUM_CLASSES * 4)
        segmentationMasks.order(ByteOrder.nativeOrder())
    }

    fun execute(inputImage: Image, imageRotation: Int): ModelExecutionResult {
        try {
            fullTimeExecutionTime = SystemClock.uptimeMillis()

            // Convert media.Image to Bitmap
            val originalBitmap = imageToRGB(inputImage, inputImage.width, inputImage.height)
            // Rotate image if needed
            var bitmap = Bitmap.createBitmap(
                originalBitmap.width,
                originalBitmap.height,
                Bitmap.Config.ARGB_8888
            )
            if (imageRotation != 0) {
                bitmap = rotateBitmap(originalBitmap, imageRotation)
            }

            preprocessTime = SystemClock.uptimeMillis()
            // Create scaled bitmap
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true)
            // Create ByteBuffer to use with Interpreter
            val contentArray =
                ImageUtils.bitmapToByteBuffer(
                    scaledBitmap,
                    imageSize,
                    imageSize,
                    IMAGE_MEAN,
                    IMAGE_STD
                )

            /*val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(imageSize, imageSize, ResizeOp.ResizeMethod.BILINEAR))
                .add(Rot90Op(1))
                .add(NormalizeOp(127.5f, 127.5f))
                .build()

            var tensorImage = TensorImage(DataType.UINT8)

            tensorImage.load(originalBitmap)
            tensorImage = imageProcessor.process(tensorImage)

            // Create scaled bitmap
            var bitmap = Bitmap.createBitmap(
                originalBitmap.width,
                originalBitmap.height,
                Bitmap.Config.ARGB_8888
            )
            if (imageRotation != 0) {
                bitmap = rotateBitmap(originalBitmap, imageRotation)
            }
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true)*/

            preprocessTime = SystemClock.uptimeMillis() - preprocessTime
            imageSegmentationTime = SystemClock.uptimeMillis()
            interpreter.run(contentArray, segmentationMasks)
            imageSegmentationTime = SystemClock.uptimeMillis() - imageSegmentationTime
            Log.d(TAG, "Time to run the model $imageSegmentationTime")

            maskFlatteningTime = SystemClock.uptimeMillis()
            val (maskImageApplied, maskOnly, itemsFound) =
                convertBytebufferMaskToBitmap(
                    segmentationMasks, imageSize, imageSize, scaledBitmap,
                    segmentColors
                )
            maskFlatteningTime = SystemClock.uptimeMillis() - maskFlatteningTime
            Log.d(TAG, "Time to flatten the mask result $maskFlatteningTime")

            fullTimeExecutionTime = SystemClock.uptimeMillis() - fullTimeExecutionTime
            Log.d(TAG, "Total time execution $fullTimeExecutionTime")

            return ModelExecutionResult(
                maskImageApplied,
                scaledBitmap,
                maskOnly,
                formatExecutionLog(),
                itemsFound
            )
        } catch (e: Exception) {
            val exceptionLog = "something went wrong: ${e.message}"
            Log.d(TAG, exceptionLog)

            val emptyBitmap =
                ImageUtils.createEmptyBitmap(
                    imageSize,
                    imageSize
                )
            return ModelExecutionResult(
                emptyBitmap,
                emptyBitmap,
                emptyBitmap,
                exceptionLog,
                HashMap<String, Int>()
            )
        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(context: Context, modelFile: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelFile)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val retFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        fileDescriptor.close()
        return retFile
    }

    @Throws(IOException::class)
    private fun getInterpreter(
        context: Context,
        modelName: String,
        useGpu: Boolean = false
    ): Interpreter {
        val tfliteOptions = Interpreter.Options()
        tfliteOptions.setNumThreads(numberThreads)

        gpuDelegate = null
        if (useGpu) {
            gpuDelegate = GpuDelegate()
            tfliteOptions.addDelegate(gpuDelegate)
        }

        return Interpreter(loadModelFile(context, modelName), tfliteOptions)
    }

    private fun formatExecutionLog(): String {
        val sb = StringBuilder()
        sb.append("Input Image Size: $imageSize x $imageSize\n")
        sb.append("GPU enabled: $useGPU\n")
        sb.append("Number of threads: $numberThreads\n")
        sb.append("Pre-process execution time: $preprocessTime ms\n")
        sb.append("Model execution time: $imageSegmentationTime ms\n")
        sb.append("Mask flatten time: $maskFlatteningTime ms\n")
        sb.append("Full execution time: $fullTimeExecutionTime ms\n")
        return sb.toString()
    }

    fun close() {
        interpreter.close()
        if (gpuDelegate != null) {
            gpuDelegate!!.close()
        }
    }

    private fun convertBytebufferMaskToBitmap(
        inputBuffer: ByteBuffer,
        imageWidth: Int,
        imageHeight: Int,
        backgroundImage: Bitmap,
        colors: IntArray
    ): Triple<Bitmap, Bitmap, Map<String, Int>> {
        val conf = Bitmap.Config.ARGB_8888
        val maskBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf)
        val resultBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf)
        val scaledBackgroundImage =
            ImageUtils.scaleBitmapAndKeepRatio(
                backgroundImage,
                imageWidth,
                imageHeight
            )
        val mSegmentBits = Array(imageWidth) { IntArray(imageHeight) }
        val itemsFound = HashMap<String, Int>()
        inputBuffer.rewind()

        for (y in 0 until imageHeight) {
            for (x in 0 until imageWidth) {
                var maxVal = 0f
                mSegmentBits[x][y] = 0

                for (c in 0 until NUM_CLASSES) {
                    val value = inputBuffer
                        .getFloat((y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + c) * 4)
                    if (c == 0 || value > maxVal) {
                        maxVal = value
                        mSegmentBits[x][y] = c
                    }
                }
                val label = labelsArrays[mSegmentBits[x][y]]
                val color = colors[mSegmentBits[x][y]]
                itemsFound.put(label, color)
                val newPixelColor = ColorUtils.compositeColors(
                    colors[mSegmentBits[x][y]],
                    scaledBackgroundImage.getPixel(x, y)
                )
                resultBitmap.setPixel(x, y, newPixelColor)
                maskBitmap.setPixel(x, y, colors[mSegmentBits[x][y]])
            }
        }

        return Triple(resultBitmap, maskBitmap, itemsFound)
    }

    private fun imageToRGB(image: Image?, width: Int, height: Int): Bitmap {
        if (rgbBytes == null) {
            rgbBytes = IntArray(width * height)
        }
        val rgbFrameBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        try {
            if (image == null) {
                return rgbFrameBitmap
            }
            Log.e("Degrees_length", rgbBytes?.size.toString())
            val planes = image.planes
            fillBytesCameraX(planes, yuvBytes)
            yRowStride = planes[0].rowStride
            val uvRowStride = planes[1].rowStride
            val uvPixelStride = planes[1].pixelStride
            convertYUV420ToARGB8888(
                yuvBytes[0] ?: byteArrayOf(),
                yuvBytes[1] ?: byteArrayOf(),
                yuvBytes[2] ?: byteArrayOf(),
                width,
                height,
                yRowStride,
                uvRowStride,
                uvPixelStride,
                rgbBytes!!
            )
            rgbFrameBitmap.setPixels(rgbBytes, 0, width, 0, 0, width, height)
        } catch (e: Exception) {
            Log.e(e.toString(), "Exception!")
        }
        return rgbFrameBitmap
    }

    private fun fillBytesCameraX(planes: Array<Image.Plane>, yuvBytes: Array<ByteArray?>) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (i in planes.indices) {
            val buffer = planes[i].buffer
            if (yuvBytes[i] == null) {
                yuvBytes[i] = ByteArray(buffer.capacity())
            }
            buffer[yuvBytes[i]]
        }
    }

    fun convertYUV420ToARGB8888(
        yData: ByteArray,
        uData: ByteArray,
        vData: ByteArray,
        width: Int,
        height: Int,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        out: IntArray
    ) {
        var yp = 0
        for (j in 0 until height) {
            val pY = yRowStride * j
            val pUV = uvRowStride * (j shr 1)
            for (i in 0 until width) {
                val uv_offset = pUV + (i shr 1) * uvPixelStride
                out[yp++] = YUV2RGB(
                    0xff and yData[pY + i].toInt(), 0xff and uData[uv_offset]
                        .toInt(), 0xff and vData[uv_offset].toInt()
                )
            }
        }
    }

    private fun YUV2RGB(y: Int, u: Int, v: Int): Int {
        // Adjust and check YUV values
        var y = y
        var u = u
        var v = v
        y = Math.max(y - 16, 0)
        u -= 128
        v -= 128

        // This is the floating point equivalent. We do the conversion in integer
        // because some Android devices do not have floating point in hardware.
        // nR = (int)(1.164 * nY + 2.018 * nU);
        // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
        // nB = (int)(1.164 * nY + 1.596 * nV);
        val y1192 = 1192 * y
        var r = y1192 + 1634 * v
        var g = y1192 - 833 * v - 400 * u
        var b = y1192 + 2066 * u

        // Clipping RGB values to be inside boundaries [ 0 , kMaxChannelValue ]
        r = if (r > kMaxChannelValue) kMaxChannelValue else Math.max(r, 0)
        g = if (g > kMaxChannelValue) kMaxChannelValue else Math.max(g, 0)
        b = if (b > kMaxChannelValue) kMaxChannelValue else Math.max(b, 0)
        return -0x1000000 or (r shl 6 and 0xff0000) or (g shr 2 and 0xff00) or (b shr 10 and 0xff)
    }

    private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        val rotationMatrix = Matrix()
        rotationMatrix.postRotate(rotationDegrees.toFloat())
        val rotatedBitmap =
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, rotationMatrix, true)
        bitmap.recycle()
        return rotatedBitmap
    }

    companion object {

        const val TAG = "SegmentationInterpreter"
        private const val imageSegmentationModel = "deeplabv3_257_mv_gpu.tflite"
        private const val imageSize = 257
        const val NUM_CLASSES = 21
        private const val IMAGE_MEAN = 127.5f
        private const val IMAGE_STD = 127.5f

        val segmentColors = IntArray(NUM_CLASSES)
        val labelsArrays = arrayOf(
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
            "person", "potted plant", "sheep", "sofa", "train", "tv"
        )

        init {

            val random = Random(System.currentTimeMillis())
            segmentColors[0] = Color.TRANSPARENT
            for (i in 1 until NUM_CLASSES) {
                segmentColors[i] = Color.argb(
                    (128),
                    getRandomRGBInt(
                        random
                    ),
                    getRandomRGBInt(
                        random
                    ),
                    getRandomRGBInt(
                        random
                    )
                )
            }
        }

        private fun getRandomRGBInt(random: Random) = (255 * random.nextFloat()).toInt()
    }
}
