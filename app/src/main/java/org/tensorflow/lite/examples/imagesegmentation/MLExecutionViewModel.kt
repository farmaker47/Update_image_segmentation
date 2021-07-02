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

package org.tensorflow.lite.examples.imagesegmentation

import android.app.Application
import android.content.Context
import android.util.Log
import android.widget.TextView
import androidx.lifecycle.*
import kotlinx.coroutines.*
import java.io.File
import org.tensorflow.lite.examples.imagesegmentation.tflite.ImageSegmentationModelExecutor
import org.tensorflow.lite.examples.imagesegmentation.tflite.ModelExecutionResult
import org.tensorflow.lite.examples.imagesegmentation.utils.ImageUtils

private const val TAG = "MLExecutionViewModel"

class MLExecutionViewModel(application: Application) : AndroidViewModel(application) {

    private var imageSegmentationModel: ImageSegmentationModelExecutor? = null
    private var useGPU = false

    private val _resultingBitmap = MutableLiveData<ModelExecutionResult>()
    val resultingBitmap: LiveData<ModelExecutionResult>
        get() = _resultingBitmap

    private val _errorString = MutableLiveData<String>()
    val errorString: LiveData<String>
        get() = _errorString

    init {
        createModelExecutor(useGPU)
    }

    fun createModelExecutor(useGPU: Boolean) {
        if (imageSegmentationModel != null) {
            imageSegmentationModel!!.close()
            imageSegmentationModel = null
        }
        try {
            imageSegmentationModel = ImageSegmentationModelExecutor(getApplication(), useGPU)
        } catch (e: Exception) {
            Log.e(TAG, "Fail to create ImageSegmentationModelExecutor: ${e.message}")
            //_errorString.value = e.message
        }
    }

    // the execution of the model has to be on the same thread where the interpreter
    // was created
    fun onApplyModel(
        filePath: String
    ) {
        viewModelScope.launch(Dispatchers.Default) {
            val contentImage =
                ImageUtils.decodeBitmap(
                    File(filePath)
                )
            try {
                val result = imageSegmentationModel?.execute(contentImage)
                _resultingBitmap.postValue(result)
            } catch (e: Exception) {
                Log.e(TAG, "Fail to execute ImageSegmentationModelExecutor: ${e.message}")
                _resultingBitmap.postValue(null)
            }
        }
    }
}
