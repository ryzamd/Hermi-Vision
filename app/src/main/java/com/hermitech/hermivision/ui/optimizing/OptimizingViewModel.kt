package com.hermitech.hermivision.ui.optimizing

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.hermitech.hermivision.data.AppDatabase
import com.hermitech.hermivision.data.RoomDeviceConfigRepository
import com.hermitech.hermivision.domain.inference.DeviceProfiler
import com.hermitech.hermivision.shared.domain.usecase.BuildBenchmarkResultsTextUseCase
import com.hermitech.hermivision.shared.presentation.optimizing.OptimizingStateHolder
import com.hermitech.hermivision.shared.presentation.optimizing.OptimizingUiState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class OptimizingViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "OptimizingViewModel"
    }

    private val configRepository = RoomDeviceConfigRepository(AppDatabase.getInstance(application))
    private val buildBenchmarkResultsText = BuildBenchmarkResultsTextUseCase()
    private val stateHolder = OptimizingStateHolder()
    private val _uiState = MutableStateFlow(OptimizingUiState())
    val uiState: StateFlow<OptimizingUiState> = _uiState.asStateFlow()

    /**
     * Check if device config already exists in DB.
     * Returns true if optimizing can be skipped.
     */
    suspend fun hasExistingConfig(): Boolean = withContext(Dispatchers.IO) {
        configRepository.getConfig() != null
    }

    /**
     * Run the device benchmark and save results to Room DB.
     * Should only be called once (first launch).
     */
    fun startBenchmark() {
        stateHolder.startBenchmark()
        _uiState.value = stateHolder.state

        viewModelScope.launch {
            try {
                val profiler = DeviceProfiler(getApplication())

                val config = withContext(Dispatchers.IO) {
                    profiler.runBenchmark { progress ->
                        stateHolder.onBenchmarkProgress(progress)
                        _uiState.value = stateHolder.state
                    }
                }

                // Save to Room DB
                withContext(Dispatchers.IO) {
                    configRepository.saveConfig(config)
                }

                Log.i(TAG, "Benchmark complete, saved to DB: ${config.deviceSummary}")

                stateHolder.onBenchmarkCompleted(
                    deviceSummary = config.deviceSummary,
                    benchmarkResults = buildBenchmarkResultsText(config),
                )
                _uiState.value = stateHolder.state

            } catch (e: Exception) {
                Log.e(TAG, "Benchmark failed", e)
                stateHolder.onBenchmarkFailed(e.message)
                _uiState.value = stateHolder.state
            }
        }
    }

}
