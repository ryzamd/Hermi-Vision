package com.hermitech.hermivision.ui

import android.app.Application
import android.net.Uri
import androidx.camera.core.ImageAnalysis
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.hermitech.hermivision.camera.FrameAnalyzer
import com.hermitech.hermivision.domain.BallState
import com.hermitech.hermivision.domain.BounceEvent
import com.hermitech.hermivision.R
import com.hermitech.hermivision.inference.TennisBroadcastDetector
import com.hermitech.hermivision.inference.Postprocessor
import com.hermitech.hermivision.inference.Preprocessor
import com.hermitech.hermivision.persistence.BounceSessionStore
import com.hermitech.hermivision.tracking.BallTracker
import com.hermitech.hermivision.tracking.BounceDetector
import com.hermitech.hermivision.video.VideoBounceAnalyzer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class CameraUiState(
    val ballState: BallState = BallState(),
    val detectorLabel: String = "",
    val bounceEvents: List<BounceEvent> = emptyList(),
    val bounceSessionPath: String = "",
    val selectedVideoUri: Uri? = null,
    val isAnalyzingVideo: Boolean = false,
    val analysisProgress: Float = 0f,
    val analysisStatus: String = ""
)

class CameraViewModel(application: Application) : AndroidViewModel(application) {
    private val tracker = BallTracker()
    private val preprocessor = Preprocessor()
    private val detector = TennisBroadcastDetector(application.applicationContext)
    private val postprocessor = Postprocessor()
    private val bounceDetector = BounceDetector()
    private val bounceSessionStore = BounceSessionStore(application.applicationContext)
    private val videoAnalyzer = VideoBounceAnalyzer(
        context = application.applicationContext,
        preprocessor = preprocessor,
        detector = detector,
        postprocessor = postprocessor,
        tracker = tracker,
        bounceDetector = bounceDetector
    )
    private var analysisJob: Job? = null

    private val _uiState = MutableStateFlow(CameraUiState())
    val uiState: StateFlow<CameraUiState> = _uiState.asStateFlow()

    private val analyzer = FrameAnalyzer(
        preprocessor = preprocessor,
        detector = detector,
        postprocessor = postprocessor,
        tracker = tracker
    ) { ballState ->
        val bounceEvent = bounceDetector.update(ballState)
        if (bounceEvent != null) {
            bounceSessionStore.append(bounceEvent)
        }

        _uiState.update { currentState ->
            currentState.copy(
                ballState = ballState,
                bounceEvents = if (bounceEvent == null) {
                    currentState.bounceEvents
                } else {
                    (currentState.bounceEvents + bounceEvent).takeLast(MAX_BOUNCE_EVENTS)
                }
            )
        }
    }

    val frameAnalyzer: ImageAnalysis.Analyzer = analyzer

    init {
        _uiState.update { currentState ->
            currentState.copy(
                detectorLabel = getApplication<Application>().getString(R.string.detector_label),
                bounceSessionPath = bounceSessionStore.sessionPath(),
                analysisStatus = getApplication<Application>().getString(R.string.analysis_idle)
            )
        }
    }

    fun onVideoSelected(videoUri: Uri?) {
        viewModelScope.launch {
            analysisJob?.cancelAndJoin()
            bounceDetector.reset()
            tracker.reset()
            _uiState.update { currentState ->
                currentState.copy(
                    selectedVideoUri = videoUri,
                    bounceEvents = emptyList(),
                    ballState = BallState(),
                    analysisProgress = 0f,
                    isAnalyzingVideo = false,
                    analysisStatus = if (videoUri == null) {
                        getApplication<Application>().getString(R.string.analysis_idle)
                    } else {
                        getApplication<Application>().getString(R.string.analysis_ready)
                    }
                )
            }

            if (videoUri != null) {
                analyzeSelectedVideo()
            }
        }
    }

    fun analyzeSelectedVideo() {
        val videoUri = _uiState.value.selectedVideoUri ?: return
        analysisJob?.cancel()
        bounceDetector.reset()
        tracker.reset()

        _uiState.update { currentState ->
            currentState.copy(
                isAnalyzingVideo = true,
                analysisProgress = 0f,
                bounceEvents = emptyList(),
                ballState = BallState(),
                analysisStatus = getApplication<Application>().getString(R.string.analysis_running)
            )
        }

        analysisJob = viewModelScope.launch(Dispatchers.Default) {
            runCatching {
                videoAnalyzer.analyze(videoUri) { processedFrames, totalFramesEstimate, latestBallState ->
                    _uiState.update { currentState ->
                        currentState.copy(
                            ballState = latestBallState,
                            analysisProgress = (processedFrames.toFloat() / totalFramesEstimate.toFloat()).coerceIn(0f, 1f)
                        )
                    }
                }
            }.onSuccess { result ->
                result.bounceEvents.forEach(bounceSessionStore::append)
                _uiState.update { currentState ->
                    currentState.copy(
                        ballState = result.lastBallState,
                        bounceEvents = result.bounceEvents.takeLast(MAX_BOUNCE_EVENTS),
                        isAnalyzingVideo = false,
                        analysisProgress = 1f,
                        analysisStatus = getApplication<Application>().getString(R.string.analysis_complete)
                    )
                }
            }.onFailure {
                _uiState.update { currentState ->
                    currentState.copy(
                        isAnalyzingVideo = false,
                        analysisStatus = getApplication<Application>().getString(R.string.analysis_failed)
                    )
                }
            }
        }
    }

    fun clearBounceLog() {
        bounceDetector.reset()
        tracker.reset()
        _uiState.update { currentState ->
            currentState.copy(
                bounceEvents = emptyList(),
                ballState = BallState(),
                analysisProgress = 0f,
                analysisStatus = getApplication<Application>().getString(R.string.analysis_idle)
            )
        }
    }

    override fun onCleared() {
        super.onCleared()
        analysisJob?.cancel()
        analyzer.close()
    }

    private companion object {
        const val MAX_BOUNCE_EVENTS = 24
    }
}
