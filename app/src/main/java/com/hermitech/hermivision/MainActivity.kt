package com.hermitech.hermivision

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.hermitech.hermivision.ui.picker.VideoPickerScreen
import com.hermitech.hermivision.ui.processing.ProcessingScreen
import com.hermitech.hermivision.ui.processing.ProcessingViewModel
import com.hermitech.hermivision.ui.results.ResultsScreen
import com.hermitech.hermivision.ui.theme.HermivisionTheme
import com.hermitech.hermivision.worker.VideoProcessingWorker

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            HermivisionTheme {
                HermivisionApp()
            }
        }
    }
}

private object Routes {
    const val PICKER = "picker"
    const val PROCESSING = "processing"
    const val RESULTS = "results"
}

@Composable
fun HermivisionApp() {
    val navController = rememberNavController()
    val processingViewModel: ProcessingViewModel = viewModel()

    NavHost(navController = navController, startDestination = Routes.PICKER) {

        composable(Routes.PICKER) {
            VideoPickerScreen(
                onVideoSelected = { uri: Uri ->
                    processingViewModel.processVideo(uri)
                    navController.navigate(Routes.PROCESSING) {
                        launchSingleTop = true
                    }
                }
            )
        }

        composable(Routes.PROCESSING) {
            val uiState by processingViewModel.uiState.collectAsState()

            ProcessingScreen(
                uiState = uiState,
                onCancelClick = {
                    processingViewModel.cancelProcessing()
                    navController.popBackStack(Routes.PICKER, inclusive = false)
                },
                onViewResultsClick = {
                    navController.navigate(Routes.RESULTS) {
                        popUpTo(Routes.PICKER) { inclusive = false }
                        launchSingleTop = true
                    }
                },
                onRetryClick = {
                    navController.popBackStack(Routes.PICKER, inclusive = false)
                }
            )
        }

        composable(Routes.RESULTS) {
            val holder = VideoProcessingWorker.ResultHolder

            ResultsScreen(
                ballFrames = holder.ballFrames,
                bounceFrameIds = holder.bounceFrameIds,
                bounceCourtPoints = holder.bounceCourtPoints,
                trajectoryCourtPoints = holder.trajectoryCourtPoints,
                totalDurationMs = holder.durationMs,
                onBackClick = {
                    VideoProcessingWorker.ResultHolder.clear()
                    navController.popBackStack(Routes.PICKER, inclusive = false)
                }
            )
        }
    }
}