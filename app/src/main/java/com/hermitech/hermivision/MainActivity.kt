package com.hermitech.hermivision

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.hermitech.hermivision.shared.AppEnvironment
import com.hermitech.hermivision.ui.optimizing.OptimizingScreen
import com.hermitech.hermivision.ui.optimizing.OptimizingViewModel
import com.hermitech.hermivision.ui.picker.VideoPickerScreen
import com.hermitech.hermivision.ui.processing.ProcessingScreen
import com.hermitech.hermivision.ui.processing.ProcessingViewModel
import com.hermitech.hermivision.ui.results.ResultsScreen
import com.hermitech.hermivision.ui.theme.HermivisionTheme
import com.hermitech.hermivision.worker.VideoProcessingWorker

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = AppEnvironment.displayName()
        enableEdgeToEdge()
        setContent {
            HermivisionTheme {
                HermivisionApp()
            }
        }
    }
}

private object Routes {
    const val OPTIMIZING = "optimizing"
    const val PICKER = "picker"
    const val PROCESSING = "processing"
    const val RESULTS = "results"
}

@Composable
fun HermivisionApp() {
    val navController = rememberNavController()
    val processingViewModel: ProcessingViewModel = viewModel()
    val optimizingViewModel: OptimizingViewModel = viewModel()

    // Determine start destination based on whether config exists in DB
    var startDestination by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(Unit) {
        startDestination = if (optimizingViewModel.hasExistingConfig()) {
            Routes.PICKER
        } else {
            Routes.OPTIMIZING
        }
    }

    // Wait until we know the start destination
    val destination = startDestination ?: return

    NavHost(navController = navController, startDestination = destination) {

        composable(Routes.OPTIMIZING) {
            val uiState by optimizingViewModel.uiState.collectAsState()

            // Start benchmark when screen appears
            LaunchedEffect(Unit) {
                optimizingViewModel.startBenchmark()
            }

            OptimizingScreen(
                uiState = uiState,
                onComplete = {
                    navController.navigate(Routes.PICKER) {
                        popUpTo(Routes.OPTIMIZING) { inclusive = true }
                        launchSingleTop = true
                    }
                }
            )
        }

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
                totalFrames = holder.totalFrames,
                visibleFrames = holder.visibleFrames,
                inferenceTimeMs = holder.inferenceTimeMs,
                totalDurationMs = holder.durationMs,
                onBackClick = {
                    VideoProcessingWorker.ResultHolder.clear()
                    navController.popBackStack(Routes.PICKER, inclusive = false)
                }
            )
        }
    }
}
