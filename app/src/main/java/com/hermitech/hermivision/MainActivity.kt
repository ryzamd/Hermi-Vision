package com.hermitech.hermivision

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.lifecycle.viewmodel.compose.viewModel
import com.hermitech.hermivision.ui.VideoAnalysisRoute
import com.hermitech.hermivision.ui.CameraViewModel
import com.hermitech.hermivision.ui.theme.HermivisionTheme

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

@Composable
private fun HermivisionApp() {
    val viewModel: CameraViewModel = viewModel()
    val videoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        viewModel.onVideoSelected(uri)
    }

    VideoAnalysisRoute(
        viewModel = viewModel,
        onPickVideo = { videoPicker.launch("video/*") }
    )
}
