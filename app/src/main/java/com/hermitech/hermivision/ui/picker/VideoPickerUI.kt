package com.hermitech.hermivision.ui.picker

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier

@Composable
fun VideoPickerScreen(onVideoSelected: (Uri) -> Unit) {
    val launcher = rememberLauncherForActivityResult(contract = ActivityResultContracts.GetContent())
    { uri: Uri? ->
        uri?.let { onVideoSelected(it) }
    }

    Scaffold { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues),
            contentAlignment = Alignment.Center
        ) {
            Button(onClick = { launcher.launch("video/*") }) {
                Text("Select Tennis Video")
            }
        }
    }
}