package com.hermitech.hermivision.ui.live

import android.Manifest
import android.app.Activity
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.util.Log
import android.util.Size
import android.view.ViewGroup
import android.view.WindowManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close

import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect as ComposeRect
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import com.hermitech.hermivision.data.model.CourtResult
import com.hermitech.hermivision.domain.camera.CameraAnalyzer
import com.hermitech.hermivision.domain.court.CourtHomography
import java.util.concurrent.Executors

// ── Design Tokens ──
private val AccentGreen = Color(0xFF00C853)
private val AccentOrange = Color(0xFFFF6B35)
private val AccentTeal = Color(0xFF00BFA5)
private val CourtLineColor = Color(0xFF4DD0E1)
private val CourtNetColor = Color(0xFFFFD54F)
private val CourtDotColor = Color(0xFFFF8A65)
private val TextPrimary = Color(0xFFE0E0E0)
private val TextSecondary = Color(0xFF9E9E9E)
private val BallDotColor = Color(0xFFFF1744)
private val BallBoxColor = Color(0xFFFF5252)
private val OverlayBg = Color(0xFF0F0F23)
private val StopRed = Color(0xFFD32F2F)
private val MiniMapBg = Color(0xFF1B5E20)
private val MiniMapLine = Color.White
private val MiniMapNet = Color(0xFF2196F3)
private val MiniMapBall = Color(0xFFFFEB3B)

private const val TAG = "LiveAnalysisScreen"

@Composable
fun LiveAnalysisScreen(
    onBackClick: () -> Unit = {},
    liveViewModel: LiveViewModel = viewModel()
) {
    val context = LocalContext.current
    val activity = context as? Activity

    // ── Force landscape + fullscreen ──
    DisposableEffect(Unit) {
        val originalOrientation = activity?.requestedOrientation
        activity?.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE

        activity?.window?.let { window ->
            WindowCompat.setDecorFitsSystemWindows(window, false)
            val controller = WindowInsetsControllerCompat(window, window.decorView)
            controller.hide(WindowInsetsCompat.Type.systemBars())
            controller.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }

        onDispose {
            activity?.requestedOrientation =
                originalOrientation ?: ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
            activity?.window?.let { window ->
                WindowCompat.setDecorFitsSystemWindows(window, true)
                val controller = WindowInsetsControllerCompat(window, window.decorView)
                controller.show(WindowInsetsCompat.Type.systemBars())
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
        }
    }

    // ── State ──
    val isInitialized by liveViewModel.isInitialized.collectAsState()
    val initError by liveViewModel.initError.collectAsState()
    val ballState by liveViewModel.ballState.collectAsState()
    val courtResult by liveViewModel.courtResult.collectAsState()
    val fps by liveViewModel.fps.collectAsState()
    val isStopped by liveViewModel.isStopped.collectAsState()
    val sessionStats by liveViewModel.sessionStats.collectAsState()
    val miniMapState by liveViewModel.miniMapState.collectAsState()

    // ── Camera permission ──
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted -> hasCameraPermission = granted }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    LaunchedEffect(hasCameraPermission) {
        if (hasCameraPermission && !isInitialized) liveViewModel.initPipeline(context)
    }

    // ── Full-screen content ──
    Box(modifier = Modifier.fillMaxSize()) {
        when {
            !hasCameraPermission -> PermissionDeniedContent(
                onRetry = { permissionLauncher.launch(Manifest.permission.CAMERA) }
            )
            initError != null -> ErrorContent(error = initError!!)
            !isInitialized -> LoadingContent()
            else -> FullscreenCameraContent(
                liveViewModel = liveViewModel,
                ballState = ballState,
                courtResult = courtResult,
                fps = fps,
                isStopped = isStopped,
                sessionStats = sessionStats,
                miniMapState = miniMapState,
                onStop = { liveViewModel.stopSession() },
                onClose = onBackClick
            )
        }
    }
}

// ═══════════════════════════════════════════════════════
// ██  FULLSCREEN CAMERA + OVERLAYS
// ═══════════════════════════════════════════════════════

@Composable
private fun FullscreenCameraContent(
    liveViewModel: LiveViewModel,
    ballState: LiveViewModel.BallState,
    courtResult: CourtResult,
    fps: Float,
    isStopped: Boolean,
    sessionStats: LiveViewModel.SessionStats,
    miniMapState: LiveViewModel.MiniMapState,
    onStop: () -> Unit,
    onClose: () -> Unit
) {
    val lifecycleOwner = LocalLifecycleOwner.current

    var previewWidth by remember { mutableIntStateOf(1) }
    @Suppress("UNUSED_VARIABLE")
    var previewHeight by remember { mutableIntStateOf(1) }
    val cameraFrameWidth by remember { mutableIntStateOf(1280) }
    val cameraFrameHeight by remember { mutableIntStateOf(720) }

    Box(modifier = Modifier.fillMaxSize()) {

        // ── Layer 1: Camera preview ──
        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx).apply {
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    val analysisExecutor = Executors.newSingleThreadExecutor()

                    val preview = Preview.Builder().build()
                        .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setResolutionSelector(
                            androidx.camera.core.resolutionselector.ResolutionSelector.Builder()
                                .setResolutionStrategy(
                                    androidx.camera.core.resolutionselector.ResolutionStrategy(
                                        Size(1280, 720),
                                        androidx.camera.core.resolutionselector.ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER
                                    )
                                ).build()
                        )
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                        .build()
                        .also {
                            it.setAnalyzer(analysisExecutor, CameraAnalyzer(
                                pipeline = liveViewModel.pipeline,
                                onResult = { result -> liveViewModel.onResult(result) }
                            ))
                        }

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA,
                            preview, imageAnalysis
                        )
                        Log.i(TAG, "Camera bound to lifecycle")
                    } catch (e: Exception) {
                        Log.e(TAG, "Camera bind failed", e)
                    }
                }, ContextCompat.getMainExecutor(ctx))
                previewView
            },
            modifier = Modifier.fillMaxSize(),
            update = { view ->
                previewWidth = view.width.coerceAtLeast(1)
                previewHeight = view.height.coerceAtLeast(1)
            }
        )

        // ── Layer 2: Detection overlays (ball bbox + court lines) ──
        if (!isStopped && previewWidth > 1) {
            DetectionOverlays(
                ballState = ballState,
                courtResult = courtResult,
                cameraFrameWidth = cameraFrameWidth,
                cameraFrameHeight = cameraFrameHeight
            )
        }

        // ── Layer 3: HUD overlay (stats + controls + mini-map) ──
        AnimatedVisibility(visible = !isStopped, enter = fadeIn(), exit = fadeOut()) {
            HudOverlay(
                ballState = ballState,
                courtResult = courtResult,
                fps = fps,
                miniMapState = miniMapState,
                onStop = onStop,
                onClose = onClose
            )
        }

        // ── Layer 4: Results overlay ──
        AnimatedVisibility(
            visible = isStopped,
            enter = fadeIn() + slideInVertically { it / 2 },
            exit = fadeOut()
        ) {
            ResultsOverlay(sessionStats = sessionStats, onClose = onClose)
        }
    }
}

// ═══════════════════════════════════════════════════════
// ██  DETECTION OVERLAYS (Ball bbox + Court)
// ═══════════════════════════════════════════════════════

@Composable
private fun DetectionOverlays(
    ballState: LiveViewModel.BallState,
    courtResult: CourtResult,
    cameraFrameWidth: Int,
    cameraFrameHeight: Int
) {
    val textMeasurer = rememberTextMeasurer()

    // Ball bounding box + dot
    if (ballState.isVisible) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val scaleX = size.width / cameraFrameWidth.toFloat()
            val scaleY = size.height / cameraFrameHeight.toFloat()
            val cx = ballState.x * scaleX
            val cy = ballState.y * scaleY
            val bw = ballState.w * scaleX
            val bh = ballState.h * scaleY

            // Bounding box
            if (bw > 2f && bh > 2f) {
                val boxLeft = cx - bw / 2f
                val boxTop = cy - bh / 2f

                // Box outline
                drawRect(
                    color = BallBoxColor,
                    topLeft = Offset(boxLeft, boxTop),
                    size = androidx.compose.ui.geometry.Size(bw, bh),
                    style = Stroke(width = 2f)
                )

                // Corner accents (thicker L-shaped corners)
                val cornerLen = minOf(bw, bh) * 0.3f
                val cornerStroke = 3f
                // Top-left
                drawLine(BallBoxColor, Offset(boxLeft, boxTop), Offset(boxLeft + cornerLen, boxTop), cornerStroke)
                drawLine(BallBoxColor, Offset(boxLeft, boxTop), Offset(boxLeft, boxTop + cornerLen), cornerStroke)
                // Top-right
                drawLine(BallBoxColor, Offset(boxLeft + bw, boxTop), Offset(boxLeft + bw - cornerLen, boxTop), cornerStroke)
                drawLine(BallBoxColor, Offset(boxLeft + bw, boxTop), Offset(boxLeft + bw, boxTop + cornerLen), cornerStroke)
                // Bottom-left
                drawLine(BallBoxColor, Offset(boxLeft, boxTop + bh), Offset(boxLeft + cornerLen, boxTop + bh), cornerStroke)
                drawLine(BallBoxColor, Offset(boxLeft, boxTop + bh), Offset(boxLeft, boxTop + bh - cornerLen), cornerStroke)
                // Bottom-right
                drawLine(BallBoxColor, Offset(boxLeft + bw, boxTop + bh), Offset(boxLeft + bw - cornerLen, boxTop + bh), cornerStroke)
                drawLine(BallBoxColor, Offset(boxLeft + bw, boxTop + bh), Offset(boxLeft + bw, boxTop + bh - cornerLen), cornerStroke)

                // Score label above box
                val label = "%.0f%%".format(ballState.score * 100)
                val textResult = textMeasurer.measure(
                    text = label,
                    style = TextStyle(
                        color = Color.White,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold
                    )
                )
                // Label background
                drawRect(
                    color = BallBoxColor,
                    topLeft = Offset(boxLeft, boxTop - textResult.size.height - 2f),
                    size = androidx.compose.ui.geometry.Size(
                        textResult.size.width.toFloat() + 6f,
                        textResult.size.height.toFloat() + 2f
                    )
                )
                drawText(
                    textLayoutResult = textResult,
                    topLeft = Offset(boxLeft + 3f, boxTop - textResult.size.height - 1f)
                )
            }

            // Center dot
            drawCircle(color = BallDotColor.copy(alpha = 0.4f), radius = 16f, center = Offset(cx, cy))
            drawCircle(color = BallDotColor, radius = 8f, center = Offset(cx, cy))
            drawCircle(color = Color.White, radius = 3f, center = Offset(cx, cy))
        }
    }

    // Court overlay
    if (courtResult.valid && courtResult.keypoints.size == 14) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val scaleX = size.width / cameraFrameWidth.toFloat()
            val scaleY = size.height / cameraFrameHeight.toFloat()
            val pts = courtResult.keypoints.map { (x, y) -> Offset(x * scaleX, y * scaleY) }
            val lineStroke = Stroke(width = 2f, cap = StrokeCap.Round)

            // Outer rectangle
            drawOverlayLine(pts[0], pts[1], CourtLineColor.copy(alpha = 0.6f), lineStroke)
            drawOverlayLine(pts[2], pts[3], CourtLineColor.copy(alpha = 0.6f), lineStroke)
            drawOverlayLine(pts[0], pts[2], CourtLineColor.copy(alpha = 0.6f), lineStroke)
            drawOverlayLine(pts[1], pts[3], CourtLineColor.copy(alpha = 0.6f), lineStroke)
            // Singles sidelines
            drawOverlayLine(pts[4], pts[8], CourtLineColor.copy(alpha = 0.4f), lineStroke)
            drawOverlayLine(pts[8], pts[10], CourtLineColor.copy(alpha = 0.4f), lineStroke)
            drawOverlayLine(pts[10], pts[5], CourtLineColor.copy(alpha = 0.4f), lineStroke)
            drawOverlayLine(pts[6], pts[9], CourtLineColor.copy(alpha = 0.4f), lineStroke)
            drawOverlayLine(pts[9], pts[11], CourtLineColor.copy(alpha = 0.4f), lineStroke)
            drawOverlayLine(pts[11], pts[7], CourtLineColor.copy(alpha = 0.4f), lineStroke)
            // Service lines
            drawOverlayLine(pts[8], pts[9], CourtLineColor.copy(alpha = 0.4f), lineStroke)
            drawOverlayLine(pts[10], pts[11], CourtLineColor.copy(alpha = 0.4f), lineStroke)
            // Net
            drawOverlayLine(pts[12], pts[13], CourtNetColor.copy(alpha = 0.6f),
                Stroke(width = 3f, cap = StrokeCap.Round))
            // Keypoint dots
            pts.forEach { point ->
                drawCircle(color = CourtDotColor.copy(alpha = 0.7f), radius = 5f, center = point)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
// ██  HUD OVERLAY (floating controls + stats + mini-map)
// ═══════════════════════════════════════════════════════

@Composable
private fun HudOverlay(
    ballState: LiveViewModel.BallState,
    courtResult: CourtResult,
    fps: Float,
    miniMapState: LiveViewModel.MiniMapState,
    onStop: () -> Unit,
    onClose: () -> Unit
) {
    Box(modifier = Modifier.fillMaxSize()) {

        // ── Top-left: FPS badge ──
        Surface(
            shape = RoundedCornerShape(8.dp),
            color = Color.Black.copy(alpha = 0.5f),
            modifier = Modifier.align(Alignment.TopStart).padding(16.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
            ) {
                Surface(
                    shape = CircleShape,
                    color = when { fps > 7f -> AccentGreen; fps > 4f -> AccentOrange; else -> BallDotColor },
                    modifier = Modifier.size(8.dp)
                ) {}
                Spacer(modifier = Modifier.width(6.dp))
                Text("%.1f FPS".format(fps), color = TextPrimary, fontSize = 13.sp, fontWeight = FontWeight.Bold)
            }
        }

        // ── Top-right: Close button ──
        IconButton(
            onClick = onClose,
            modifier = Modifier.align(Alignment.TopEnd).padding(8.dp)
        ) {
            Surface(shape = CircleShape, color = Color.Black.copy(alpha = 0.5f), modifier = Modifier.size(36.dp)) {
                Icon(Icons.Default.Close, contentDescription = "Close", tint = TextPrimary,
                    modifier = Modifier.padding(8.dp).fillMaxSize())
            }
        }

        // ── Bottom-left: Status chips ──
        Column(
            modifier = Modifier.align(Alignment.BottomStart).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            StatusChip(
                label = "Ball",
                value = if (ballState.isVisible)
                    "(%.0f, %.0f) %.0f%%".format(ballState.x, ballState.y, ballState.score * 100)
                else "—",
                dotColor = if (ballState.isVisible) AccentGreen else TextSecondary
            )
            StatusChip(
                label = "Court",
                value = if (courtResult.valid) "Detected" else "—",
                dotColor = if (courtResult.valid) CourtLineColor else TextSecondary
            )
        }

        // ── Bottom-center: STOP button ──
        Button(
            onClick = onStop,
            modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 16.dp).height(48.dp).width(140.dp),
            shape = RoundedCornerShape(24.dp),
            colors = ButtonDefaults.buttonColors(containerColor = StopRed),
            elevation = ButtonDefaults.buttonElevation(defaultElevation = 6.dp)
        ) {
            Text("STOP", color = Color.White, fontSize = 14.sp, fontWeight = FontWeight.Bold)
        }

        // ── Right side: Mini-map court ──
        if (miniMapState.isValid) {
            MiniCourtMap(
                miniMapState = miniMapState,
                modifier = Modifier
                    .align(Alignment.CenterEnd)
                    .padding(end = 12.dp)
            )
        }

        // ── Bottom-right: LIVE indicator ──
        Surface(
            shape = RoundedCornerShape(8.dp),
            color = Color.Black.copy(alpha = 0.5f),
            modifier = Modifier.align(Alignment.BottomEnd).padding(16.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
            ) {
                Surface(shape = CircleShape, color = BallDotColor, modifier = Modifier.size(8.dp)) {}
                Spacer(modifier = Modifier.width(6.dp))
                Text("LIVE", color = BallDotColor, fontSize = 12.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
// ██  MINI COURT MAP (2D top-down projection)
// ═══════════════════════════════════════════════════════

@Composable
private fun MiniCourtMap(
    miniMapState: LiveViewModel.MiniMapState,
    modifier: Modifier = Modifier
) {
    val mapWidth = CourtHomography.MINI_MAP_WIDTH
    val mapHeight = CourtHomography.MINI_MAP_HEIGHT
    val miniKps = CourtHomography.miniMapKeypoints
    val lines = CourtHomography.courtLines

    Surface(
        shape = RoundedCornerShape(8.dp),
        color = Color.Black.copy(alpha = 0.3f),
        modifier = modifier.width((mapWidth + 8).dp).height((mapHeight + 8).dp)
    ) {
        Canvas(
            modifier = Modifier
                .padding(4.dp)
                .width(mapWidth.dp)
                .height(mapHeight.dp)
        ) {
            val sx = size.width / mapWidth
            val sy = size.height / mapHeight

            // ── Green court background ──
            drawRect(
                color = MiniMapBg,
                topLeft = Offset.Zero,
                size = size
            )

            // ── Court lines ──
            for ((i1, i2) in lines) {
                val (x1, y1) = miniKps[i1]
                val (x2, y2) = miniKps[i2]
                val isNet = (i1 == 12 && i2 == 13)
                drawLine(
                    color = if (isNet) MiniMapNet else MiniMapLine,
                    start = Offset(x1 * sx, y1 * sy),
                    end = Offset(x2 * sx, y2 * sy),
                    strokeWidth = if (isNet) 2.5f else 1.5f,
                    cap = StrokeCap.Round
                )
            }

            // ── Center service line (vertical through center of service boxes) ──
            val centerX = (miniKps[8].first + miniKps[9].first) / 2f
            drawLine(
                color = MiniMapLine,
                start = Offset(centerX * sx, miniKps[8].second * sy),
                end = Offset(centerX * sx, miniKps[10].second * sy),
                strokeWidth = 1.5f,
                cap = StrokeCap.Round
            )

            // ── Ball position ──
            miniMapState.ballPosition?.let { (bx, by) ->
                val px = bx * sx
                val py = by * sy
                // Glow
                drawCircle(color = MiniMapBall.copy(alpha = 0.4f), radius = 8f, center = Offset(px, py))
                // Ball
                drawCircle(color = MiniMapBall, radius = 5f, center = Offset(px, py))
                // Highlight
                drawCircle(color = Color.White, radius = 2f, center = Offset(px, py))
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
// ██  STATUS CHIP
// ═══════════════════════════════════════════════════════

@Composable
private fun StatusChip(label: String, value: String, dotColor: Color) {
    Surface(shape = RoundedCornerShape(8.dp), color = Color.Black.copy(alpha = 0.5f)) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 5.dp)
        ) {
            Surface(shape = CircleShape, color = dotColor, modifier = Modifier.size(6.dp)) {}
            Spacer(modifier = Modifier.width(6.dp))
            Text("$label: ", color = TextSecondary, fontSize = 11.sp)
            Text(value, color = TextPrimary, fontSize = 11.sp, fontWeight = FontWeight.Bold)
        }
    }
}

// ═══════════════════════════════════════════════════════
// ██  RESULTS OVERLAY
// ═══════════════════════════════════════════════════════

@Composable
private fun ResultsOverlay(sessionStats: LiveViewModel.SessionStats, onClose: () -> Unit) {
    Box(
        modifier = Modifier.fillMaxSize().background(
            Brush.verticalGradient(listOf(OverlayBg.copy(alpha = 0.85f), OverlayBg.copy(alpha = 0.95f)))
        ),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(20.dp),
            modifier = Modifier.padding(32.dp)
        ) {
            Text("Session Results", color = TextPrimary, fontSize = 24.sp, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(8.dp))

            Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) {
                ResultStatCard("Frames", "${sessionStats.totalFrames}", AccentTeal)
                ResultStatCard("Ball Hits", "${sessionStats.ballDetections}", AccentGreen)
                ResultStatCard("Court", "${sessionStats.courtDetections}", CourtLineColor)
            }
            Row(horizontalArrangement = Arrangement.spacedBy(24.dp)) {
                val detectionRate = if (sessionStats.totalFrames > 0)
                    sessionStats.ballDetections * 100f / sessionStats.totalFrames else 0f
                ResultStatCard("Detect Rate", "%.1f%%".format(detectionRate), AccentOrange)
                ResultStatCard("Avg FPS", "%.1f".format(sessionStats.avgFps), AccentGreen)
                ResultStatCard("Duration", "%.1fs".format(sessionStats.durationMs / 1000f), TextPrimary)
            }

            Spacer(modifier = Modifier.height(16.dp))
            Button(
                onClick = onClose,
                modifier = Modifier.width(200.dp).height(48.dp),
                shape = RoundedCornerShape(24.dp),
                colors = ButtonDefaults.buttonColors(containerColor = AccentOrange)
            ) {
                Text("Done", fontSize = 16.sp, fontWeight = FontWeight.Bold, color = Color.White)
            }
        }
    }
}

@Composable
private fun ResultStatCard(label: String, value: String, color: Color) {
    Surface(shape = RoundedCornerShape(12.dp), color = Color.White.copy(alpha = 0.08f), modifier = Modifier.width(100.dp)) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.padding(vertical = 12.dp, horizontal = 8.dp)
        ) {
            Text(value, color = color, fontSize = 22.sp, fontWeight = FontWeight.Bold)
            Spacer(modifier = Modifier.height(4.dp))
            Text(label, color = TextSecondary, fontSize = 11.sp, fontWeight = FontWeight.Medium)
        }
    }
}

// ═══════════════════════════════════════════════════════
// ██  FALLBACK STATES
// ═══════════════════════════════════════════════════════

@Composable
private fun PermissionDeniedContent(onRetry: () -> Unit) {
    Box(modifier = Modifier.fillMaxSize().background(OverlayBg), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)) {
            Text("📷", fontSize = 64.sp)
            Text("Camera Permission Required", color = TextPrimary, fontSize = 18.sp, fontWeight = FontWeight.Bold)
            Text("Allow camera access to use live analysis", color = TextSecondary, fontSize = 14.sp, textAlign = TextAlign.Center)
            Button(onClick = onRetry, colors = ButtonDefaults.buttonColors(containerColor = AccentOrange)) {
                Text("Grant Permission")
            }
        }
    }
}

@Composable
private fun LoadingContent() {
    Box(modifier = Modifier.fillMaxSize().background(OverlayBg), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(16.dp)) {
            CircularProgressIndicator(color = AccentOrange, modifier = Modifier.size(48.dp))
            Text("Loading AI Models...", color = TextPrimary, fontSize = 16.sp, fontWeight = FontWeight.Medium)
            Text("Initializing YOLO + Court Detection", color = TextSecondary, fontSize = 13.sp)
        }
    }
}

@Composable
private fun ErrorContent(error: String) {
    Box(modifier = Modifier.fillMaxSize().background(OverlayBg), contentAlignment = Alignment.Center) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier.padding(32.dp)
        ) {
            Text("⚠️", fontSize = 64.sp)
            Text("Initialization Error", color = BallDotColor, fontSize = 18.sp, fontWeight = FontWeight.Bold)
            Text(error, color = TextSecondary, fontSize = 14.sp, textAlign = TextAlign.Center)
        }
    }
}

// ═══════════════════════════════════════════════════════
// ██  HELPERS
// ═══════════════════════════════════════════════════════

private fun DrawScope.drawOverlayLine(from: Offset, to: Offset, color: Color, stroke: Stroke) {
    drawLine(color = color, start = from, end = to, strokeWidth = stroke.width, cap = stroke.cap)
}
