package com.hermitech.hermivision.ui.results

import android.graphics.PointF
import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import com.hermitech.hermivision.data.model.BallFrame
import kotlinx.coroutines.launch

private val TabTitles = listOf("Court View", "Ball Track")
private val TopBarBg = Color(0xFF0F0F23)
private val AccentColor = Color(0xFFFF6B35)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultsScreen(
    ballFrames: List<BallFrame>,
    bounceFrameIds: Set<Int>,
    bounceCourtPoints: List<Pair<Int, PointF>>,
    trajectoryCourtPoints: List<Pair<Int, PointF>>?,
    totalDurationMs: Long = 0L,
    onBackClick: () -> Unit = {}
) {
    val pagerState = rememberPagerState(pageCount = { TabTitles.size })
    val coroutineScope = rememberCoroutineScope()

    Scaffold(
        topBar = {
            CenterAlignedTopAppBar(
                title = {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "Analysis Results",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                        if (totalDurationMs > 0) {
                            Text(
                                text = "Processed in %.1fs".format(totalDurationMs / 1000.0),
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.5f)
                            )
                        }
                    }
                },
                navigationIcon = {
                    IconButton(onClick = onBackClick) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                },
                colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
                    containerColor = TopBarBg
                )
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            TabRow(
                selectedTabIndex = pagerState.currentPage,
                containerColor = TopBarBg,
                contentColor = AccentColor,
                divider = {
                    HorizontalDivider(color = AccentColor.copy(alpha = 0.3f))
                }
            ) {
                TabTitles.forEachIndexed { index, title ->
                    val selected = pagerState.currentPage == index
                    val animatedColor by animateColorAsState(
                        targetValue = if (selected) AccentColor else Color.White.copy(alpha = 0.5f),
                        label = "tab_color"
                    )

                    Tab(
                        selected = selected,
                        onClick = {
                            coroutineScope.launch { pagerState.animateScrollToPage(index) }
                        },
                        text = {
                            Text(
                                text = title,
                                fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
                                color = animatedColor
                            )
                        }
                    )
                }
            }

            HorizontalPager(state = pagerState, modifier = Modifier.fillMaxSize()) { page ->
                when (page) {
                    0 -> CourtMatrix2DView(
                        bouncePoints = bounceCourtPoints,
                        trajectoryPoints = trajectoryCourtPoints,
                        modifier = Modifier.fillMaxSize()
                    )
                    1 -> BallTrackTable(
                        ballFrames = ballFrames,
                        bounceFrameIds = bounceFrameIds,
                        modifier = Modifier.fillMaxSize()
                    )
                }
            }
        }
    }
}