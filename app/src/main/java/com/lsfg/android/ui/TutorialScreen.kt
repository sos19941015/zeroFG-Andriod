package com.lsfg.android.ui

import androidx.annotation.DrawableRes
import androidx.annotation.StringRes
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.School
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import androidx.navigation.NavHostController
import com.lsfg.android.R
import com.lsfg.android.ui.components.IconBadge
import com.lsfg.android.ui.components.LsfgCard
import com.lsfg.android.ui.components.LsfgTopBar
import com.lsfg.android.ui.theme.LsfgPrimary

/**
 * One step of the tutorial. Add new entries to [tutorialSteps] below as you
 * drop screenshots into res/drawable-nodpi.
 */
private data class TutorialStep(
    @StringRes val titleRes: Int,
    @StringRes val descriptionRes: Int?,
    @DrawableRes val image: Int?,
)

private val tutorialSteps: List<TutorialStep> = listOf(
    TutorialStep(R.string.tut_1_title, R.string.tut_1_desc, R.drawable.tutorial_step_1),
    TutorialStep(R.string.tut_2_title, R.string.tut_2_desc, R.drawable.tutorial_step_2),
    TutorialStep(R.string.tut_3_title, R.string.tut_3_desc, R.drawable.tutorial_step_3),
    TutorialStep(R.string.tut_4_title, R.string.tut_4_desc, R.drawable.tutorial_step_4),
    TutorialStep(R.string.tut_5_title, R.string.tut_5_desc, R.drawable.tutorial_step_9),
    TutorialStep(R.string.tut_6_title, R.string.tut_6_desc, R.drawable.tutorial_step_10),
    TutorialStep(R.string.tut_7_title, R.string.tut_7_desc, R.drawable.tutorial_step_11),
    TutorialStep(R.string.tut_8_title, R.string.tut_8_desc, R.drawable.tutorial_step_12),
    TutorialStep(R.string.tut_9_title, R.string.tut_9_desc, R.drawable.tutorial_step_13),
    TutorialStep(R.string.tut_10_title, R.string.tut_10_desc, R.drawable.tutorial_step_14),
    TutorialStep(R.string.tut_11_title, R.string.tut_11_desc, R.drawable.tutorial_step_15),
    TutorialStep(R.string.tut_12_title, R.string.tut_12_desc, R.drawable.tutorial_step_16),
    TutorialStep(R.string.tut_13_title, R.string.tut_13_desc, R.drawable.tutorial_step_17),
    TutorialStep(R.string.tut_14_title, R.string.tut_14_desc, R.drawable.tutorial_step_18),
    TutorialStep(R.string.tut_15_title, R.string.tut_15_desc, R.drawable.tutorial_step_19),
    TutorialStep(R.string.tut_16_title, R.string.tut_16_desc, R.drawable.tutorial_step_20),
    TutorialStep(R.string.tut_17_title, R.string.tut_17_desc, R.drawable.tutorial_step_21),
    TutorialStep(R.string.tut_18_title, R.string.tut_18_desc, R.drawable.tutorial_step_22),
    TutorialStep(R.string.tut_19_title, R.string.tut_19_desc, R.drawable.tutorial_step_23),
    TutorialStep(R.string.tut_20_title, R.string.tut_20_desc, R.drawable.tutorial_step_24),
    TutorialStep(R.string.tut_21_title, null, R.drawable.tutorial_step_25),
    TutorialStep(R.string.tut_22_title, null, R.drawable.tutorial_step_26),
    TutorialStep(R.string.tut_23_title, null, R.drawable.tutorial_step_27),
)

@Composable
fun TutorialScreen(nav: NavHostController) {
    var zoomedImage by remember { mutableStateOf<Int?>(null) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .statusBarsPadding()
            .padding(horizontal = 20.dp)
            .padding(bottom = 20.dp),
    ) {
        LsfgTopBar(
            title = stringResource(R.string.nav_tutorial),
            onBack = { nav.popBackStack() },
        )

        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 12.dp, bottom = 4.dp),
            ) {
                IconBadge(icon = Icons.Filled.School, tint = LsfgPrimary, size = 56.dp)
            }

            LsfgCard(accent = true) {
                Text(
                    text = stringResource(R.string.tutorial_subtitle).uppercase(),
                    style = MaterialTheme.typography.labelSmall,
                    color = LsfgPrimary,
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = stringResource(R.string.tutorial_intro),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                )
            }

            if (tutorialSteps.isEmpty()) {
                LsfgCard {
                    Text(
                        text = stringResource(R.string.tutorial_empty),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            } else {
                tutorialSteps.forEachIndexed { index, step ->
                    TutorialStepCard(
                        index = index + 1,
                        step = step,
                        onImageClick = { res -> zoomedImage = res },
                    )
                }
            }

            Spacer(Modifier.height(8.dp))
        }
    }

    zoomedImage?.let { res ->
        Dialog(
            onDismissRequest = { zoomedImage = null },
            properties = DialogProperties(usePlatformDefaultWidth = false),
        ) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .clickable { zoomedImage = null },
                contentAlignment = Alignment.Center,
            ) {
                Image(
                    painter = painterResource(id = res),
                    contentDescription = null,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    contentScale = ContentScale.Fit,
                )
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(20.dp)
                        .size(40.dp)
                        .clip(RoundedCornerShape(20.dp))
                        .clickable { zoomedImage = null },
                    contentAlignment = Alignment.Center,
                ) {
                    Icon(
                        imageVector = Icons.Filled.Close,
                        contentDescription = "Close",
                        tint = Color.White,
                    )
                }
            }
        }
    }
}

@Composable
private fun TutorialStepCard(
    index: Int,
    step: TutorialStep,
    onImageClick: (Int) -> Unit,
) {
    val title = stringResource(step.titleRes)
    val description = step.descriptionRes?.let { stringResource(it) }.orEmpty()
    LsfgCard {
        Text(
            text = stringResource(R.string.tutorial_step_badge, index),
            style = MaterialTheme.typography.labelSmall,
            color = LsfgPrimary,
        )
        Spacer(Modifier.height(6.dp))
        Text(
            text = title,
            style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.SemiBold),
            color = MaterialTheme.colorScheme.onSurface,
        )
        if (step.image != null) {
            Spacer(Modifier.height(12.dp))
            Image(
                painter = painterResource(id = step.image),
                contentDescription = title,
                contentScale = ContentScale.Fit,
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(16f / 9f)
                    .clip(RoundedCornerShape(12.dp))
                    .clickable { onImageClick(step.image) },
            )
        }
        if (description.isNotBlank()) {
            Spacer(Modifier.height(12.dp))
            Text(
                text = description,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
