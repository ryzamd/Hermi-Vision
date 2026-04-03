package com.hermitech.hermivision.persistence

import android.content.Context
import com.hermitech.hermivision.domain.BounceEvent
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class BounceSessionStore(
    context: Context
) {
    private val sessionDirectory = File(context.filesDir, "bounce-sessions").apply {
        mkdirs()
    }

    private val sessionFile = File(
        sessionDirectory,
        "session-${SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())}.jsonl"
    )

    @Synchronized
    fun append(event: BounceEvent) {
        val line = buildString {
            append("{")
            append("\"id\":${event.id},")
            append("\"x\":${event.position.x},")
            append("\"y\":${event.position.y},")
            append("\"confidence\":${event.confidence},")
            append("\"timestampMs\":${event.timestampMs}")
            append("}")
        }
        sessionFile.appendText(line + "\n")
    }

    fun sessionPath(): String = sessionFile.absolutePath
}
