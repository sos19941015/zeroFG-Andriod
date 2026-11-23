package com.lsfg.android.session

import android.content.Context
import android.graphics.Color
import android.graphics.PixelFormat
import android.hardware.input.InputManager
import android.os.Build
import android.util.DisplayMetrics
import android.util.Log
import android.view.Gravity
import android.view.Surface
import android.view.SurfaceControl
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import com.lsfg.android.prefs.LsfgPreferences
import com.lsfg.android.prefs.VsyncRefreshOverride

/** Displays captured frames in a non-touchable window. The drawer has its own window. */
class OverlayManager(private val ctx: Context) {

    private var root: FrameLayout? = null
    private var surfaceView: SurfaceView? = null
    private var hostWindowManager: WindowManager? = null
    private var fpsView: TextView? = null
    private var graphView: FrameGraphView? = null
    private var benchmarkPanel: LinearLayout? = null
    private var benchmarkLabel: TextView? = null
    private var benchmarkBar: ProgressBar? = null

    @Volatile
    private var surfaceReadyListener: ((Surface, Int, Int) -> Unit)? = null

    @Volatile
    private var surfaceLostListener: (() -> Unit)? = null
    private var requestedRefreshRateHz: Float = 0f
    private var lastVsyncPeriodNs: Long = -1
    private var overlayWidth: Int = 0
    private var overlayHeight: Int = 0

    /** Callback invoked every time the overlay Surface becomes valid for writing. */
    fun onSurfaceReady(cb: (Surface, Int, Int) -> Unit) {
        surfaceReadyListener = cb
    }

    /** Callback invoked every time the Surface is torn down. */
    fun onSurfaceLost(cb: () -> Unit) {
        surfaceLostListener = cb
    }

    fun show(): Boolean {
        if (root != null) return true

        val prefs = LsfgPreferences(ctx).load()
        val a11y = LsfgAccessibilityService.instance
        if (prefs.trustedOverlay && a11y == null) {
            Log.w(TAG, "Trusted overlay requested, but the accessibility service is not connected")
            return false
        }
        val useTrusted = prefs.trustedOverlay && a11y != null
        val hostCtx: Context = if (useTrusted) a11y!! else ctx
        val wm = hostCtx.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        hostWindowManager = wm

        val (screenW, screenH) = queryDisplaySize(wm)
        @Suppress("DEPRECATION")
        requestedRefreshRateHz = wm.defaultDisplay.supportedModes
            .maxOfOrNull { it.refreshRate }
            ?: wm.defaultDisplay.refreshRate
        syncDisplayPacing()
        overlayWidth = screenW
        overlayHeight = screenH
        Log.i(TAG, "Showing overlay at ${screenW}x${screenH} targetRefresh=${requestedRefreshRateHz}Hz")

        val layoutType = when {
            useTrusted -> WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.O ->
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            else -> @Suppress("DEPRECATION")
            WindowManager.LayoutParams.TYPE_SYSTEM_ALERT
        }
        Log.i(TAG, "Overlay host=${if (useTrusted) "a11y/TRUSTED" else "app/UNTRUSTED"}")

        // Android 12+ also checks the opacity of non-touchable application overlays.
        val overlayOpacity = if (!useTrusted && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            ctx.getSystemService(InputManager::class.java)
                ?.maximumObscuringOpacityForTouch ?: 0.8f
        } else {
            1.0f
        }
        val flags = WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS or
                WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED or
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT,
            layoutType,
            flags,
            PixelFormat.TRANSLUCENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = 0
            y = 0
            alpha = overlayOpacity
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                layoutInDisplayCutoutMode =
                    WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS
            }
            // PRIVATE_FLAG_EXCLUDE_FROM_SCREEN_CAPTURE (0x00080000, API 31+): instructs
            // WMS to tell SurfaceFlinger to skip this window in VirtualDisplay/screenshot
            // composition at window-creation time — no SurfaceControl timing race.
            runCatching {
                val f = javaClass.getDeclaredField("privateFlags")
                f.isAccessible = true
                f.setInt(this, f.getInt(this) or 0x00080000)
                Log.i("lsfg-vk-loop", "privateFlags EXCLUDE_FROM_SCREEN_CAPTURE applied")
            }.onFailure {
                Log.d("lsfg-vk-loop", "privateFlags field unavailable: ${it.message}")
            }
        }

        // FrameLayout stays transparent so the SurfaceView punches through directly
        // to SurfaceFlinger/HWC without HWUI TextureView compositing overhead or
        // Vsync-invalidate jitter, while keeping FPS/graph HUD views drawn on top.
        val layout = FrameLayout(ctx).apply {
            setBackgroundColor(Color.TRANSPARENT)
        }
        val sv = SurfaceView(ctx)
        sv.setZOrderOnTop(false)
        sv.holder.setFormat(PixelFormat.TRANSLUCENT)
        sv.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                installSkipScreenshot(layout)
                applySkipScreenshotToSurfaceControl(sv.surfaceControl, "SurfaceView")
                syncOverlayGeometry()
                val s = holder.surface
                requestMaxRefreshRate(s)
                val w = sv.width.takeIf { it > 0 } ?: overlayWidth
                val h = sv.height.takeIf { it > 0 } ?: overlayHeight
                Log.i(TAG, "SurfaceView created ${w}x${h} valid=${s.isValid}")
                if (s.isValid && w > 0 && h > 0) {
                    surfaceReadyListener?.invoke(s, w, h)
                }
            }

            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
                applySkipScreenshotToSurfaceControl(sv.surfaceControl, "SurfaceView")
                syncOverlayGeometry()
                val s = holder.surface
                requestMaxRefreshRate(s)
                val w = width.takeIf { it > 0 } ?: overlayWidth
                val h = height.takeIf { it > 0 } ?: overlayHeight
                overlayWidth = w
                overlayHeight = h
                Log.i(TAG, "SurfaceView changed ${w}x${h} valid=${s.isValid}")
                if (s.isValid && w > 0 && h > 0) {
                    surfaceReadyListener?.invoke(s, w, h)
                }
            }

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                Log.i(TAG, "SurfaceView destroyed")
                surfaceLostListener?.invoke()
            }
        })
        val fps = TextView(ctx).apply {
            text = ""
            setTextColor(Color.WHITE)
            setBackgroundColor(0x80000000.toInt())
            textSize = 14f
            setPadding(16, 8, 16, 8)
            visibility = View.GONE
        }
        layout.addView(
            sv,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            ),
        )
        layout.addView(
            fps,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.START,
            ).apply {
                topMargin = 24
                leftMargin = 24
            },
        )

        val graph = FrameGraphView(ctx).apply {
            visibility = View.GONE
        }
        val density = ctx.resources.displayMetrics.density
        val graphWidthPx = (density * 220f).toInt()
        val graphHeightPx = (density * 80f).toInt()
        layout.addView(
            graph,
            FrameLayout.LayoutParams(
                graphWidthPx,
                graphHeightPx,
                Gravity.TOP or Gravity.START,
            ).apply {
                // Position just below the FPS text so both stay in the top-left cluster.
                topMargin = 112
                leftMargin = 24
            },
        )

        // Benchmark progress panel — pinned to the top-center of the overlay,
        // hidden until BenchmarkController publishes a Running state. Hosts
        // a one-line label ("Run x/y · FP16 ×3 — sampling 28s left") and a
        // horizontal ProgressBar that fills as the current phase advances and
        // is reset at every phase transition.
        val benchPanel = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(0xB0000000.toInt())
            val padH = (density * 12f).toInt()
            val padV = (density * 8f).toInt()
            setPadding(padH, padV, padH, padV)
            visibility = View.GONE
        }
        val benchLabel = TextView(ctx).apply {
            text = ""
            setTextColor(Color.WHITE)
            textSize = 13f
        }
        val benchBar = ProgressBar(ctx, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 1000
            progress = 0
            isIndeterminate = false
        }
        benchPanel.addView(
            benchLabel,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ),
        )
        benchPanel.addView(
            benchBar,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                (density * 6f).toInt(),
            ).apply {
                topMargin = (density * 4f).toInt()
            },
        )
        val benchWidthPx = (density * 320f).toInt()
        layout.addView(
            benchPanel,
            FrameLayout.LayoutParams(
                benchWidthPx,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.CENTER_HORIZONTAL,
            ).apply {
                topMargin = 24
            },
        )

        wm.addView(layout, params)

        // Mark the overlay's SurfaceControl with skipScreenshot so SurfaceFlinger
        // excludes it from virtual-display composition (MediaProjection) while still
        // rendering it normally on the physical display.  This prevents the feedback
        // loop where MediaProjection captures our overlay output instead of the live
        // game when the overlay is opaque.
        installSkipScreenshot(layout)

        root = layout
        fpsView = fps
        graphView = graph
        benchmarkPanel = benchPanel
        benchmarkLabel = benchLabel
        benchmarkBar = benchBar
        surfaceView = sv
        return true
    }

    /**
     * Re-asserts the overlay as the topmost window. Call this after the target app is
     * launched so the new foreground activity doesn't leave a stale z-order with our
     * overlay stuck behind it.
     */
    fun bringToFront() {
        val r = root ?: return
        val wm = hostWindowManager ?: return
        if (!r.isAttachedToWindow) return
        val lp = r.layoutParams as? WindowManager.LayoutParams ?: return
        runCatching { wm.updateViewLayout(r, lp) }
            .onFailure { Log.w(TAG, "bringToFront updateViewLayout failed", it) }
    }

    /**
     * Re-reads display metrics and resizes the overlay window to match a new
     * orientation / display configuration. Idempotent; safe to call multiple
     * times for a single rotation. The actual `wm.updateViewLayout` is posted
     * to the root view's main-thread looper to avoid racing with WindowManager
     * while it is still distributing the configuration change.
     */
    fun onDisplayConfigurationChanged() {
        val r = root ?: return
        r.post {
            syncOverlayGeometry()
            syncDisplayPacing()
        }
    }

    fun updateStatus(line: String) {
        // Status text was removed from the overlay; keep the API as a no-op so
        // existing call sites in the foreground service continue to compile.
    }

    fun setFpsVisible(visible: Boolean) {
        fpsView?.post { fpsView?.visibility = if (visible) View.VISIBLE else View.GONE }
    }

    fun updateFps(capturedFps: Float, postedFps: Float) {
        val v = fpsView ?: return
        val queueMs = runCatching { NativeBridge.getAverageQueueMs() }.getOrDefault(0.0)
        val latencyMs = runCatching { NativeBridge.getAverageLatencyMs() }.getOrDefault(0.0)
        val text = "real ${"%.1f".format(capturedFps)} fps · total ${"%.1f".format(postedFps)} fps (latency: ${"%.1f".format(latencyMs)} ms · queue: ${"%.1f".format(queueMs)} ms)"
        v.post { v.text = text }
    }

    fun setFrameGraphVisible(visible: Boolean) {
        val g = graphView ?: return
        g.post {
            g.visibility = if (visible) View.VISIBLE else View.GONE
            if (!visible) g.reset()
        }
    }

    fun pushFrameGraphSample(realFps: Float, generatedFps: Float) {
        val g = graphView ?: return
        g.post { g.pushSample(realFps, generatedFps, realFps + generatedFps) }
    }

    /**
     * Show the benchmark progress panel (idempotent). The panel is hidden by
     * default and only revealed while a benchmark is running.
     */
    fun setBenchmarkProgressVisible(visible: Boolean) {
        val p = benchmarkPanel ?: return
        p.post {
            p.visibility = if (visible) View.VISIBLE else View.GONE
            if (!visible) {
                benchmarkBar?.progress = 0
                benchmarkLabel?.text = ""
            }
        }
    }

    /**
     * Update the benchmark progress panel.
     *
     * @param label one-line status, e.g. "Run 2/6 · FP32 ×3 — sampling"
     * @param elapsedMs elapsed time inside the current phase
     * @param totalMs total length of the current phase
     */
    fun updateBenchmarkProgress(label: String, elapsedMs: Long, totalMs: Long) {
        val l = benchmarkLabel ?: return
        val b = benchmarkBar ?: return
        val clampedTotal = totalMs.coerceAtLeast(1L)
        val ratio = (elapsedMs.toFloat() / clampedTotal).coerceIn(0f, 1f)
        val remainingSec = ((clampedTotal - elapsedMs).coerceAtLeast(0L) + 999L) / 1000L
        val text = "$label  ·  ${remainingSec}s left"
        l.post {
            l.text = text
            b.progress = (ratio * 1000f).toInt()
        }
    }

    fun hide() {
        val r = root ?: return
        val wm = hostWindowManager
        if (wm != null) {
            runCatching { wm.removeView(r) }
        }
        lastVsyncPeriodNs = -1
        runCatching { NativeBridge.setVsyncPeriodNs(0L) }
        root = null
        surfaceView = null
        fpsView = null
        graphView = null
        benchmarkPanel = null
        benchmarkLabel = null
        benchmarkBar = null
        hostWindowManager = null
    }

    fun syncDisplayPacing() {
        val wm = hostWindowManager ?: return
        val prefs = LsfgPreferences(ctx).load()
        @Suppress("DEPRECATION")
        val actualHz = wm.defaultDisplay.refreshRate
        val effectiveHz = when {
            !prefs.vsyncAlignmentEnabled -> 0f
            prefs.vsyncRefreshOverride != VsyncRefreshOverride.AUTO ->
                prefs.vsyncRefreshOverride.hz.toFloat()
            else -> actualHz
        }
        val periodNs = if (effectiveHz > 0) (1_000_000_000.0 / effectiveHz).toLong() else 0L
        if (periodNs != lastVsyncPeriodNs) {
            lastVsyncPeriodNs = periodNs
            runCatching { NativeBridge.setVsyncPeriodNs(periodNs) }
            Log.i(TAG, "Display refresh=${actualHz}Hz; frame pacing=${effectiveHz}Hz")
        }
    }

    private fun requestMaxRefreshRate(surface: Surface) {
        if (!surface.isValid || requestedRefreshRateHz <= 0f || Build.VERSION.SDK_INT < Build.VERSION_CODES.R) {
            return
        }
        runCatching {
            surface.setFrameRate(
                requestedRefreshRateHz,
                Surface.FRAME_RATE_COMPATIBILITY_FIXED_SOURCE,
                Surface.CHANGE_FRAME_RATE_ALWAYS,
            )
        }.onFailure { Log.w(TAG, "setFrameRate(${requestedRefreshRateHz}Hz) failed", it) }
    }

    private fun queryDisplaySize(wm: WindowManager): Pair<Int, Int> {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val bounds = wm.maximumWindowMetrics.bounds
            if (bounds.width() > 0 && bounds.height() > 0) {
                return bounds.width() to bounds.height()
            }
        }
        val metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        wm.defaultDisplay.getRealMetrics(metrics)
        return metrics.widthPixels to metrics.heightPixels
    }

    private fun syncOverlayGeometry() {
        val wm = hostWindowManager ?: return
        val r = root ?: return
        if (!r.isAttachedToWindow) {
            return
        }
        val (newW, newH) = queryDisplaySize(wm)
        if (newW == overlayWidth && newH == overlayHeight) return

        overlayWidth = newW
        overlayHeight = newH
        Log.i(TAG, "Overlay geometry synced to ${newW}x${newH}")
    }

    /**
     * Sets the overlay window's `eSkipScreenshot` SurfaceFlinger flag so SurfaceFlinger
     * excludes it from VirtualDisplay composition (MediaProjection capture) while rendering
     * it normally on the physical display.
     *
     * Reflection path:
     *   View.getViewRootImpl() (@hide, API 11+) → ViewRootImpl
     *   ViewRootImpl.getSurfaceControl() (@hide, API 29+) → window root SurfaceControl
     *   SurfaceControl.Transaction.setSkipScreenshot(@hide, API 12+)
     *
     * Note: rootSurfaceControl/getRootSurfaceControl() returns AttachedSurfaceControl, not
     * ViewRootImpl, so getSurfaceControl() would throw NoSuchMethodException on that path.
     */
    private fun installSkipScreenshot(host: View) {
        val apply: () -> Unit = {
            runCatching {
                // getViewRootImpl() is @hide but stable since API 11 and returns ViewRootImpl
                // directly — the object that owns the window's root SurfaceControl.
                val viewRootImpl = host.javaClass.getMethod("getViewRootImpl").invoke(host)
                    ?: error("getViewRootImpl() returned null (view not attached?)")

                // Get the window's SurfaceControl.
                // Primary path:   ViewRootImpl.getSurfaceControl() — @hide, API 29+.
                // Fallback path:  direct ViewRootImpl.mSurfaceControl field — some OEM
                //   ROMs (MediaTek Android 13 confirmed) strip the accessor method while
                //   leaving the underlying field intact in the class hierarchy.
                val sc: SurfaceControl = run {
                    runCatching {
                        viewRootImpl.javaClass.getMethod("getSurfaceControl")
                            .invoke(viewRootImpl) as? SurfaceControl
                    }.getOrNull()?.let { return@run it }

                    var cls: Class<*>? = viewRootImpl.javaClass
                    while (cls != null) {
                        runCatching {
                            cls!!.getDeclaredField("mSurfaceControl")
                                .also { it.isAccessible = true }
                                .get(viewRootImpl) as? SurfaceControl
                        }.getOrNull()?.let {
                            Log.d("lsfg-vk-loop", "installSkipScreenshot: SurfaceControl via ${cls!!.simpleName}.mSurfaceControl")
                            return@run it
                        }
                        cls = cls.superclass
                    }

                    // Third fallback: enumerate all fields by type — catches OEM renames.
                    val scFieldsFound = mutableListOf<String>()
                    var typeCls: Class<*>? = viewRootImpl.javaClass
                    while (typeCls != null) {
                        for (field in typeCls!!.declaredFields) {
                            if (SurfaceControl::class.java.isAssignableFrom(field.type)) {
                                val fieldId = "${typeCls.simpleName}.${field.name}"
                                scFieldsFound.add(fieldId)
                                runCatching {
                                    field.isAccessible = true
                                    field.get(viewRootImpl) as? SurfaceControl
                                }.getOrNull()?.let { found ->
                                    Log.i("lsfg-vk-loop", "installSkipScreenshot: SurfaceControl via type-search: $fieldId")
                                    return@run found
                                }
                            }
                        }
                        typeCls = typeCls.superclass
                    }
                    Log.i("lsfg-vk-loop", "installSkipScreenshot: type-search SC-typed fields: $scFieldsFound")

                    // Fourth fallback: Android 12+ BLAST pipeline — ViewRootImpl stores a
                    // BLASTBufferQueue which owns the window's SurfaceControl internally.
                    // Some OEM ROMs strip all direct SC fields from ViewRootImpl but leave
                    // the BLASTBufferQueue field intact.
                    val bbqClass = runCatching {
                        Class.forName("android.graphics.BLASTBufferQueue")
                    }.getOrNull()
                    if (bbqClass != null) {
                        var bbqSearchCls: Class<*>? = viewRootImpl.javaClass
                        outer@ while (bbqSearchCls != null) {
                            for (field in bbqSearchCls!!.declaredFields) {
                                if (bbqClass.isAssignableFrom(field.type)) {
                                    runCatching {
                                        field.isAccessible = true
                                        val bbq = field.get(viewRootImpl) ?: return@runCatching
                                        for (methodName in listOf(
                                            "getSyncedSurfaceControl",
                                            "getSurfaceControl",
                                        )) {
                                            runCatching {
                                                bbq.javaClass.getMethod(methodName)
                                                    .invoke(bbq) as? SurfaceControl
                                            }.getOrNull()?.let { found ->
                                                Log.i("lsfg-vk-loop",
                                                    "installSkipScreenshot: SurfaceControl via " +
                                                    "BLASTBufferQueue.${field.name}.$methodName()")
                                                return@run found
                                            }
                                        }
                                    }
                                }
                            }
                            bbqSearchCls = bbqSearchCls.superclass
                        }
                    }

                    // All paths exhausted — dump surface/blast/buffer related field names from
                    // the ViewRootImpl hierarchy so we can see what the OEM actually has.
                    val relatedFields = mutableListOf<String>()
                    var dumpCls: Class<*>? = viewRootImpl.javaClass
                    while (dumpCls != null) {
                        for (field in dumpCls!!.declaredFields) {
                            val n = field.name.lowercase()
                            if (n.contains("surface") || n.contains("blast") ||
                                n.contains("buffer") || n.contains("mbbq") ||
                                n == "msc" || n == "mwrl") {
                                relatedFields.add(
                                    "${dumpCls.simpleName}.${field.name}:${field.type.simpleName}"
                                )
                            }
                        }
                        dumpCls = dumpCls.superclass
                    }
                    Log.i("lsfg-vk-loop", "installSkipScreenshot: related fields dump: $relatedFields")
                    error("getSurfaceControl() missing and no SurfaceControl field found in class hierarchy")
                }

                // Verify the native handle is non-zero — it is 0 if relayoutWindow()
                // hasn't run yet. The onSurfaceTextureAvailable retry guarantees it is set.
                val nativeHandle = runCatching {
                    sc.javaClass.getDeclaredField("mNativeObject")
                        .also { it.isAccessible = true }.getLong(sc)
                }.getOrDefault(-1L)
                Log.i("lsfg-vk-loop", "installSkipScreenshot: SurfaceControl nativeHandle=0x${nativeHandle.toString(16)}")
                if (nativeHandle == 0L) {
                    error("SurfaceControl native handle is 0 — relayoutWindow not yet run; onSurfaceTextureAvailable will retry")
                }

                // setSkipScreenshot (@hide) — layer skipped during VirtualDisplay composition
                // (MediaProjection) but rendered normally on physical display.
                val txn = SurfaceControl.Transaction()
                val method = txn.javaClass.getMethod(
                    "setSkipScreenshot",
                    SurfaceControl::class.java,
                    Boolean::class.javaPrimitiveType,
                )
                method.invoke(txn, sc, true)
                txn.apply()
                Log.i("lsfg-vk-loop", "installSkipScreenshot: OK — overlay excluded from MediaProjection")
            }.onFailure {
                Log.w("lsfg-vk-loop", "installSkipScreenshot FAILED (${it.javaClass.simpleName}): ${it.message}" +
                           " — feedback loop may occur")
            }
        }

        if (host.isAttachedToWindow) {
            apply()
        } else {
            host.addOnAttachStateChangeListener(object : View.OnAttachStateChangeListener {
                override fun onViewAttachedToWindow(v: View) { apply() }
                override fun onViewDetachedFromWindow(v: View) = Unit
            })
        }
    }

    private fun applySkipScreenshotToSurfaceControl(sc: SurfaceControl?, label: String) {
        if (sc == null || !sc.isValid) return
        runCatching {
            val txn = SurfaceControl.Transaction()
            val method = txn.javaClass.getMethod(
                "setSkipScreenshot",
                SurfaceControl::class.java,
                Boolean::class.javaPrimitiveType,
            )
            method.invoke(txn, sc, true)
            txn.apply()
            Log.i("lsfg-vk-loop", "installSkipScreenshot($label): OK")
        }.onFailure {
            Log.w("lsfg-vk-loop", "installSkipScreenshot($label) FAILED: ${it.message}")
        }
    }

    companion object {
        private const val TAG = "OverlayManager"
    }
}
