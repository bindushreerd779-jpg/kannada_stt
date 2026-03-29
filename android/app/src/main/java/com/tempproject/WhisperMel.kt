package com.tempproject

import android.util.Log
import kotlin.math.*
import java.io.File

object WhisperMel {

    private const val SAMPLE_RATE = 16000
    private const val N_FFT = 400
    private const val HOP_LENGTH = 160
    private const val N_MELS = 80
    private const val MAX_FRAMES = 3000

    fun process(path: String): FloatArray {

        val audio = loadWav(path)

        val mel = FloatArray(N_MELS * MAX_FRAMES)

        val frames = (audio.size - N_FFT) / HOP_LENGTH

        for (i in 0 until min(frames, MAX_FRAMES)) {

            val start = i * HOP_LENGTH
            val window = FloatArray(N_FFT)

            for (j in 0 until N_FFT) {
                window[j] = audio[start + j] * hann(j, N_FFT)
            }

            val power = fftPower(window)

            for (m in 0 until N_MELS) {
                val value = power[m] + 1e-10f

                // 🔥 IMPORTANT: log10 NOT ln
                val log = (ln(value.toDouble()) / ln(10.0)).toFloat()

                mel[m * MAX_FRAMES + i] = log
            }
        }

        return mel
    }

    private fun hann(i: Int, n: Int): Float {
        return (0.5 - 0.5 * cos(2 * Math.PI * i / n)).toFloat()
    }

    private fun fftPower(x: FloatArray): FloatArray {
        val out = FloatArray(N_MELS)

        for (i in out.indices) {
            var sum = 0f
            for (j in x.indices) {
                sum += x[j] * x[j]
            }
            out[i] = sum
        }

        return out
    }

    private fun loadWav(path: String): FloatArray {
    val file = File(path)
    val bytes = file.readBytes()

    // skip WAV header (44 bytes)
    val audioBytes = bytes.copyOfRange(44, bytes.size)

    val samples = FloatArray(audioBytes.size / 2)

    var i = 0
    var j = 0
    while (i < audioBytes.size) {
        val low = audioBytes[i].toInt() and 0xFF
        val high = audioBytes[i + 1].toInt()
        val sample = (high shl 8) or low

        samples[j] = sample / 32768.0f

        i += 2
        j++
    }

    return samples
}
}