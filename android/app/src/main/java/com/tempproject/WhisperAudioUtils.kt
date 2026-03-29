package com.tempproject

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

object WhisperAudioUtils {

    private const val SAMPLE_RATE = 16000
    private const val N_FFT = 400
    private const val HOP_LENGTH = 160
    private const val N_MELS = 80
    private const val CHUNK_LENGTH = 30
    private const val MAX_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH
    private const val MEL_FMIN = 0.0
    private const val MEL_FMAX = 8000.0

    fun decodeWavToFloatArray(wavBytes: ByteArray): FloatArray {
        if (wavBytes.size < 44) {
            throw IllegalArgumentException("Invalid WAV file")
        }

        val dataStart = 44
        val pcmBytes = wavBytes.copyOfRange(dataStart, wavBytes.size)

        val sampleCount = pcmBytes.size / 2
        val samples = FloatArray(MAX_SAMPLES)

        val usable = minOf(sampleCount, MAX_SAMPLES)

        var i = 0
        while (i < usable) {
            val lo = pcmBytes[i * 2].toInt() and 0xFF
            val hi = pcmBytes[i * 2 + 1].toInt()
            val sample = (hi shl 8) or lo
            samples[i] = (sample.toShort() / 32768.0f)
            i++
        }

        return samples
    }

    fun logMelSpectrogram(audio: FloatArray): FloatArray {
        val window = hannWindow(N_FFT)
        val melFilterBank = buildMelFilterBank()
        val nFrames = 3000

        val melSpec = FloatArray(N_MELS * nFrames)

        for (frame in 0 until nFrames) {
            val start = frame * HOP_LENGTH
            val fftInput = FloatArray(N_FFT)

            for (i in 0 until N_FFT) {
                val idx = start + i
                val sample = if (idx < audio.size) audio[idx] else 0f
                fftInput[i] = sample * window[i]
            }

            val powerSpec = powerSpectrum(fftInput)

            for (m in 0 until N_MELS) {
                var sum = 0f
                for (k in powerSpec.indices) {
                    sum += melFilterBank[m][k] * powerSpec[k]
                }
                melSpec[m * nFrames + frame] = kotlin.math.max(1e-10f, sum)
            }
        }

        for (i in melSpec.indices) {
            melSpec[i] = ln(melSpec[i]) / ln(10.0f)
        }

        val maxVal = melSpec.maxOrNull() ?: 0f
        for (i in melSpec.indices) {
            melSpec[i] = kotlin.math.max(melSpec[i], maxVal - 8.0f)
        }
        for (i in melSpec.indices) {
            melSpec[i] = (melSpec[i] + 4.0f) / 4.0f
        }

        return melSpec
    }

    private fun hannWindow(size: Int): FloatArray {
        val w = FloatArray(size)
        for (i in 0 until size) {
            w[i] = (0.5f - 0.5f * cos((2.0 * PI * i) / size).toFloat())
        }
        return w
    }

    private fun powerSpectrum(frame: FloatArray): FloatArray {
        val nFreq = N_FFT / 2 + 1
        val out = FloatArray(nFreq)

        for (k in 0 until nFreq) {
            var real = 0.0
            var imag = 0.0

            for (n in frame.indices) {
                val angle = 2.0 * PI * k * n / N_FFT
                real += frame[n] * cos(angle)
                imag -= frame[n] * sin(angle)
            }

            out[k] = ((real * real + imag * imag) / N_FFT).toFloat()
        }

        return out
    }

    private fun hzToMel(hz: Double): Double {
        return 2595.0 * kotlin.math.log10(1.0 + hz / 700.0)
    }

    private fun melToHz(mel: Double): Double {
        return 700.0 * (10.0.pow(mel / 2595.0) - 1.0)
    }

    private fun buildMelFilterBank(): Array<FloatArray> {
        val nFreq = N_FFT / 2 + 1
        val filters = Array(N_MELS) { FloatArray(nFreq) }

        val melMin = hzToMel(MEL_FMIN)
        val melMax = hzToMel(MEL_FMAX)

        val melPoints = DoubleArray(N_MELS + 2)
        for (i in melPoints.indices) {
            melPoints[i] = melMin + (melMax - melMin) * i / (N_MELS + 1)
        }

        val hzPoints = melPoints.map { melToHz(it) }
        val bins = hzPoints.map { ((N_FFT + 1) * it / SAMPLE_RATE).toInt() }

        for (m in 1..N_MELS) {
            val left = bins[m - 1]
            val center = bins[m]
            val right = bins[m + 1]

            for (k in left until center) {
                if (k in 0 until nFreq) {
                    filters[m - 1][k] = (k - left).toFloat() / (center - left).toFloat()
                }
            }

            for (k in center until right) {
                if (k in 0 until nFreq) {
                    filters[m - 1][k] = (right - k).toFloat() / (right - center).toFloat()
                }
            }
        }

        return filters
    }
}