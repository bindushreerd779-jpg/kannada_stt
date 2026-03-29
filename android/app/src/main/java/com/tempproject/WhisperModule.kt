package com.tempproject

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.content.ContextCompat
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin

class WhisperModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    override fun getName(): String = "WhisperModule"

    private var encoderModule: Module? = null
    private var decoderModule: Module? = null
    private val tokenizer = WhisperTokenizer(reactContext)

    companion object {
        private const val TAG = "WHISPER"

        private const val SAMPLE_RATE = 16000
        private const val N_FFT = 400
        private const val HOP_LENGTH = 160
        private const val N_MELS = 80
        private const val CHUNK_LENGTH = 30
        private const val MAX_SAMPLES = SAMPLE_RATE * CHUNK_LENGTH
        private const val MEL_FRAMES = 3000

        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT

        private const val TOKEN_SOT = 50258
        private const val TOKEN_KN = 50306
        private const val TOKEN_TRANSCRIBE = 50359
        private const val TOKEN_NOTIMESTAMPS = 50363
        private const val TOKEN_EOT = 50257
        private const val MAX_DECODE_LEN = 24
    }

    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null

    @Volatile
    private var isRecording = false
    private var recordingPath: String? = null

    private val hannWindowCache by lazy { hannWindowWhisper(N_FFT) }
    private val melFilterBankCache by lazy { buildWhisperMelFilterBank(SAMPLE_RATE, N_FFT, N_MELS) }

    private fun getHannWindow(): FloatArray = hannWindowCache
    private fun getMelFilterBank(): Array<FloatArray> = melFilterBankCache

    private fun assetToFile(assetName: String): String {
        val file = File(reactContext.filesDir, assetName)
        if (!file.exists()) {
            reactContext.assets.open(assetName).use { input ->
                file.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        return file.absolutePath
    }

    private fun loadModels() {
        if (encoderModule == null || decoderModule == null) {
            System.loadLibrary("pytorch_jni")

            val encPath = assetToFile("encoder.pt")
            val decPath = assetToFile("decoder.pt")

            encoderModule = Module.load(encPath)
            decoderModule = Module.load(decPath)

            Log.d(TAG, "✅ Models loaded")
        }
    }

    @ReactMethod
    fun startRecording(promise: Promise) {
        try {
            if (isRecording) {
                promise.reject("RECORDING", "Already recording")
                return
            }

            val hasPermission = ContextCompat.checkSelfPermission(
                reactContext,
                Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED

            if (!hasPermission) {
                promise.reject("PERMISSION", "RECORD_AUDIO permission not granted")
                return
            }

            val minBuffer = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT
            )

            if (minBuffer == AudioRecord.ERROR || minBuffer == AudioRecord.ERROR_BAD_VALUE) {
                promise.reject("AUDIO", "Invalid AudioRecord buffer size")
                return
            }

            val bufferSize = max(minBuffer, 4096)

            val recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )

            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                recorder.release()
                promise.reject("AUDIO", "AudioRecord initialization failed")
                return
            }

            val outFile = File(reactContext.cacheDir, "recording_native.wav")
            if (outFile.exists()) outFile.delete()

            FileOutputStream(outFile).use { fos ->
                writeWavHeader(fos, 0, SAMPLE_RATE, 1, 16)
            }

            audioRecord = recorder
            recordingPath = outFile.absolutePath
            isRecording = true

            recorder.startRecording()

            recordingThread = Thread {
                writePcmToWavFile(recorder, outFile, bufferSize)
            }.apply { start() }

            Log.d(TAG, "Native recording started: ${outFile.absolutePath}")
            promise.resolve(outFile.absolutePath)
        } catch (e: Exception) {
            Log.e(TAG, "startRecording error", e)
            promise.reject("START_RECORDING_ERROR", e.message ?: "Unknown error")
        }
    }

    @ReactMethod
    fun stopRecording(promise: Promise) {
        try {
            if (!isRecording) {
                promise.reject("RECORDING", "Not recording")
                return
            }

            isRecording = false

            try {
                recordingThread?.join(3000)
            } catch (_: Exception) {
            }

            try {
                audioRecord?.stop()
            } catch (_: Exception) {
            }

            try {
                audioRecord?.release()
            } catch (_: Exception) {
            }

            audioRecord = null
            recordingThread = null

            val path = recordingPath
            if (path == null) {
                promise.reject("STOP_RECORDING_ERROR", "Recording path is null")
                return
            }

            Log.d(TAG, "Native recording stopped: $path")
            promise.resolve(path)
        } catch (e: Exception) {
            Log.e(TAG, "stopRecording error", e)
            promise.reject("STOP_RECORDING_ERROR", e.message ?: "Unknown error")
        }
    }

    private fun writePcmToWavFile(recorder: AudioRecord, outFile: File, bufferSize: Int) {
        var totalAudioLen = 0L
        val buffer = ByteArray(bufferSize)

        try {
            FileOutputStream(outFile, true).use { fos ->
                while (isRecording) {
                    val read = recorder.read(buffer, 0, buffer.size)
                    if (read > 0) {
                        fos.write(buffer, 0, read)
                        totalAudioLen += read.toLong()
                    }
                }
                fos.flush()
            }

            updateWavHeader(outFile, totalAudioLen, SAMPLE_RATE, 1, 16)
            Log.d(TAG, "WAV finalized: ${outFile.absolutePath}, pcmBytes=$totalAudioLen")
        } catch (e: Exception) {
            Log.e(TAG, "writePcmToWavFile error", e)
        }
    }

    private fun writeWavHeader(
        fos: FileOutputStream,
        totalAudioLen: Long,
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int
    ) {
        val totalDataLen = totalAudioLen + 36
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8

        val header = ByteArray(44)

        header[0] = 'R'.code.toByte()
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()

        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = ((totalDataLen shr 8) and 0xff).toByte()
        header[6] = ((totalDataLen shr 16) and 0xff).toByte()
        header[7] = ((totalDataLen shr 24) and 0xff).toByte()

        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()

        header[12] = 'f'.code.toByte()
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()

        header[16] = 16
        header[17] = 0
        header[18] = 0
        header[19] = 0

        header[20] = 1
        header[21] = 0

        header[22] = channels.toByte()
        header[23] = 0

        header[24] = (sampleRate and 0xff).toByte()
        header[25] = ((sampleRate shr 8) and 0xff).toByte()
        header[26] = ((sampleRate shr 16) and 0xff).toByte()
        header[27] = ((sampleRate shr 24) and 0xff).toByte()

        header[28] = (byteRate and 0xff).toByte()
        header[29] = ((byteRate shr 8) and 0xff).toByte()
        header[30] = ((byteRate shr 16) and 0xff).toByte()
        header[31] = ((byteRate shr 24) and 0xff).toByte()

        header[32] = blockAlign.toByte()
        header[33] = 0

        header[34] = bitsPerSample.toByte()
        header[35] = 0

        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()

        header[40] = (totalAudioLen and 0xff).toByte()
        header[41] = ((totalAudioLen shr 8) and 0xff).toByte()
        header[42] = ((totalAudioLen shr 16) and 0xff).toByte()
        header[43] = ((totalAudioLen shr 24) and 0xff).toByte()

        fos.write(header, 0, 44)
    }

    private fun updateWavHeader(
        outFile: File,
        totalAudioLen: Long,
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int
    ) {
        val totalDataLen = totalAudioLen + 36
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8

        RandomAccessFile(outFile, "rw").use { raf ->
            raf.seek(0)
            val header = ByteArray(44)

            header[0] = 'R'.code.toByte()
            header[1] = 'I'.code.toByte()
            header[2] = 'F'.code.toByte()
            header[3] = 'F'.code.toByte()

            header[4] = (totalDataLen and 0xff).toByte()
            header[5] = ((totalDataLen shr 8) and 0xff).toByte()
            header[6] = ((totalDataLen shr 16) and 0xff).toByte()
            header[7] = ((totalDataLen shr 24) and 0xff).toByte()

            header[8] = 'W'.code.toByte()
            header[9] = 'A'.code.toByte()
            header[10] = 'V'.code.toByte()
            header[11] = 'E'.code.toByte()

            header[12] = 'f'.code.toByte()
            header[13] = 'm'.code.toByte()
            header[14] = 't'.code.toByte()
            header[15] = ' '.code.toByte()

            header[16] = 16
            header[17] = 0
            header[18] = 0
            header[19] = 0

            header[20] = 1
            header[21] = 0

            header[22] = channels.toByte()
            header[23] = 0

            header[24] = (sampleRate and 0xff).toByte()
            header[25] = ((sampleRate shr 8) and 0xff).toByte()
            header[26] = ((sampleRate shr 16) and 0xff).toByte()
            header[27] = ((sampleRate shr 24) and 0xff).toByte()

            header[28] = (byteRate and 0xff).toByte()
            header[29] = ((byteRate shr 8) and 0xff).toByte()
            header[30] = ((byteRate shr 16) and 0xff).toByte()
            header[31] = ((byteRate shr 24) and 0xff).toByte()

            header[32] = blockAlign.toByte()
            header[33] = 0

            header[34] = bitsPerSample.toByte()
            header[35] = 0

            header[36] = 'd'.code.toByte()
            header[37] = 'a'.code.toByte()
            header[38] = 't'.code.toByte()
            header[39] = 'a'.code.toByte()

            header[40] = (totalAudioLen and 0xff).toByte()
            header[41] = ((totalAudioLen shr 8) and 0xff).toByte()
            header[42] = ((totalAudioLen shr 16) and 0xff).toByte()
            header[43] = ((totalAudioLen shr 24) and 0xff).toByte()

            raf.write(header)
        }
    }

    @ReactMethod
    fun transcribe(audioPath: String, promise: Promise) {
        try {
            Log.d(TAG, "transcribe() started")
            Log.d(TAG, "audioPath = $audioPath")

            loadModels()

            val mel = extractLogMelFromWav(audioPath)

            val melTensor = Tensor.fromBlob(
                mel,
                longArrayOf(1, N_MELS.toLong(), MEL_FRAMES.toLong())
            )

            val encoderOutput = encoderModule!!.forward(
                IValue.from(melTensor)
            ).toTensor()

            val tokens = mutableListOf(
                TOKEN_SOT,
                TOKEN_KN,
                TOKEN_TRANSCRIBE,
                TOKEN_NOTIMESTAMPS
            )

            for (step in 0 until MAX_DECODE_LEN) {
                val tokenTensor = Tensor.fromBlob(
                    LongArray(tokens.size) { tokens[it].toLong() },
                    longArrayOf(1, tokens.size.toLong())
                )

                val output = decoderModule!!.forward(
                    IValue.from(tokenTensor),
                    IValue.from(encoderOutput)
                ).toTensor()

                val logits = output.dataAsFloatArray
                val shape = output.shape()
                val seqLen = shape[1].toInt()
                val vocabSize = shape[2].toInt()
                val offset = (seqLen - 1) * vocabSize

                var nextToken = 0
                var maxVal = Float.NEGATIVE_INFINITY

                for (i in 0 until vocabSize) {
                    val v = logits[offset + i]
                    if (v > maxVal) {
                        maxVal = v
                        nextToken = i
                    }
                }

                if (nextToken == TOKEN_EOT) break
                tokens.add(nextToken)
            }

            val text = tokenizer.decode(tokens)
            Log.d(TAG, "TEXT: $text")
            promise.resolve(text)

        } catch (e: Exception) {
            Log.e(TAG, "Error", e)
            promise.reject("ERROR", e.message ?: "Unknown error")
        }
    }

    private fun extractLogMelFromWav(path: String): FloatArray {
        val audio = loadWavMono(path)

        val effectiveAudio = if (audio.size > MAX_SAMPLES) {
            audio.copyOfRange(0, MAX_SAMPLES)
        } else {
            audio
        }

        val spec = stftMagnitudeSquaredWhisper(effectiveAudio, N_FFT, HOP_LENGTH)
        val melFilter = getMelFilterBank()
        val frames = spec[0].size

        Log.d(TAG, "frames = $frames, expected <= $MEL_FRAMES")

        val out = FloatArray(N_MELS * MEL_FRAMES)
        val mel2d = Array(N_MELS) { FloatArray(frames) }

        var globalMax = Float.NEGATIVE_INFINITY

        for (m in 0 until N_MELS) {
            for (t in 0 until frames) {
                var sum = 0.0f
                for (f in spec.indices) {
                    sum += melFilter[m][f] * spec[f][t]
                }
                val v = log10(max(1e-10f, sum))
                mel2d[m][t] = v
                if (v > globalMax) globalMax = v
            }
        }

        val floorVal = globalMax - 8.0f

        for (m in 0 until N_MELS) {
            val base = m * MEL_FRAMES
            val limit = min(frames, MEL_FRAMES)

            for (t in 0 until limit) {
                var v = mel2d[m][t]
                if (v < floorVal) v = floorVal
                out[base + t] = (v + 4.0f) / 4.0f
            }

            for (t in limit until MEL_FRAMES) {
                out[base + t] = 0.0f
            }
        }

        return out
    }

    private fun loadWavMono(path: String): FloatArray {
        val file = File(path)
        if (!file.exists()) {
            throw IllegalArgumentException("Audio file does not exist: $path")
        }

        val bytes = file.readBytes()
        if (bytes.size < 44) {
            throw IllegalArgumentException("Invalid WAV: file too small (${bytes.size} bytes)")
        }

        val riff = String(bytes.copyOfRange(0, 4), Charsets.US_ASCII)
        val wave = String(bytes.copyOfRange(8, 12), Charsets.US_ASCII)

        Log.d(TAG, "Header[0..15] = ${
            bytes.take(16).joinToString(" ") { String.format("%02X", it) }
        }")
        Log.d(TAG, "RIFF = $riff, WAVE = $wave")

        if (riff != "RIFF" || wave != "WAVE") {
            throw IllegalArgumentException("Not a WAV file")
        }

        val fmtOffset = findChunk(bytes, "fmt ")
        if (fmtOffset < 0) throw IllegalArgumentException("fmt chunk not found")

        val audioFormat = readLE16(bytes, fmtOffset + 8)
        val channels = readLE16(bytes, fmtOffset + 10)
        val sampleRate = readLE32(bytes, fmtOffset + 12)
        val bitsPerSample = readLE16(bytes, fmtOffset + 22)

        Log.d(TAG, "audioFormat=$audioFormat channels=$channels sampleRate=$sampleRate bitsPerSample=$bitsPerSample")

        if (audioFormat != 1) throw IllegalArgumentException("Expected PCM WAV, got format=$audioFormat")
        if (channels != 1) throw IllegalArgumentException("Expected mono WAV, got $channels channels")
        if (bitsPerSample != 16) throw IllegalArgumentException("Expected 16-bit PCM WAV, got $bitsPerSample-bit")

        val dataOffset = findChunk(bytes, "data")
        if (dataOffset < 0) throw IllegalArgumentException("data chunk not found")

        val dataSize = readLE32(bytes, dataOffset + 4)
        val pcmStart = dataOffset + 8

        if (pcmStart + dataSize > bytes.size) {
            throw IllegalArgumentException("Invalid WAV data size")
        }

        val sampleCount = dataSize / 2
        val output = FloatArray(sampleCount)

        var j = pcmStart
        for (i in 0 until sampleCount) {
            val lo = bytes[j].toInt() and 0xff
            val hi = bytes[j + 1].toInt()
            val sample = (hi shl 8) or lo
            output[i] = sample.toShort() / 32768.0f
            j += 2
        }

        return if (sampleRate == SAMPLE_RATE) output else resampleTo16k(output, sampleRate)
    }

    private fun findChunk(bytes: ByteArray, target: String): Int {
        var i = 12
        while (i + 8 <= bytes.size) {
            val chunkId = String(bytes.copyOfRange(i, i + 4), Charsets.US_ASCII)
            val chunkSize = readLE32(bytes, i + 4)
            if (chunkId == target) return i
            i += 8 + chunkSize
            if (i % 2 == 1) i++
        }
        return -1
    }

    private fun readLE16(bytes: ByteArray, offset: Int): Int {
        return (bytes[offset].toInt() and 0xff) or
                ((bytes[offset + 1].toInt() and 0xff) shl 8)
    }

    private fun readLE32(bytes: ByteArray, offset: Int): Int {
        return (bytes[offset].toInt() and 0xff) or
                ((bytes[offset + 1].toInt() and 0xff) shl 8) or
                ((bytes[offset + 2].toInt() and 0xff) shl 16) or
                ((bytes[offset + 3].toInt() and 0xff) shl 24)
    }

    private fun resampleTo16k(input: FloatArray, originalRate: Int): FloatArray {
        if (originalRate == SAMPLE_RATE) return input

        val ratio = SAMPLE_RATE.toDouble() / originalRate.toDouble()
        val outputLength = max(1, (input.size * ratio).toInt())
        val output = FloatArray(outputLength)

        for (i in 0 until outputLength) {
            val srcIndex = i / ratio
            val left = srcIndex.toInt()
            val right = min(left + 1, input.size - 1)
            val frac = (srcIndex - left).toFloat()
            output[i] = input[left] * (1 - frac) + input[right] * frac
        }

        return output
    }

    private fun stftMagnitudeSquaredWhisper(audio: FloatArray, nFft: Int, hopLength: Int): Array<FloatArray> {
        val padded = reflectPad(audio, nFft / 2)
        val window = getHannWindow()
        val nFrames = 1 + (padded.size - nFft) / hopLength
        val nFreq = nFft / 2 + 1
        val out = Array(nFreq) { FloatArray(nFrames) }

        for (frame in 0 until nFrames) {
            val start = frame * hopLength

            for (k in 0 until nFreq) {
                var real = 0.0
                var imag = 0.0
                val coeff = 2.0 * PI * k / nFft

                for (n in 0 until nFft) {
                    val x = padded[start + n] * window[n]
                    val angle = coeff * n
                    real += x * cos(angle)
                    imag -= x * sin(angle)
                }

                out[k][frame] = (real * real + imag * imag).toFloat()
            }
        }

        return out
    }

    private fun reflectPad(input: FloatArray, pad: Int): FloatArray {
        val out = FloatArray(input.size + 2 * pad)

        for (i in 0 until pad) {
            val src = min(pad - i, input.size - 1)
            out[i] = input[src]
        }

        System.arraycopy(input, 0, out, pad, input.size)

        for (i in 0 until pad) {
            val src = max(input.size - 2 - i, 0)
            out[pad + input.size + i] = input[src]
        }

        return out
    }

    private fun hannWindowWhisper(n: Int): FloatArray {
        val w = FloatArray(n)
        for (i in 0 until n) {
            w[i] = (0.5 - 0.5 * cos(2.0 * PI * i / (n - 1))).toFloat()
        }
        return w
    }

    private fun hzToMel(hz: Double): Double {
        return 2595.0 * log10(1.0 + hz / 700.0)
    }

    private fun melToHz(mel: Double): Double {
        return 700.0 * (10.0.pow(mel / 2595.0) - 1.0)
    }

    private fun buildWhisperMelFilterBank(sampleRate: Int, nFft: Int, nMels: Int): Array<FloatArray> {
        val nFreq = nFft / 2 + 1
        val fMin = 0.0
        val fMax = sampleRate / 2.0

        val melMin = hzToMel(fMin)
        val melMax = hzToMel(fMax)

        val melPoints = DoubleArray(nMels + 2)
        for (i in melPoints.indices) {
            melPoints[i] = melMin + (melMax - melMin) * i / (nMels + 1)
        }

        val hzPoints = DoubleArray(nMels + 2)
        for (i in hzPoints.indices) {
            hzPoints[i] = melToHz(melPoints[i])
        }

        val bins = IntArray(nMels + 2)
        for (i in bins.indices) {
            bins[i] = floor((nFft + 1) * hzPoints[i] / sampleRate).toInt()
        }

        val fb = Array(nMels) { FloatArray(nFreq) }

        for (m in 1..nMels) {
            val left = bins[m - 1]
            val center = bins[m]
            val right = bins[m + 1]

            if (center > left) {
                for (k in left until center) {
                    if (k in 0 until nFreq) {
                        fb[m - 1][k] = (k - left).toFloat() / (center - left).toFloat()
                    }
                }
            }

            if (right > center) {
                for (k in center until right) {
                    if (k in 0 until nFreq) {
                        fb[m - 1][k] = (right - k).toFloat() / (right - center).toFloat()
                    }
                }
            }
        }

        return fb
    }
}