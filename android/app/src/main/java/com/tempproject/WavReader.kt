package com.tempproject

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

object WavReader {

    fun read(path: String): FloatArray {

        val bytes = File(path).readBytes()

        val buffer = ByteBuffer.wrap(bytes)
        buffer.order(ByteOrder.LITTLE_ENDIAN)

        val samples = FloatArray((bytes.size - 44) / 2)

        var i = 44
        var j = 0

        while (i < bytes.size) {

            val sample = buffer.getShort(i)

            samples[j] = sample / 32768f

            i += 2
            j++
        }

        return samples
    }
}