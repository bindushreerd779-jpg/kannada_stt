package com.tempproject

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.ByteArrayOutputStream

class WhisperTokenizer(context: Context) {

    private val idToToken = HashMap<Int, String>()
    private val byteDecoder = HashMap<Char, Int>()

    init {
        val vocabText = context.assets.open("vocab.json")
            .bufferedReader()
            .use { it.readText() }

        val vocabJson = JSONObject(vocabText)
        val keys = vocabJson.keys()
        while (keys.hasNext()) {
            val token = keys.next()
            val id = vocabJson.getInt(token)
            idToToken[id] = token
        }

        try {
            val addedText = context.assets.open("added_tokens.json")
                .bufferedReader()
                .use { it.readText() }

            val addedJson = JSONObject(addedText)
            val addedKeys = addedJson.keys()
            while (addedKeys.hasNext()) {
                val token = addedKeys.next()
                val id = addedJson.getInt(token)
                idToToken[id] = token
            }
        } catch (_: Exception) {
            // optional
        }

        buildByteDecoder()
    }

    private fun buildByteDecoder() {
        val bs = mutableListOf<Int>()
        for (i in 33..126) bs.add(i)
        for (i in 161..172) bs.add(i)
        for (i in 174..255) bs.add(i)

        val cs = bs.toMutableList()
        var n = 0
        for (b in 0..255) {
            if (!bs.contains(b)) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }

        for (i in bs.indices) {
            byteDecoder[cs[i].toChar()] = bs[i]
        }
    }

    fun decode(tokenIds: List<Int>): String {
        val byteStream = ByteArrayOutputStream()

        for (id in tokenIds) {
            val token = idToToken[id] ?: continue

            // skip special tokens
            if (token.startsWith("<|") && token.endsWith("|>")) continue

            // Whisper/GPT byte-level BPE:
            // Ġ means leading space
            val piece = token.replace('Ġ', ' ')

            for (ch in piece) {
                val b = byteDecoder[ch]
                if (b != null) {
                    byteStream.write(b)
                } else {
                    // write plain ASCII chars like normal space directly
                    if (ch.code in 0..127) {
                        byteStream.write(ch.code)
                    } else {
                        Log.d("WHISPER_TOKENIZER", "Skipping unknown char in token piece: '$ch' from token '$token'")
                    }
                }
            }
        }

        val rawBytes = byteStream.toByteArray()
        val decoded = rawBytes.toString(Charsets.UTF_8)

        Log.d("WHISPER_TOKENIZER", "Decoded text raw = $decoded")

        return decoded
            .replace(Regex("<\\|.*?\\|>"), "")
            .replace("�", "")
            .trim()
    }
}