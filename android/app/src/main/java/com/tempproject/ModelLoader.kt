package com.tempproject

import android.content.Context
import java.io.File
import java.io.FileOutputStream

object ModelLoader {

    fun copyModel(context: Context, modelName: String): String {
        val file = File(context.filesDir, modelName)

        if (!file.exists()) {
            context.assets.open(modelName).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
        }

        return file.absolutePath
    }
}