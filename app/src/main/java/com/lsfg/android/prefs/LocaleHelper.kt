package com.lsfg.android.prefs

import android.content.Context
import android.content.res.Configuration
import android.os.LocaleList
import java.util.Locale

enum class AppLanguage(val tag: String) {
    SYSTEM("system"),
    ENGLISH("en"),
    TRADITIONAL_CHINESE("zh-TW");

    companion object {
        fun fromTag(tag: String?): AppLanguage =
            values().firstOrNull { it.tag == tag } ?: SYSTEM
    }
}

object LocaleHelper {
    private const val PREFS_NAME = "lsfg_prefs"
    private const val KEY_APP_LANGUAGE = "app_language"

    fun getSavedLanguage(context: Context): AppLanguage {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return AppLanguage.fromTag(prefs.getString(KEY_APP_LANGUAGE, AppLanguage.SYSTEM.tag))
    }

    fun setSavedLanguage(context: Context, language: AppLanguage) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_APP_LANGUAGE, language.tag)
            .commit()
    }

    fun wrapContext(base: Context): Context {
        val lang = getSavedLanguage(base)
        if (lang == AppLanguage.SYSTEM) return base

        val locale = when (lang) {
            AppLanguage.ENGLISH -> Locale.ENGLISH
            AppLanguage.TRADITIONAL_CHINESE -> Locale.TRADITIONAL_CHINESE
            AppLanguage.SYSTEM -> return base
        }
        Locale.setDefault(locale)
        val config = Configuration(base.resources.configuration)
        config.setLocales(LocaleList(locale))
        config.setLayoutDirection(locale)
        return base.createConfigurationContext(config)
    }
}
