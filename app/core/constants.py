from __future__ import annotations

ALLOWED_AUDIO_EXTENSIONS: set[str] = {".wav", ".mp3", ".m4a"}
ALLOWED_AUDIO_MIME_TYPES: set[str] = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/aac",
    "video/mp4",
}

SUPPORTED_LANGUAGES: set[str] = {"en", "hi"}
LANGUAGE_ALIASES: dict[str, str] = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "hi": "hi",
    "hin": "hi",
    "hindi": "hi",
    "हिंदी": "hi",
}

LANGUAGE_PACKS: dict[str, dict[str, object]] = {
    "en": {
        "filler_words": ("you know", "basically", "actually", "um", "uh", "like"),
        "positive_words": {
            "accurate",
            "confident",
            "clear",
            "excellent",
            "good",
            "great",
            "organized",
            "positive",
            "strong",
            "smooth",
            "stable",
            "well",
        },
        "negative_words": {
            "bad",
            "confusing",
            "difficult",
            "error",
            "fail",
            "hesitant",
            "messy",
            "negative",
            "poor",
            "rough",
            "slow",
            "unclear",
            "weak",
        },
        "insights": {
            "strong_overall": "Strong overall communication score",
            "clear_delivery": "Clear speech delivery",
            "confident_delivery": "Confident vocal delivery",
            "positive_tone": "Positive conversational tone",
            "frequent_fillers": "Frequent filler words",
            "reduce_fillers": "Reduce filler word usage",
            "extended_pauses": "Extended speech pauses",
            "shorten_pauses": "Shorten long pauses between phrases",
            "background_noise": "Noticeable background noise",
            "quieter_environment": "Record in a quieter environment",
            "pace_outside": "Pace is outside the ideal range",
            "adjust_pace": "Adjust speaking pace toward the target range",
            "fluency_improve": "Fluency can be improved",
            "smoother_transitions": "Practice smoother sentence transitions",
            "confidence_weak": "Confidence signals are weak",
            "increase_projection": "Increase vocal projection and reduce hesitation",
            "negative_tone": "Negative tone detected",
            "clearer_positive_language": "Shift toward clearer and more positive language",
            "consistent_audio": "Consistent audio capture",
        },
    },
    "hi": {
        "filler_words": ("मतलब", "तो", "वो", "यानी", "असल में", "जी", "हां", "अच्छा"),
        "positive_words": {
            "अच्छा",
            "उत्कृष्ट",
            "साफ",
            "सकारात्मक",
            "मजबूत",
            "आत्मविश्वासी",
            "संतुलित",
            "स्पष्ट",
            "सहज",
            "बेहतर",
        },
        "negative_words": {
            "कमजोर",
            "खराब",
            "धीमा",
            "अस्पष्ट",
            "गलत",
            "मुश्किल",
            "नकारात्मक",
            "उलझन",
            "घबराया",
            "अव्यवस्थित",
        },
        "insights": {
            "strong_overall": "कुल संचार स्कोर मजबूत है",
            "clear_delivery": "भाषण स्पष्ट है",
            "confident_delivery": "आवाज़ में आत्मविश्वास है",
            "positive_tone": "बातचीत का स्वर सकारात्मक है",
            "frequent_fillers": "फिलर शब्द बार-बार आ रहे हैं",
            "reduce_fillers": "फिलर शब्दों का उपयोग कम करें",
            "extended_pauses": "बोलने के बीच लंबे विराम हैं",
            "shorten_pauses": "वाक्यों के बीच लंबे विराम कम करें",
            "background_noise": "पृष्ठभूमि में शोर महसूस हो रहा है",
            "quieter_environment": "कृपया शांत वातावरण में रिकॉर्ड करें",
            "pace_outside": "बोलने की गति आदर्श सीमा से बाहर है",
            "adjust_pace": "अपनी बोलने की गति को लक्ष्य सीमा के करीब रखें",
            "fluency_improve": "प्रवाह को और बेहतर किया जा सकता है",
            "smoother_transitions": "वाक्यों के बीच सहज बदलाव का अभ्यास करें",
            "confidence_weak": "आत्मविश्वास के संकेत कमजोर हैं",
            "increase_projection": "आवाज़ की स्पष्टता और आत्मविश्वास बढ़ाएं",
            "negative_tone": "नकारात्मक स्वर पाया गया",
            "clearer_positive_language": "और स्पष्ट तथा सकारात्मक भाषा का उपयोग करें",
            "consistent_audio": "ऑडियो कैप्चर स्थिर है",
        },
    },
}


def normalize_language_code(language: str | None) -> str | None:
    if language is None:
        return None

    normalized = language.strip().lower()
    return LANGUAGE_ALIASES.get(normalized)


def get_language_pack(language: str | None) -> dict[str, object]:
    normalized = normalize_language_code(language) or "en"
    return LANGUAGE_PACKS.get(normalized, LANGUAGE_PACKS["en"])
