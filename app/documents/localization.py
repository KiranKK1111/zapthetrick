"""Localization — Phase 7 of the Document Generation roadmap (real minimal).

A full machine-translation + cultural-adaptation pipeline is large and needs a
translation model; this ships the two REAL, deterministic pieces that make a
document localizable without one:

  * ``localization_directive(lang)`` — a generation directive (like the Phase-7
    persona directive) that instructs the model to WRITE the document in the
    target language with locale-appropriate conventions (date format, number
    format, measurement system, register, text direction). This is what the
    generation path prepends to translate the *content*.

  * ``localize_labels(lang)`` — translated document FURNITURE labels ("Table of
    Contents", "Glossary", "Appendix", "Table"/"Figure", "List of Figures &
    Tables"). This is deterministic and is wired straight into the Phase-4
    ``structure.enrich`` pass, so the generated document's navigation chrome is
    localized at render time even with no LLM.

Fail-open + additive: an unknown/blank language → English defaults (today's
output, unchanged). No network, no model — pure data + string building.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Locale:
    code: str
    name: str            # English name
    native: str          # endonym
    rtl: bool = False
    date_format: str = "MMMM D, YYYY"
    decimal: str = "."
    measurement: str = "metric"
    labels: dict = None  # furniture label overrides (None → English)


# English furniture labels — the default and the key set every locale overrides.
_EN_LABELS = {
    "toc": "Table of Contents",
    "glossary": "Glossary",
    "appendix": "Appendix",
    "table": "Table",
    "figure": "Figure",
    "exhibits": "List of Figures & Tables",
}


_LOCALES: dict[str, Locale] = {
    "en": Locale("en", "English", "English",
                 date_format="MMMM D, YYYY", measurement="imperial",
                 labels=dict(_EN_LABELS)),
    "es": Locale("es", "Spanish", "Español", date_format="D 'de' MMMM 'de' YYYY",
                 decimal=",", labels={
                     "toc": "Índice", "glossary": "Glosario",
                     "appendix": "Apéndice", "table": "Tabla", "figure": "Figura",
                     "exhibits": "Lista de figuras y tablas"}),
    "fr": Locale("fr", "French", "Français", date_format="D MMMM YYYY",
                 decimal=",", labels={
                     "toc": "Table des matières", "glossary": "Glossaire",
                     "appendix": "Annexe", "table": "Tableau", "figure": "Figure",
                     "exhibits": "Liste des figures et tableaux"}),
    "de": Locale("de", "German", "Deutsch", date_format="D. MMMM YYYY",
                 decimal=",", labels={
                     "toc": "Inhaltsverzeichnis", "glossary": "Glossar",
                     "appendix": "Anhang", "table": "Tabelle", "figure": "Abbildung",
                     "exhibits": "Abbildungs- und Tabellenverzeichnis"}),
    "pt": Locale("pt", "Portuguese", "Português", date_format="D 'de' MMMM 'de' YYYY",
                 decimal=",", labels={
                     "toc": "Sumário", "glossary": "Glossário",
                     "appendix": "Apêndice", "table": "Tabela", "figure": "Figura",
                     "exhibits": "Lista de figuras e tabelas"}),
    "it": Locale("it", "Italian", "Italiano", date_format="D MMMM YYYY",
                 decimal=",", labels={
                     "toc": "Indice", "glossary": "Glossario",
                     "appendix": "Appendice", "table": "Tabella", "figure": "Figura",
                     "exhibits": "Elenco di figure e tabelle"}),
    "hi": Locale("hi", "Hindi", "हिन्दी", date_format="D MMMM YYYY", labels={
        "toc": "विषय-सूची", "glossary": "शब्दावली", "appendix": "परिशिष्ट",
        "table": "तालिका", "figure": "चित्र",
        "exhibits": "चित्र और तालिका सूची"}),
    "ar": Locale("ar", "Arabic", "العربية", rtl=True, date_format="D MMMM YYYY",
                 labels={
                     "toc": "جدول المحتويات", "glossary": "مسرد المصطلحات",
                     "appendix": "الملحق", "table": "جدول", "figure": "شكل",
                     "exhibits": "قائمة الأشكال والجداول"}),
    "ja": Locale("ja", "Japanese", "日本語", date_format="YYYY年M月D日", labels={
        "toc": "目次", "glossary": "用語集", "appendix": "付録",
        "table": "表", "figure": "図", "exhibits": "図表一覧"}),
    "zh": Locale("zh", "Chinese", "中文", date_format="YYYY年M月D日", labels={
        "toc": "目录", "glossary": "术语表", "appendix": "附录",
        "table": "表", "figure": "图", "exhibits": "图表清单"}),
}

# Endonyms + English names → code, for tolerant normalization.
_ALIASES: dict[str, str] = {}
for _c, _loc in _LOCALES.items():
    _ALIASES[_c] = _c
    _ALIASES[_loc.name.lower()] = _c
    _ALIASES[_loc.native.lower()] = _c
_ALIASES.update({"english": "en", "castellano": "es", "deutsch": "de",
                 "mandarin": "zh", "brazilian": "pt", "portuguese (brazil)": "pt"})


def normalize_language(value) -> str | None:
    """Canonical 2-letter code for a language name/code/endonym, or None when it
    isn't a supported/known language. ``"French"``/``"fr"``/``"français"`` → ``fr``;
    English or blank → ``None`` (no localization needed — the default path)."""
    if not value:
        return None
    v = str(value).strip().lower()
    code = _ALIASES.get(v) or _ALIASES.get(v[:2])
    if code is None or code == "en":
        return None
    return code


def is_supported(value) -> bool:
    return normalize_language(value) is not None


def localize_labels(language) -> dict:
    """Furniture labels for a language (toc/glossary/appendix/table/figure/
    exhibits). English (or unknown) → the English defaults."""
    code = normalize_language(language)
    if code is None:
        return dict(_EN_LABELS)
    loc = _LOCALES.get(code)
    if loc is None or not loc.labels:
        return dict(_EN_LABELS)
    merged = dict(_EN_LABELS)
    merged.update(loc.labels)
    return merged


def is_rtl(language) -> bool:
    code = normalize_language(language)
    loc = _LOCALES.get(code) if code else None
    return bool(loc and loc.rtl)


def localization_directive(target_language, source_language: str = "en") -> str:
    """A generation directive that tells the model to author the document in the
    target language with locale-appropriate conventions. Empty string when the
    target is English/blank/unknown (no directive → today's behavior). This is
    prepended to the generation prompt the way ``persona_directive`` is."""
    code = normalize_language(target_language)
    if code is None:
        return ""
    loc = _LOCALES[code]
    parts = [
        f"Write the ENTIRE document in {loc.name} ({loc.native}). Translate all "
        "prose, headings, and captions — do not leave English text. Keep code, "
        "commands, identifiers, URLs, and proper product names unchanged."]
    conv = [f"dates as {loc.date_format}",
            f"'{loc.decimal}' as the decimal separator",
            f"{loc.measurement} units"]
    parts.append("Use locale conventions: " + ", ".join(conv) + ".")
    if loc.rtl:
        parts.append("This is a right-to-left script — order content accordingly.")
    parts.append("Use a natural, culturally appropriate register for this "
                 "language rather than a word-for-word translation.")
    return " ".join(parts)


def language_name(value) -> str:
    code = normalize_language(value)
    return _LOCALES[code].name if code else "English"


def supported_languages() -> list[dict]:
    return [{"code": c, "name": loc.name, "native": loc.native, "rtl": loc.rtl}
            for c, loc in _LOCALES.items()]


__all__ = [
    "Locale", "normalize_language", "is_supported", "is_rtl", "localize_labels",
    "localization_directive", "language_name", "supported_languages",
]
