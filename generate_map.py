"""
Генератор интерактивной HTML-карты КРТ-анализа.

Читает профильный GeoPackage, загружает имена критериев из выбранного листа
Критерии.xlsx и создаёт/обновляет results/krt_map.html. Карта хранит несколько
профилей оценки (например, «Город» и «Инвестор») в отдельных вкладках.

Запуск:
    python generate_map.py
"""
from __future__ import annotations
import os
import base64
import json
import pathlib
import re
import sys
import urllib.request

import geopandas as gpd
import numpy as np
from dotenv import load_dotenv


# ── Конфигурация ──────────────────────────────────────────────────────────────

load_dotenv(pathlib.Path(__file__).with_name(".env"))
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "").strip()

MAPBOX_TILES_URL = (
    "https://api.mapbox.com/styles/v1/okamidan/"
    "cmqupgwnh000b01s41y4r8ams/tiles/256/{z}/{x}/{y}"
    f"?access_token={MAPBOX_TOKEN}"
)
MAPBOX_ATTRIBUTION = (
    '© <a href="https://www.mapbox.com/about/maps/" target="_blank">Mapbox</a> '
    '© <a href="http://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>'
)
OSM_ATTRIBUTION = (
    '© <a href="http://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors'
)

DEFAULT_RESULTS_GPKG = pathlib.Path("results/krt_scores.gpkg")
XLSX_PATH      = "Критерии.xlsx"
OUTPUT_HTML    = pathlib.Path("results/krt_map.html")
KRT_PLOTS_SHP  = pathlib.Path("input_data/Площадки КРТ/Площадки КРТ.shp")
DEFAULT_CENTER = (51.95, 113.5)   # Чита, WGS-84

# ── Шрифты ────────────────────────────────────────────────────────────────────
# Положите файлы шрифтов (.woff2 / .woff / .ttf / .otf) в папку fonts/ и
# перечислите их здесь. Они будут «запечены» в HTML как base64 @font-face,
# поэтому карта работает офлайн и не зависит от установленных в системе шрифтов.
# family должен совпадать с font-family в CSS ниже.
FONTS_DIR   = pathlib.Path("fonts")
FONT_FACES  = [
    {"family": "Atlas Grotesk LC Regular", "file": "AtlasGroteskLC-Regular.woff2",
     "weight": 400, "style": "normal"},
    {"family": "FugueMonoKB",              "file": "FugueMonoKB.woff2",
     "weight": 700, "style": "normal"},
]

_FONT_MIME = {
    ".woff2": "font/woff2",
    ".woff":  "font/woff",
    ".ttf":   "font/ttf",
    ".otf":   "font/otf",
}
_FONT_FORMAT = {
    ".woff2": "woff2",
    ".woff":  "woff",
    ".ttf":   "truetype",
    ".otf":   "opentype",
}

_LEAFLET_VERSION = "1.9.4"
_LEAFLET_CSS_URL = f"https://unpkg.com/leaflet@{_LEAFLET_VERSION}/dist/leaflet.css"
_LEAFLET_JS_URL  = f"https://unpkg.com/leaflet@{_LEAFLET_VERSION}/dist/leaflet.js"
_LEAFLET_CACHE   = pathlib.Path(__file__).parent / ".leaflet_cache"

# ─────────────────────────────────────────────────────────────────────────────


def _fetch_leaflet() -> str:
    """Возвращает HTML-фрагмент с Leaflet CSS+JS (встроенный или CDN-ссылки).

    При первом запуске скачивает файлы и кеширует рядом со скриптом.
    Последующие запуски используют кеш без обращения к сети.
    Если скачать не удалось и кеша нет — возвращает CDN-ссылки как запасной вариант.
    """
    css_cache = _LEAFLET_CACHE / "leaflet.css"
    js_cache  = _LEAFLET_CACHE / "leaflet.js"

    if css_cache.exists() and js_cache.exists():
        print("  Leaflet: используется локальный кеш")
        css = css_cache.read_text(encoding="utf-8")
        js  = js_cache.read_text(encoding="utf-8")
        return f"<style>{css}</style>\n  <script>{js}</script>"

    print("  Leaflet: скачивание с CDN …")
    try:
        css = urllib.request.urlopen(_LEAFLET_CSS_URL, timeout=20).read().decode("utf-8")
        js  = urllib.request.urlopen(_LEAFLET_JS_URL,  timeout=20).read().decode("utf-8")
        _LEAFLET_CACHE.mkdir(exist_ok=True)
        css_cache.write_text(css, encoding="utf-8")
        js_cache.write_text(js,  encoding="utf-8")
        print("  Leaflet: скачан и закеширован")
        return f"<style>{css}</style>\n  <script>{js}</script>"
    except Exception as exc:
        print(f"  [warn] Leaflet не скачался ({exc}), используются CDN-ссылки")
        return (
            f'<link rel="stylesheet" href="{_LEAFLET_CSS_URL}">\n'
            f'  <script src="{_LEAFLET_JS_URL}"></script>'
        )


def build_fonts_css() -> str:
    """Возвращает <style> с @font-face, где шрифты встроены как base64 data-URI.

    Читает файлы из FONTS_DIR по списку FONT_FACES. Отсутствующие файлы
    пропускаются с предупреждением — карта тогда использует системный фолбэк.
    """
    faces: list[str] = []
    for face in FONT_FACES:
        path = FONTS_DIR / face["file"]
        ext  = path.suffix.lower()
        if not path.exists():
            print(f"  [предупреждение] шрифт не найден: {path} — используется системный фолбэк")
            continue
        if ext not in _FONT_MIME:
            print(f"  [предупреждение] неизвестный формат шрифта: {path.name} — пропущен")
            continue
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        data_uri = f"data:{_FONT_MIME[ext]};base64,{b64}"
        faces.append(
            "    @font-face {\n"
            f"      font-family: '{face['family']}';\n"
            f"      font-weight: {face.get('weight', 400)};\n"
            f"      font-style: {face.get('style', 'normal')};\n"
            "      font-display: swap;\n"
            f"      src: url('{data_uri}') format('{_FONT_FORMAT[ext]}');\n"
            "    }"
        )
        print(f"  Шрифт встроен: {face['family']} ({path.name}, {len(b64)//1024} КБ base64)")

    if not faces:
        return ""
    return "<style>\n" + "\n".join(faces) + "\n  </style>"


def load_criterion_metadata(
    xlsx_path: str = XLSX_PATH,
    sheet_name: str | None = None,
) -> dict[str, dict[str, str]]:
    """Возвращает подписи и направления критериев из выбранного листа."""
    try:
        import crt_pipeline as P
        resolved_sheet = P.normalize_criteria_sheet(sheet_name)
        criteria = P.load_criteria(xlsx_path, sheet_name=resolved_sheet)
        return {
            f"crit_{c.cid}_score": {
                "label": f"Критерий {c.cid}: {c.name}",
                "direction": c.direction,
                "direction_label": "Обратная" if c.direction == "inverse" else "Прямая",
            }
            for c in criteria
        }
    except Exception as exc:
        print(f"  [предупреждение] Не удалось загрузить метаданные критериев: {exc}")
        return {}


def load_criterion_names(
    xlsx_path: str = XLSX_PATH,
    sheet_name: str | None = None,
) -> dict[str, str]:
    """Обратная совместимость: возвращает только подписи критериев."""
    metadata = load_criterion_metadata(xlsx_path, sheet_name)
    return {key: value["label"] for key, value in metadata.items()}


def enrich_profile_layers_metadata(
    profiles: dict[str, dict],
    xlsx_path: str = XLSX_PATH,
) -> None:
    """Синхронизирует сохранённые HTML-профили с актуальными листами Excel.

    Помимо обновления подписей и направления шкалы функция:
    - удаляет из панели слоёв устаревшие ``crit_*_score``, которых больше нет
      в соответствующем листе критериев;
    - восстанавливает порядок критериев по Excel;
    - удаляет осиротевшие поля критериев из GeoJSON-профиля.

    Благодаря этому пересборка любого профиля не оставляет в соседней вкладке
    старые плашки вроде ``crit_14_score`` после перенумерации таблицы.
    """
    criterion_key_re = re.compile(r"^crit_\d+_score$")

    for profile_key, profile in profiles.items():
        sheet_name = profile.get("sheet_name")
        layers = profile.get("layers")
        if not sheet_name or not isinstance(layers, list):
            continue

        metadata = load_criterion_metadata(xlsx_path, sheet_name)
        if not metadata:
            continue

        allowed_keys = set(metadata)
        existing_by_key = {
            str(layer.get("key")): dict(layer)
            for layer in layers
            if isinstance(layer, dict) and layer.get("key") is not None
        }
        stale_keys = [
            key for key in existing_by_key
            if criterion_key_re.fullmatch(key) and key not in allowed_keys
        ]
        if stale_keys:
            print(
                f"  [предупреждение] Профиль {profile_key}: "
                f"удалены устаревшие критерии из HTML: {stale_keys}"
            )

        ordered_layers: list[dict] = []
        for key, layer_meta in metadata.items():
            layer = existing_by_key.get(key)
            if layer is None:
                continue
            layer.update(layer_meta)
            ordered_layers.append(layer)
        profile["layers"] = ordered_layers

        features = (profile.get("geojson") or {}).get("features") or []
        for feature in features:
            properties = feature.get("properties") if isinstance(feature, dict) else None
            if not isinstance(properties, dict):
                continue
            for key in list(properties):
                if criterion_key_re.fullmatch(str(key)) and key not in allowed_keys:
                    properties.pop(key, None)


def load_geodata(
    results_gpkg: pathlib.Path,
    expected_score_cols: list[str] | None = None,
) -> tuple[list[dict], list[str]]:
    """Читает GPKG и возвращает только критерии, разрешённые текущим Excel.

    ``expected_score_cols`` задаётся метаданными выбранного листа. Лишние
    колонки в старом GPKG игнорируются, отсутствующие явно выводятся в лог.
    """
    if not results_gpkg.exists():
        sys.exit(
            f"Ошибка: файл {results_gpkg} не найден.\n"
            "Сначала запустите run_pipeline.py, чтобы получить результаты."
        )

    print(f"  Загрузка {results_gpkg} …")
    gdf = gpd.read_file(results_gpkg)

    # Для больших сеток упрощаем геометрию перед сериализацией в GeoJSON:
    # это резко снижает объём данных и ускоряет отрисовку на карте.
    if len(gdf) > 5000:
        print("  Упрощение геометрии гексов для быстрой отрисовки …")
        gdf = gdf.to_crs(32649)
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(40.0, preserve_topology=True)
        gdf = gdf.to_crs(4326)
    else:
        gdf = gdf.to_crs(4326)

    all_score_cols = sorted(
        [c for c in gdf.columns if c.startswith("crit_") and c.endswith("_score")],
        key=lambda c: int(c.split("_")[1]),
    )
    if not all_score_cols:
        sys.exit("Ошибка: в файле результатов нет колонок crit_*_score.")

    if expected_score_cols is None:
        score_cols = all_score_cols
    else:
        available = set(all_score_cols)
        expected = list(dict.fromkeys(expected_score_cols))
        stale = [c for c in all_score_cols if c not in set(expected)]
        missing = [c for c in expected if c not in available]
        score_cols = [c for c in expected if c in available]

        if stale:
            print(
                "  [предупреждение] В GPKG найдены устаревшие критерии, "
                f"которые отсутствуют в Excel и не попадут в HTML: {stale}"
            )
        if missing:
            print(
                "  [предупреждение] В Excel есть критерии без рассчитанной "
                f"колонки в GPKG: {missing}"
            )

    if not score_cols:
        sys.exit(
            "Ошибка: ни одна колонка crit_*_score из GPKG не соответствует "
            "выбранному листу критериев."
        )

    print(f"  Найдено актуальных критериев: {score_cols}")
    print(f"  Ячеек в сетке:     {len(gdf)}")

    for col in score_cols:
        gdf[col] = gdf[col].round(3)

    keep = [c for c in ["hex_id", "geometry"] + score_cols if c in gdf.columns]
    gdf = gdf[keep]

    features = json.loads(gdf.to_json())["features"]
    return features, score_cols


def estimate_center(features: list[dict]) -> tuple[float, float]:
    lats, lons = [], []
    for f in features[:300]:
        ring = f["geometry"]["coordinates"][0]
        lon, lat = ring[0]
        lats.append(lat)
        lons.append(lon)
    if not lats:
        return DEFAULT_CENTER
    return float(np.mean(lats)), float(np.mean(lons))


def load_krt_plots() -> str:
    """Читает шейп-файл площадок КРТ, конвертирует в WGS-84, возвращает GeoJSON-строку."""
    if not KRT_PLOTS_SHP.exists():
        print(f"  [предупреждение] {KRT_PLOTS_SHP} не найден — слой площадок пропущен")
        return '{"type":"FeatureCollection","features":[]}'
    gdf = gpd.read_file(KRT_PLOTS_SHP).to_crs(4326)
    print(f"  Площадок КРТ: {len(gdf)}")
    return gdf.to_json(ensure_ascii=False)


_PROFILE_STORE_RE = re.compile(
    r'<script\s+id="krt-profile-store"\s+type="application/json">(.*?)</script>',
    re.DOTALL,
)


def _extract_js_json_assignment(html_text: str, variable: str):
    """Извлекает JSON-литерал из старого HTML вида ``const NAME = {...};``."""
    match = re.search(rf"const\s+{re.escape(variable)}\s*=\s*", html_text)
    if not match:
        return None
    tail = html_text[match.end():].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(tail)
        return value
    except json.JSONDecodeError:
        return None


def load_existing_profiles(output_html: pathlib.Path = OUTPUT_HTML) -> dict[str, dict]:
    """Читает профили из существующего HTML и поддерживает миграцию старой карты.

    В новой версии данные лежат в ``#krt-profile-store``. Если найден HTML старого
    формата с одиночными константами GEOJSON/LAYERS, он импортируется как «Город».
    """
    if not output_html.exists():
        return {}

    try:
        html_text = output_html.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"  [предупреждение] Не удалось прочитать существующий HTML: {exc}")
        return {}

    store_match = _PROFILE_STORE_RE.search(html_text)
    if store_match:
        try:
            data = json.loads(store_match.group(1).strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            print(f"  [предупреждение] Профили из HTML не прочитаны: {exc}")

    # Миграция HTML, созданного предыдущей версией generate_map.py.
    geojson = _extract_js_json_assignment(html_text, "GEOJSON")
    layers = _extract_js_json_assignment(html_text, "LAYERS")
    if isinstance(geojson, dict) and isinstance(layers, list):
        print("  Найден HTML старого формата — профиль импортирован как «Город»")
        return {
            "city": {
                "label": "Город",
                "sheet_name": "7.2 Критерии (город)",
                "geojson": geojson,
                "layers": layers,
            }
        }
    return {}


def render_html(
    profiles: dict[str, dict],
    active_profile: str,
    center: tuple[float, float],
    leaflet_head: str,
    krt_plots_json: str = '{"type":"FeatureCollection","features":[]}',
    fonts_css: str = "",
) -> str:
    profiles_json = json.dumps(profiles, ensure_ascii=False).replace("</", "<\\/")
    active_profile_json = json.dumps(active_profile, ensure_ascii=False)

    return (
        _HTML_TEMPLATE
        .replace("__LEAFLET_HEAD__",  leaflet_head)
        .replace("__FONTS_CSS__",     fonts_css)
        .replace("__MAPBOX_URL__",    MAPBOX_TILES_URL)
        .replace("__HAS_MAPBOX__",    "true" if MAPBOX_TOKEN else "false")
        .replace("__MAPBOX_ATTR__",   MAPBOX_ATTRIBUTION)
        .replace("__OSM_ATTR__",      OSM_ATTRIBUTION)
        .replace("__PROFILE_STORE_JSON__", profiles_json)
        .replace("__ACTIVE_PROFILE__", active_profile_json)
        .replace("__CENTER_LAT__",    str(center[0]))
        .replace("__CENTER_LON__",    str(center[1]))
        .replace("__KRT_GEOJSON__",   krt_plots_json)
    )


# ── HTML-шаблон ───────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>КРТ Чита — Аналитическая карта</title>
  __LEAFLET_HEAD__
  __FONTS_CSS__
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      position: relative; height: 100vh;
      font-family: 'Atlas Grotesk LC Regular', 'Segoe UI', Arial, sans-serif; font-size: 13px;
      background: #E9EDED; color: #1F2023;
      overflow: hidden;
    }

    /* ── Боковая панель — плавающее «жидкое стекло» ──────────────── */
    #sidebar {
      position: absolute; z-index: 1000;
      top: 16px; left: 16px; bottom: 16px;
      width: 310px; min-width: 260px; max-width: 360px;
      display: flex; flex-direction: column;
      border-radius: 20px;
      overflow: hidden;
      /* Полупрозрачная стеклянная подложка + блик по верхнему краю */
      background:
        linear-gradient(180deg, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0.12) 22%, rgba(233,237,237,0.10) 100%),
        rgba(233, 237, 237, 0.42);
      border: 1px solid rgba(255, 255, 255, 0.55);
      backdrop-filter: blur(28px) saturate(180%);
      -webkit-backdrop-filter: blur(28px) saturate(180%);
      box-shadow:
        0 18px 50px rgba(31, 32, 35, 0.22),
        0 2px 8px rgba(31, 32, 35, 0.10),
        inset 0 1px 0 rgba(255, 255, 255, 0.65),
        inset 0 -1px 0 rgba(255, 255, 255, 0.18);
    }

    #sidebar-header {
      padding: 16px 18px 12px;
      background: transparent;
      border-bottom: 1px solid rgba(255, 255, 255, 0.35);
      flex-shrink: 0;
    }
    #sidebar-header h1 {
      font-family: 'FugueMonoKB', 'Courier New', monospace;
      font-size: 15px; font-weight: 700;
      color: #028BA8; letter-spacing: 0.2px;
      text-transform: uppercase;
    }
    #sidebar-header p {
      font-size: 11px; color: #5f6466; margin-top: 3px;
    }

    /* ── Профили оценки: город / инвестор / совместный ─────────── */
    #profile-tabs {
      display: flex; gap: 6px;
      padding: 9px 16px 7px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.35);
      flex-shrink: 0;
    }
    .profile-tab {
      flex: 1; min-width: 0;
      padding: 6px 5px;
      border: 1px solid rgba(198, 208, 208, 0.75);
      border-radius: 7px;
      background: rgba(248, 252, 252, 0.55);
      color: #5f6466;
      font-size: 10px; font-weight: 600; white-space: nowrap;
      cursor: pointer;
      transition: border-color 0.15s, color 0.15s, background 0.15s;
    }
    .profile-tab[data-profile="combined"] { flex-grow: 1.55; }
    .profile-tab:hover { border-color: #028BA8; color: #028BA8; }
    .profile-tab.active {
      background: rgba(246, 251, 251, 0.90);
      border-color: #028BA8;
      color: #028BA8;
    }

    #section-label {
      padding: 9px 16px 4px;
      font-size: 10px; font-weight: 600;
      letter-spacing: 1px; text-transform: uppercase;
      color: #5f6466; flex-shrink: 0;
    }

    /* ── Список слоёв ───────────────────────────────────────────── */
    #layers-scroll { flex: 1; overflow-y: auto; }
    #layers-scroll::-webkit-scrollbar { width: 4px;}
    #layers-scroll::-webkit-scrollbar-track { background: #E9EDED; }
    #layers-scroll::-webkit-scrollbar-thumb { background: #028BA8; border-radius: 2px; }

    .layer-item {
      padding: 7px 18px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.28);
      transition: background 0.12s;
      background: transparent;
    }
    .layer-item:hover { background: rgba(255, 255, 255, 0.30); }
    .layer-item.active {
      background: rgba(255, 255, 255, 0.42);
      border-left: 2px solid #028BA8;
      padding-left: 16px;
    }
    .layer-item.active .layer-label { color: #028BA8; }

    .layer-item-header {
      display: flex; align-items: flex-start; gap: 8px; cursor: pointer;
    }
    .layer-item-header input[type="radio"] {
      accent-color: #028BA8; cursor: pointer; margin-top: 2px; flex-shrink: 0;
    }
    .layer-label {
      flex: 1; min-width: 0;
      display: flex; align-items: flex-start; gap: 6px;
      font-size: 12px; line-height: 1.45; color: #1F2023;
    }
    .layer-label-text { flex: 1; min-width: 0; }
    .scale-chip {
      display: inline-block; flex-shrink: 0;
      margin-top: 1px; padding: 1px 7px;
      border-radius: 999px;
      font-size: 9px; line-height: 1.45; font-weight: 600;
      white-space: nowrap;
    }
    .scale-chip.direct {
      color: #27828C;
      background: rgba(39, 130, 140, 0.11);
      border: 1px solid rgba(39, 130, 140, 0.48);
    }
    .scale-chip.inverse {
      color: #BF6B63;
      background: rgba(191, 107, 99, 0.11);
      border: 1px solid rgba(191, 107, 99, 0.48);
    }

    .integral-chip {
      display: inline-block; margin-top: 4px; margin-left: 20px;
      font-size: 10px; color: #028BA8;
      background: rgba(2,139,168,0.10);
      border: 1px solid rgba(2,139,168,0.20);
      border-radius: 3px; padding: 1px 6px;
    }

    .weight-row {
      display: flex; align-items: center;
      margin-top: 5px; padding-left: 20px; gap: 6px;
    }
    .weight-label { font-size: 11px; color: #5f6466; white-space: nowrap; }
    .weight-input {
      width: 54px; padding: 3px 6px;
      background: #f8fcfc; border: 1px solid #c6d0d0;
      border-radius: 4px; color: #1F2023; font-size: 12px; text-align: center;
      -moz-appearance: textfield;
    }
    .weight-input::-webkit-inner-spin-button,
    .weight-input::-webkit-outer-spin-button { -webkit-appearance: none; }
    .weight-input:focus { outline: none; border-color: #028BA8; }

    /* ── Легенда ────────────────────────────────────────────────── */
    #legend-panel {
      padding: 10px 18px 13px;
      background: transparent;
      border-top: 1px solid rgba(255, 255, 255, 0.35);
      flex-shrink: 0;
    }
    #legend-title {
      font-size: 10px; color: #5f6466; margin-bottom: 5px;
    }
    #legend-bar {
      height: 9px;
      border-radius: 5px;
      background: linear-gradient(
        to right,
        #FFB400 0%,
        #F9B000 10%,
        #FCBF39 20%,
        #FECE69 30%,
        #FFDD94 40%,
        #DBD8AB 50%,
        #B0D2BF 60%,
        #77C8D2 70%,
        #4EBDC3 80%,
        #00B2B5 90%,
        #00A7A7 100%
      );
    }
    #legend-labels {
      display: flex; justify-content: space-between;
      font-size: 10px; color: #5f6466; margin-top: 3px;
    }
    #no-data-badge {
      display: inline-flex; align-items: center; gap: 4px;
      font-size: 10px; color: #5f6466; margin-top: 5px;
    }
    #no-data-badge span.swatch {
      display: inline-block; width: 12px; height: 9px;
      background: #c6d0d0; border-radius: 2px;
    }


    /* ── Переключатель подложки ─────────────────────────────────── */
    #basemap-panel {
      padding: 8px 18px 12px;
      background: transparent;
      border-top: 1px solid rgba(255, 255, 255, 0.35);
      flex-shrink: 0;
    }
    #basemap-label {
      font-size: 10px; font-weight: 600;
      letter-spacing: 1px; text-transform: uppercase;
      color: #5f6466; margin-bottom: 5px;
    }
    #basemap-btns { display: flex; gap: 6px; }
    .bm-btn {
      flex: 1; padding: 4px 0;
      background: rgba(248, 252, 252, 0.72); border: 1px solid rgba(198, 208, 208, 0.75);
      border-radius: 4px; color: #5f6466;
      font-size: 11px; cursor: pointer;
      transition: border-color 0.15s, color 0.15s;
      backdrop-filter: blur(8px) saturate(140%);
      -webkit-backdrop-filter: blur(8px) saturate(140%);
    }
    .bm-btn:hover { border-color: #028BA8; color: #028BA8; }
    .bm-btn.active {
      background: #f6fbfb; border-color: #028BA8;
      color: #028BA8; font-weight: 600;
    }

    /* ── Карта ──────────────────────────────────────────────────── */
    #map { position: absolute; inset: 0; z-index: 0; }

    .leaflet-tooltip {
      background: rgba(248, 252, 252, 0.78) !important;
      border: 1px solid rgba(198, 208, 208, 0.7) !important;
      border-radius: 10px !important;
      color: #1F2023 !important;
      padding: 8px 10px !important;
      font-size: 12px !important;
      backdrop-filter: blur(10px) saturate(140%) !important;
      -webkit-backdrop-filter: blur(10px) saturate(140%) !important;
      box-shadow: 0 8px 24px rgba(31, 32, 35, 0.16) !important;
    }
    .leaflet-tooltip::before { display: none !important; }

    .tt-title { font-weight: 600; color: #028BA8; margin-bottom: 4px; }
    .tt-integral { color: #028BA8; font-weight: 600; }
    .tt-row { display: flex; justify-content: space-between; gap: 12px; }
    .tt-val  { color: #1F2023; }
    .tt-dim  { color: #5f6466; font-size: 11px; }
  </style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <h1>КРТ Чита</h1>
    <p>Оценка интересов</p>
  </div>

  <div id="profile-tabs"></div>

  <div id="section-label">Слои (название/тип шкалы)</div>

  <div id="layers-scroll"></div>

  <div id="legend-panel">
    <div id="legend-title">Балл 0–10: низкий ↔ высокий</div>
    <div id="legend-bar"></div>
    <div id="legend-labels">
      <span>0</span><span>5</span><span>10</span>
    </div>
    <div id="no-data-badge">
      <span class="swatch"></span> нет данных
    </div>
  </div>

  <div id="basemap-panel">
    <div id="basemap-label">Подложка</div>
    <div id="basemap-btns">
      <button class="bm-btn active" id="bm-mapbox" onclick="setBasemap('mapbox')">Mapbox</button>
      <button class="bm-btn"        id="bm-osm"    onclick="setBasemap('osm')">OpenStreetMap</button>
    </div>
  </div>

</div><!-- /sidebar -->

<div id="map"></div>

<script id="krt-profile-store" type="application/json">__PROFILE_STORE_JSON__</script>
<script>
// ── Данные ────────────────────────────────────────────────────────────────
const PROFILES = JSON.parse(document.getElementById('krt-profile-store').textContent);
const KRT_PLOTS = __KRT_GEOJSON__;
let activeProfile = __ACTIVE_PROFILE__;
let GEOJSON = null;
let LAYERS = [];
const PROFILE_WEIGHTS = {};
const COMBINED_PROFILE = 'combined';
const COMBINED_LABEL = 'Общий';
const COMBINED_LAYER_KEYS = {
  result: '_combined_integral',
  city: '_city_integral',
  investor: '_investor_integral',
};
let COMBINED_GEOJSON = { type: 'FeatureCollection', features: [] };

// Показываем JS-ошибки прямо на карте для диагностики
window.onerror = function(msg, src, line) {
  var el = document.getElementById('map');
  if (el) el.innerHTML = '<div style="color:#fff;background:#b91c1c;padding:16px;font-size:13px">'
    + '<b>Ошибка JS:</b> ' + msg + ' (строка ' + line + ')</div>';
  return false;
};

// ── Инициализация карты ───────────────────────────────────────────────────
const map = L.map('map', { zoomControl: false, preferCanvas: true });
L.control.zoom({ position: 'topright' }).addTo(map);
map.attributionControl.setPrefix(false);

// Площадки КРТ всегда располагаются выше аналитических гексагонов.
map.createPane('krtPane');
map.getPane('krtPane').style.zIndex = 450;
map.getPane('krtPane').style.pointerEvents = 'none';

const krtRenderer = L.canvas({
  pane: 'krtPane',
  padding: 0.1,
});

const TILE_LAYERS = {
  mapbox: L.tileLayer('__MAPBOX_URL__', {
    attribution: '__MAPBOX_ATTR__',
    maxZoom: 20,
    tileSize: 256,
  }),
  osm: L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '__OSM_ATTR__',
    maxZoom: 19,
  }),
};

const HAS_MAPBOX = __HAS_MAPBOX__;
let activeBasemap = HAS_MAPBOX ? 'mapbox' : 'osm';
TILE_LAYERS[activeBasemap].addTo(map);
document.querySelectorAll('.bm-btn').forEach(function(b) { b.classList.remove('active'); });
document.getElementById('bm-' + activeBasemap).classList.add('active');
if (!HAS_MAPBOX) {
  const mapboxButton = document.getElementById('bm-mapbox');
  mapboxButton.disabled = true;
  mapboxButton.title = 'Задайте MAPBOX_TOKEN перед генерацией карты';
}

function setBasemap(key) {
  if (key === 'mapbox' && !HAS_MAPBOX) return;
  map.removeLayer(TILE_LAYERS[activeBasemap]);
  activeBasemap = key;
  TILE_LAYERS[key].addTo(map);
  document.querySelectorAll('.bm-btn').forEach(function(b) { b.classList.remove('active'); });
  document.getElementById('bm-' + key).classList.add('active');
}

// ── Цветовая шкала Артема ─────────────────────────────────────────────────
const COLOR_STOPS = [
  [0,  [255, 180, 0]],  // #FFB400
  [1,  [249, 176, 0]],  // #F9B000
  [2,  [252, 191, 57]],  // #FCBF39
  [3,  [254, 206, 105]],  // #FECE69  [4,  [255, 221, 148]],  // #FFDD94
  [5,  [219, 216, 171]],  // #DBD8AB
  [6,  [176, 210, 191]],  // #B0D2BF
  [7,  [119, 200, 210]],  // #77C8D2
  [8,  [78, 189, 195]],  // #4EBDC3
  [9,  [0, 178, 181]],  // #00B2B5
  [10, [ 0, 167, 167]],  // #00A7A7
];

function valToRgb(val) {
  if (val == null || isNaN(val)) return 'rgba(75,85,99,0.55)';
  const v = Math.max(0, Math.min(10, val));
  let i = 0;
  while (i < COLOR_STOPS.length - 2 && v > COLOR_STOPS[i + 1][0]) i++;
  const [v0, c0] = COLOR_STOPS[i];
  const [v1, c1] = COLOR_STOPS[i + 1];
  const t = (v - v0) / (v1 - v0);
  const lerp = (a, b) => Math.round(a + t * (b - a));
  return 'rgb(' + lerp(c0[0],c1[0]) + ',' + lerp(c0[1],c1[1]) + ',' + lerp(c0[2],c1[2]) + ')';
}

function scoreFillOpacity(val, highlighted) {
  // Нулевой балл — это валидное значение, а не NoData: оставляем контур и
  // tooltip, но полностью убираем заливку, в том числе при наведении.
  if (val != null && !isNaN(val) && Number(val) === 0) return 0;
  return highlighted ? 0.92 : 0.78;
}

// ── Профили, панель слоёв и веса ───────────────────────────────────────────
function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function hasCombinedProfile() {
  return Boolean(PROFILES.city && PROFILES.investor);
}

function profileLabel(key) {
  if (key === COMBINED_PROFILE) return COMBINED_LABEL;
  const profile = PROFILES[key] || {};
  return profile.label || key;
}

function orderedProfileKeys() {
  const priority = { city: 0, investor: 1, combined: 2 };
  const keys = Object.keys(PROFILES).filter(function(key) {
    return key !== COMBINED_PROFILE;
  });
  if (hasCombinedProfile()) keys.push(COMBINED_PROFILE);
  return keys.sort(function(a, b) {
    const pa = Object.prototype.hasOwnProperty.call(priority, a) ? priority[a] : 10;
    const pb = Object.prototype.hasOwnProperty.call(priority, b) ? priority[b] : 10;
    if (pa !== pb) return pa - pb;
    return String(profileLabel(a)).localeCompare(String(profileLabel(b)), 'ru');
  });
}

function renderProfileTabs() {
  const holder = document.getElementById('profile-tabs');
  const keys = orderedProfileKeys();
  holder.innerHTML = keys.map(function(key) {
    const activeClass = key === activeProfile ? ' active' : '';
    return '<button class="profile-tab' + activeClass + '" data-profile="'
      + escapeHtml(key) + '">' + escapeHtml(profileLabel(key)) + '</button>';
  }).join('');

  holder.querySelectorAll('.profile-tab').forEach(function(btn) {
    btn.addEventListener('click', function() {
      selectProfile(this.dataset.profile);
    });
  });
}

function profileWeights(profileKey) {
  if (!PROFILE_WEIGHTS[profileKey]) PROFILE_WEIGHTS[profileKey] = {};
  return PROFILE_WEIGHTS[profileKey];
}

function clampWeight(value) {
  const v = parseFloat(value);
  return (isNaN(v) || v < 1) ? 1 : v > 10 ? 10 : v;
}

function getCriterionWeight(profileKey, key) {
  const weights = profileWeights(profileKey);
  if (weights[key] == null) weights[key] = 5;
  return weights[key];
}

function getCombinedWeight(sourceKey) {
  return getCriterionWeight(COMBINED_PROFILE, sourceKey);
}

function directionChipHtml(layer) {
  const direction = layer && layer.direction;
  if (direction !== 'direct' && direction !== 'inverse') return '';
  const label = layer.direction_label || (direction === 'inverse' ? 'Обратная' : 'Прямая');
  return '<span class="scale-chip ' + direction + '">' + escapeHtml(label) + '</span>';
}

function renderStandardLayersPanel(holder) {
  let html = '';

  html += '<div class="layer-item active" id="item-_integral">'
    + '<div class="layer-item-header">'
    + '<input type="radio" name="active-layer" value="_integral" id="radio-_integral" checked>'
    + '<label class="layer-label" for="radio-_integral" style="font-weight:600;color:#028BA8">'
    + '<span class="layer-label-text">Интегральный слой</span></label></div>'
    + '<div class="integral-chip">Σ(балл × вес) / Σвес → нормировка 0–10</div></div>';

  LAYERS.forEach(function(layer) {
    const key = String(layer.key);
    const weight = getCriterionWeight(activeProfile, key);
    html += '<div class="layer-item" id="item-' + escapeHtml(key) + '">'
      + '<div class="layer-item-header">'
      + '<input type="radio" name="active-layer" value="' + escapeHtml(key)
      + '" id="radio-' + escapeHtml(key) + '">'
      + '<label class="layer-label" for="radio-' + escapeHtml(key) + '">'
      + '<span class="layer-label-text">' + escapeHtml(layer.label || key) + '</span>'
      + directionChipHtml(layer) + '</label></div>'
      + '<div class="weight-row"><span class="weight-label">Вес (1–10):</span>'
      + '<input type="number" class="weight-input" id="weight-' + escapeHtml(key)
      + '" data-weight-scope="criterion" data-profile-key="' + escapeHtml(activeProfile)
      + '" data-layer-key="' + escapeHtml(key) + '" min="1" max="10" value="'
      + weight + '" step="1"></div></div>';
  });

  holder.innerHTML = html;
}

function renderCombinedLayersPanel(holder) {
  const cityWeight = getCombinedWeight('city');
  const investorWeight = getCombinedWeight('investor');
  let html = '';

  html += '<div class="layer-item active" id="item-' + COMBINED_LAYER_KEYS.result + '">'
    + '<div class="layer-item-header">'
    + '<input type="radio" name="active-layer" value="' + COMBINED_LAYER_KEYS.result
    + '" id="radio-' + COMBINED_LAYER_KEYS.result + '" checked>'
    + '<label class="layer-label" for="radio-' + COMBINED_LAYER_KEYS.result
    + '" style="font-weight:600;color:#028BA8">'
    + '<span class="layer-label-text">Результирующий интегральный слой</span></label></div>'
    + '<div class="integral-chip">взвешенное среднее → min–max 0–10</div></div>';

  [
    { key: 'city', layerKey: COMBINED_LAYER_KEYS.city, label: 'Интегральный слой города', weight: cityWeight },
    { key: 'investor', layerKey: COMBINED_LAYER_KEYS.investor, label: 'Интегральный слой инвестора', weight: investorWeight },
  ].forEach(function(source) {
    html += '<div class="layer-item" id="item-' + source.layerKey + '">'
      + '<div class="layer-item-header">'
      + '<input type="radio" name="active-layer" value="' + source.layerKey
      + '" id="radio-' + source.layerKey + '">'
      + '<label class="layer-label" for="radio-' + source.layerKey + '">'
      + '<span class="layer-label-text">' + source.label + '</span></label></div>'
      + '<div class="weight-row"><span class="weight-label">Вес (1–10):</span>'
      + '<input type="number" class="weight-input" id="combined-weight-' + source.key
      + '" data-weight-scope="combined" data-layer-key="' + source.key
      + '" min="1" max="10" value="' + source.weight + '" step="1"></div></div>';
  });

  holder.innerHTML = html;
}

function renderLayersPanel() {
  const holder = document.getElementById('layers-scroll');
  if (activeProfile === COMBINED_PROFILE) {
    renderCombinedLayersPanel(holder);
  } else {
    renderStandardLayersPanel(holder);
  }
  holder.querySelectorAll('input[name="active-layer"]').forEach(function(el) {
    el.addEventListener('change', function() { selectLayer(this.value); });
  });
  attachWeightHandlers();
}

function activateProfileData(key) {
  if (key === COMBINED_PROFILE) {
    if (!hasCombinedProfile()) return false;
    calcProfileIntegral('city');
    calcProfileIntegral('investor');
    calcCombinedIntegral();
    activeProfile = key;
    GEOJSON = COMBINED_GEOJSON;
    LAYERS = [];
    activeKey = COMBINED_LAYER_KEYS.result;
    return true;
  }

  const profile = PROFILES[key];
  if (!profile) return false;
  activeProfile = key;
  GEOJSON = profile.geojson || { type: 'FeatureCollection', features: [] };
  LAYERS = profile.layers || [];
  calcProfileIntegral(key);
  activeKey = '_integral';
  return true;
}

// ── Интегральные баллы ────────────────────────────────────────────────────
function normalizeRelative(values) {
  const valid = values.filter(function(v) {
    return v != null && !isNaN(v);
  });
  if (valid.length === 0) {
    return values.map(function() { return null; });
  }
  const lo = Math.min.apply(null, valid);
  const hi = Math.max.apply(null, valid);
  if (hi === lo) {
    return values.map(function(v) {
      return (v == null || isNaN(v)) ? null : 0;
    });
  }
  return values.map(function(v) {
    if (v == null || isNaN(v)) return null;
    const z = (v - lo) / (hi - lo);
    return parseFloat((z * 10.0).toFixed(3));
  });
}

function calcProfileIntegral(profileKey) {
  const profile = PROFILES[profileKey];
  if (!profile) return;
  const geojson = profile.geojson || { type: 'FeatureCollection', features: [] };
  const layers = profile.layers || [];
  const rawValues = [];

  geojson.features.forEach(function(f) {
    let num = 0, den = 0;
    layers.forEach(function(layer) {
      const s = f.properties[layer.key];
      if (s != null && !isNaN(s)) {
        const w = getCriterionWeight(profileKey, layer.key);
        num += s * w;
        den += w;
      }
    });
    rawValues.push(den > 0 ? parseFloat((num / den).toFixed(3)) : null);
  });

  const normalizedValues = normalizeRelative(rawValues);
  geojson.features.forEach(function(f, idx) {
    f.properties._integral_raw = rawValues[idx];
    f.properties._integral = normalizedValues[idx];
  });
}

function featureIdentity(feature, index) {
  const props = feature && feature.properties ? feature.properties : {};
  if (props.hex_id != null) return 'hex:' + props.hex_id;
  if (feature && feature.id != null) return 'id:' + feature.id;
  return 'idx:' + index;
}

function ensureCombinedGeoJSON() {
  const cityFeatures = (((PROFILES.city || {}).geojson || {}).features || []);
  if (COMBINED_GEOJSON.features.length === cityFeatures.length && cityFeatures.length > 0) {
    return;
  }
  COMBINED_GEOJSON = {
    type: 'FeatureCollection',
    features: cityFeatures.map(function(feature, index) {
      const props = feature.properties || {};
      return {
        type: 'Feature',
        id: feature.id,
        geometry: feature.geometry,
        properties: {
          hex_id: props.hex_id != null ? props.hex_id : index,
          _source_identity: featureIdentity(feature, index),
          _city_integral: null,
          _investor_integral: null,
          _combined_integral_raw: null,
          _combined_integral: null,
        },
      };
    }),
  };
}

function indexFeatures(features) {
  const result = {};
  features.forEach(function(feature, index) {
    result[featureIdentity(feature, index)] = feature;
  });
  return result;
}

function calcCombinedIntegral() {
  if (!hasCombinedProfile()) return;
  ensureCombinedGeoJSON();

  const cityFeatures = (((PROFILES.city || {}).geojson || {}).features || []);
  const investorFeatures = (((PROFILES.investor || {}).geojson || {}).features || []);
  const cityById = indexFeatures(cityFeatures);
  const investorById = indexFeatures(investorFeatures);
  const cityWeight = getCombinedWeight('city');
  const investorWeight = getCombinedWeight('investor');
  const rawValues = [];

  // 1. Взвешенное среднее интегралов двух профилей.
  COMBINED_GEOJSON.features.forEach(function(feature, index) {
    const identity = feature.properties._source_identity || featureIdentity(feature, index);
    const allowIndexFallback = identity.indexOf('idx:') === 0;
    const cityFeature = cityById[identity] || (allowIndexFallback ? cityFeatures[index] : null);
    const investorFeature = investorById[identity]
      || (allowIndexFallback ? investorFeatures[index] : null);

    const cityValue = cityFeature && cityFeature.properties
      ? cityFeature.properties._integral : null;
    const investorValue = investorFeature && investorFeature.properties
      ? investorFeature.properties._integral : null;

    let numerator = 0;
    let denominator = 0;

    if (cityValue != null && !isNaN(cityValue)) {
      numerator += cityValue * cityWeight;
      denominator += cityWeight;
    }
    if (investorValue != null && !isNaN(investorValue)) {
      numerator += investorValue * investorWeight;
      denominator += investorWeight;
    }

    const rawCombined = denominator > 0 ? numerator / denominator : null;

    feature.properties._city_integral = cityValue;
    feature.properties._investor_integral = investorValue;
    feature.properties._combined_integral_raw = rawCombined;
    rawValues.push(rawCombined);
  });

  // 2. Повторная min-max-нормировка общего результата по всем гексагонам.
  // Минимум территории становится 0, максимум — 10.
  const normalizedValues = normalizeRelative(rawValues);

  COMBINED_GEOJSON.features.forEach(function(feature, index) {
    feature.properties._combined_integral = normalizedValues[index];
  });

  const validRaw = rawValues.filter(function(v) {
    return v != null && !isNaN(v);
  });
  const validNormalized = normalizedValues.filter(function(v) {
    return v != null && !isNaN(v);
  });

  if (validRaw.length > 0 && validNormalized.length > 0) {
    console.info(
      '[combined integral]',
      'raw range:',
      Math.min.apply(null, validRaw),
      Math.max.apply(null, validRaw),
      'normalized range:',
      Math.min.apply(null, validNormalized),
      Math.max.apply(null, validNormalized)
    );
  }
}

// ── Состояние ─────────────────────────────────────────────────────────────
let activeKey = '_integral';
let hexLayer  = null;

// ── Стиль полигона ────────────────────────────────────────────────────────
function featureStyle(feature) {
  const value = feature.properties[activeKey];
  return {
    fillColor:   valToRgb(value),
    fillOpacity: scoreFillOpacity(value, false),
    weight:      0.35,
    color:       '#1f2937',
  };
}

// ── Тултип — строится лениво при наведении ────────────────────────────────
function buildTooltip(feature) {
  const p   = feature.properties;
  const idv = p.hex_id != null ? p.hex_id : (p.id != null ? p.id : '-');
  let h = '<div class="tt-title">Ячейка ' + idv + '</div>';

  if (activeProfile === COMBINED_PROFILE) {
    const cityValue = p[COMBINED_LAYER_KEYS.city];
    const investorValue = p[COMBINED_LAYER_KEYS.investor];
    const combinedValue = p[COMBINED_LAYER_KEYS.result];
    const combinedRaw = p._combined_integral_raw;

    if (activeKey === COMBINED_LAYER_KEYS.result) {
      h += '<div class="tt-integral">Общий нормализованный балл: '
        + (combinedValue != null ? combinedValue.toFixed(2) : 'нет данных') + '</div>';
      h += '<div class="tt-row"><span class="tt-dim">До нормировки</span>'
        + '<span class="tt-val">' + (combinedRaw != null ? combinedRaw.toFixed(3) : '—')
        + '</span></div>';
      h += '<div style="margin-top:5px;border-top:1px solid #1e3a5f;padding-top:4px">'
        + '<div class="tt-row"><span class="tt-dim">Интеграл города</span>'
        + '<span class="tt-val">' + (cityValue != null ? cityValue.toFixed(2) : '—')
        + '<span class="tt-dim"> ×' + getCombinedWeight('city') + '</span></span></div>'
        + '<div class="tt-row"><span class="tt-dim">Интеграл инвестора</span>'
        + '<span class="tt-val">' + (investorValue != null ? investorValue.toFixed(2) : '—')
        + '<span class="tt-dim"> ×' + getCombinedWeight('investor') + '</span></span></div>'
        + '</div>';
    } else {
      const isCity = activeKey === COMBINED_LAYER_KEYS.city;
      const label = isCity ? 'Интегральный слой города' : 'Интегральный слой инвестора';
      const value = isCity ? cityValue : investorValue;
      h += '<div class="tt-row"><span class="tt-dim">' + label + '</span>'
        + '<span class="tt-val">' + (value != null ? value.toFixed(2) : 'нет данных')
        + '</span></div>';
    }
    return h;
  }

  if (activeKey === '_integral') {
    const iv = p._integral;
    h += '<div class="tt-integral">Интеграл: '
       + (iv != null ? iv.toFixed(2) : 'нет данных') + '</div>';
    h += '<div style="margin-top:5px;border-top:1px solid #1e3a5f;padding-top:4px">';
    LAYERS.forEach(function(lr) {
      const v = p[lr.key], w = getCriterionWeight(activeProfile, lr.key);
      h += '<div class="tt-row"><span class="tt-dim">' + escapeHtml(lr.label) + '</span>'
        + '<span class="tt-val">' + (v != null ? v.toFixed(2) : '—')
        + '<span class="tt-dim"> ×' + w + '</span></span></div>';
    });
    h += '</div>';
  } else {
    const lm  = LAYERS.filter(function(l) { return l.key === activeKey; })[0];
    const lbl = lm ? lm.label : activeKey;
    const v   = p[activeKey];
    h += '<div class="tt-row"><span class="tt-dim">' + escapeHtml(lbl) + '</span>'
       + '<span class="tt-val">' + (v != null ? v.toFixed(2) : 'нет данных')
       + '</span></div>';
  }
  return h;
}

// ── Создание слоя гексагонов ──────────────────────────────────────────────
function initHexLayer() {
  if (hexLayer) map.removeLayer(hexLayer);
  const hexRenderer = L.canvas({ padding: 0.1 });
  hexLayer = L.geoJSON(GEOJSON, {
    renderer: hexRenderer,
    style: featureStyle,
    onEachFeature: function(feature, layer) {
      layer.bindTooltip('', { sticky: true, opacity: 1.0 });
      layer.on('mouseover', function() {
        this.setTooltipContent(buildTooltip(feature));
        this.setStyle({
          weight: 1.5,
          color: '#f1f5f9',
          fillOpacity: scoreFillOpacity(feature.properties[activeKey], true),
        });
        this.bringToFront();
      });
      layer.on('mouseout', function() {
        if (hexLayer) hexLayer.resetStyle(this);
      });
    },
  }).addTo(map);
}

// ── Перекрашивание (без пересоздания слоя) ────────────────────────────────
function repaint() {
  if (hexLayer) hexLayer.setStyle(featureStyle);
}

// ── Выбор отображаемого слоя ──────────────────────────────────────────────
function selectLayer(key) {
  activeKey = key;
  document.querySelectorAll('.layer-item').forEach(function(el) { el.classList.remove('active'); });
  const item = document.getElementById('item-' + key);
  if (item) item.classList.add('active');
  const radio = document.getElementById('radio-' + key);
  if (radio) radio.checked = true;
  repaint();
}

// ── Изменение весов ───────────────────────────────────────────────────────
function attachWeightHandlers() {
  document.querySelectorAll('.weight-input').forEach(function(el) {
    if (el.dataset.weightBound === '1') return;
    el.dataset.weightBound = '1';
    el.addEventListener('input', onWeightChange);
    el.addEventListener('change', onWeightChange);
  });
}

function onWeightChange(event) {
  const scope = event.currentTarget.dataset.weightScope;
  const key = event.currentTarget.dataset.layerKey;
  const weight = clampWeight(event.currentTarget.value);
  event.currentTarget.value = weight;

  if (scope === 'combined') {
    profileWeights(COMBINED_PROFILE)[key] = weight;
    calcCombinedIntegral();
    repaint();
    return;
  }

  const profileKey = event.currentTarget.dataset.profileKey || activeProfile;
  profileWeights(profileKey)[key] = weight;
  calcProfileIntegral(profileKey);
  if (hasCombinedProfile()) calcCombinedIntegral();
  repaint();
}

function selectProfile(key) {
  if (!activateProfileData(key)) return;
  renderProfileTabs();
  renderLayersPanel();
  initHexLayer();
  fitToBounds();
}

// ── Автоподбор масштаба ───────────────────────────────────────────────────
function fitToBounds() {
  let minLat =  90, maxLat = -90, minLon = 180, maxLon = -180;
  GEOJSON.features.forEach(function(f) {
    if (!f.geometry || !f.geometry.coordinates || !f.geometry.coordinates[0]) return;
    f.geometry.coordinates[0].forEach(function(pt) {
      const lon = pt[0], lat = pt[1];
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
      if (lon < minLon) minLon = lon;
      if (lon > maxLon) maxLon = lon;
    });
  });
  if (minLat < maxLat) {
    map.fitBounds([[minLat, minLon], [maxLat, maxLon]], { padding: [24, 24] });
  } else {
    map.setView([__CENTER_LAT__, __CENTER_LON__], 12);
  }
}

// ── Слой площадок КРТ (чёрная обводка, без заливки) ─────────────────────
L.geoJSON(KRT_PLOTS, {
  renderer: krtRenderer,
  pane: 'krtPane',
  style: {
    color:   '#000000',
    weight:  2,
    opacity: 1,
    fill:    false,
  },
}).addTo(map);

// ── Инициализация ─────────────────────────────────────────────────────────
if ((activeProfile === COMBINED_PROFILE && !hasCombinedProfile())
    || (activeProfile !== COMBINED_PROFILE && !PROFILES[activeProfile])) {
  activeProfile = orderedProfileKeys()[0];
}
activateProfileData(activeProfile);
renderProfileTabs();
renderLayersPanel();
initHexLayer();
fitToBounds();
// Страховка: пересчёт размера карты после отрисовки DOM
setTimeout(function() { map.invalidateSize(); }, 200);
</script>
</body>
</html>
"""


def main(
    results_gpkg: str | pathlib.Path | None = None,
    criteria_sheet: str | None = None,
    profile_key: str | None = None,
    profile_label: str | None = None,
    output_html: str | pathlib.Path = OUTPUT_HTML,
) -> None:
    """Добавляет/обновляет один профиль в общей интерактивной HTML-карте."""
    import crt_pipeline as P

    output_html = pathlib.Path(output_html)
    if criteria_sheet is None:
        criteria_sheet = P.load_run_config("run_config.txt").criteria_sheet
    criteria_sheet = P.normalize_criteria_sheet(criteria_sheet)

    derived_key, derived_label = P.profile_from_sheet(criteria_sheet)
    profile_key = profile_key or derived_key
    profile_label = profile_label or derived_label

    if results_gpkg is None:
        profile_result = pathlib.Path(f"results/krt_scores_{profile_key}.gpkg")
        results_gpkg = profile_result if profile_result.exists() else DEFAULT_RESULTS_GPKG
    results_gpkg = pathlib.Path(results_gpkg)

    print("=" * 55)
    print("Генерация интерактивной карты КРТ")
    print("=" * 55)
    print(f"Профиль: {profile_label} | лист: {criteria_sheet}")

    print("\n[1/6] Подготовка Leaflet …")
    leaflet_head = _fetch_leaflet()

    print("\n[2/6] Загрузка метаданных критериев …")
    metadata = load_criterion_metadata(XLSX_PATH, criteria_sheet)
    if metadata:
        for col, item in metadata.items():
            print(f"      {col}: {item['label']} [{item['direction']}]")

    print("\n[3/6] Загрузка геоданных …")
    features, score_cols = load_geodata(
        results_gpkg,
        expected_score_cols=list(metadata) if metadata else None,
    )
    layers_meta = [
        {"key": col, **metadata.get(col, {"label": col})}
        for col in score_cols
    ]

    print("\n[4/6] Чтение существующих профилей …")
    profiles = load_existing_profiles(output_html)
    # Совместный профиль является виртуальным и всегда рассчитывается в браузере.
    profiles.pop("combined", None)
    if profiles:
        existing = ", ".join(str(p.get("label", key)) for key, p in profiles.items())
        print(f"      Сохранены профили: {existing}")
    else:
        print("      Существующий HTML не найден или профилей в нём нет")

    profiles[profile_key] = {
        "label": profile_label,
        "sheet_name": criteria_sheet,
        "geojson": {"type": "FeatureCollection", "features": features},
        "layers": layers_meta,
    }
    enrich_profile_layers_metadata(profiles, XLSX_PATH)

    print("\n[4.5/6] Оценка центра карты …")
    center = estimate_center(features)
    print(f"      Центр: {center[0]:.4f}° с.ш., {center[1]:.4f}° в.д.")

    print("\n[5/6] Загрузка площадок КРТ и шрифтов …")
    krt_plots_json = load_krt_plots()
    fonts_css = build_fonts_css()

    print("\n[6/6] Сборка HTML …")
    html = render_html(
        profiles,
        profile_key,
        center,
        leaflet_head,
        krt_plots_json,
        fonts_css,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    size_kb = output_html.stat().st_size // 1024

    print(f"\n[OK] Готово: {output_html}  ({size_kb} КБ)")
    print(f"  Активная вкладка: {profile_label}")
    print()
    print("  Следующий шаг: откройте файл в браузере.")
    print("  Для подложки Mapbox задайте переменную окружения MAPBOX_TOKEN.")
    print()


if __name__ == "__main__":
    main()
