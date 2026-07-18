# get_iso.py
from __future__ import annotations
import os, json, time, logging
from typing import Iterable, Sequence

import requests
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv
from shapely.geometry import Point
from shapely.ops import unary_union

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logger = logging.getLogger("iso")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# нормализация видов трафика -> официальные названия Valhalla
_COSTING = {
    "pedestrian": "pedestrian", "walk": "pedestrian", "foot": "pedestrian",
    "bicycle": "bicycle", "bike": "bicycle",
    "auto": "auto", "car": "auto",
    "multimodal": "multimodal", "multimodal": "multimodal",
}

def _valhalla_base():
    base_url = os.getenv("VALHALLA_URL", "").strip()
    if not base_url:
        raise RuntimeError(
            "Не задан адрес Valhalla. Установите переменную окружения VALHALLA_URL."
        )
    return base_url.rstrip("/")

def _norm_points(df_or_points: gpd.GeoDataFrame | Iterable[Sequence[float]]) -> list[tuple[float, float]]:
    """
    Возвращает список (lon, lat) в WGS84.
    GeoDataFrame -> перепроецируем в 4326, берём Point/representative_point.
    Списки/кортежи считаем как (lon, lat).
    """
    pts: list[tuple[float, float]] = []
    if isinstance(df_or_points, gpd.GeoDataFrame):
        gdf = df_or_points
        if gdf.crs is None or str(gdf.crs).lower() not in ("epsg:4326", "wgs84"):
            gdf = gdf.to_crs(4326)
        for g in gdf.geometry:
            if isinstance(g, Point):
                pts.append((g.x, g.y))
            elif g:
                rp = g.representative_point()
                pts.append((rp.x, rp.y))
    else:
        for p in df_or_points:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
    return pts

def _post_iso(url: str, payload: dict, timeout: float) -> dict:
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _call_iso_api(
    locations_lonlat: list[tuple[float, float]],
    costing: str,
    time_minutes: float | None = None,
    distance_km: float | None = None,
    polygons: bool = True,
    timeout: float = 30,
    base_url: str | None = None,
    retries: int = 2,
    backoff: float = 0.7,
) -> dict:
    """
    Вызов изохрон Valhalla.

    locations_lonlat: список точек (lon, lat).
    Можно задать ИЛИ time_minutes, ИЛИ distance_km.
    """
    if time_minutes is None and distance_km is None:
        raise ValueError("Isochrone: either time_minutes or distance_km must be provided")

    if distance_km is not None:
        contour = {"distance": float(distance_km)}
        log_suffix = f"{distance_km} km"
    else:
        contour = {"time": float(time_minutes)}
        log_suffix = f"{time_minutes} min"

    base = (base_url or _valhalla_base()).rstrip("/")
    url = f"{base}/isochrone"
    payload = {
        "locations": [{"lat": lat, "lon": lon} for lon, lat in locations_lonlat],
        "costing": costing,
        "contours": [contour],
        "polygons": bool(polygons),
    }

    attempt = 0
    while True:
        try:
            logger.debug(
                "POST %s (n=%d points, %s, %s)",
                url, len(locations_lonlat), costing, log_suffix
            )
            return _post_iso(url, payload, timeout)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            txt = (e.response.text[:300] if e.response is not None else "")
            retryable = code in (429, 500, 502, 503, 504)
            logger.warning("Valhalla HTTP %s on attempt %d: %s ... %s", code, attempt + 1, e, txt)
            if attempt >= retries or not retryable:
                raise
            time.sleep(backoff * (2 ** attempt))
            attempt += 1
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.warning("Valhalla connection error on attempt %d: %s", attempt + 1, e)
            if attempt >= retries:
                raise
            time.sleep(backoff * (2 ** attempt))
            attempt += 1

def _features_to_gdf(geojson_obj: dict | None) -> gpd.GeoDataFrame:
    if not geojson_obj:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    features = []
    if isinstance(geojson_obj, dict):
        if geojson_obj.get("type") == "FeatureCollection":
            features = geojson_obj.get("features") or []
        elif "features" in geojson_obj:
            features = geojson_obj["features"] or []
    gdf = gpd.GeoDataFrame.from_features(features, crs=4326)
    if "geometry" not in gdf.columns:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    return gdf

def get_iso(
    df_points: gpd.GeoDataFrame | Iterable[Sequence[float]],
    iso_type: str | None = None,
    iso_time: float | None = None,
    iso_distance: float | None = None,
    dissolve: bool = True,
    chunk_size: int = 10,
    timeout: float = 40,
    base_url: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Строит изохрон(ы) через HTTP Valhalla.

    Параметры прямо соответствуют колонкам таблицы алгоритмов:

      * iso_type      — 'pedestrian' | 'bicycle' | 'auto' | 'multimodal'
      * iso_time      — время изохроны в МИНУТАХ
      * iso_distance  — расстояние изохроны в МЕТРАХ

    ВАЖНО: iso_time и iso_distance взаимоисключающие.
      - Если заданы оба → ValueError.
      - Если задан только iso_distance → строим по расстоянию.
      - Если задан только iso_time → строим по времени.
      - Если не задано ни то, ни другое → по умолчанию 15 минут.
    """

    # нормализуем точки в список (lon, lat) в WGS84
    pts = _norm_points(df_points)
    if not pts:
        logger.warning("get_iso: пустой список точек → пустой GeoDataFrame")
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    
    # Дедупликация точек с одинаковыми координатами
    original_count = len(pts)
    pts = list(set(pts))
    if len(pts) < original_count:
        logger.info(
            "get_iso: дедупликация точек: %d → %d (удалено %d дубликатов)",
            original_count, len(pts), original_count - len(pts)
        )
    
    logger.info(
        "get_iso: n_points=%d, iso_type=%s, iso_time=%s, iso_distance=%s",
        len(pts), iso_type, iso_time, iso_distance
    )

    # --- тип изохроны / режим движения ---
    mode = (iso_type or "pedestrian").lower()
    costing = _COSTING.get(mode, "pedestrian")

    # --- проверка взаимоисключающих параметров ---
    if iso_time is not None and iso_distance is not None:
        raise ValueError(
            "Both iso_time and iso_distance are specified. "
            "Эти параметры взаимоисключают друг друга — оставьте только один из них."
        )

    # определяем, что именно будем передавать в Valhalla
    distance_km: float | None = None
    time_minutes: float | None = None

    if iso_distance is not None:
        # таблица даёт метры → переводим в километры
        distance_km = float(iso_distance) / 1000.0
    else:
        # используем iso_time либо дефолт 15 минут
        time_minutes = float(iso_time) if iso_time is not None else 15.0

    base = (base_url or _valhalla_base()).rstrip("/")

    parts: list[gpd.GeoDataFrame] = []

    # флаг: сервер разрешает только 1 точку на запрос
    server_single_location_mode = False

    def _process_points_individually(chunk, chunk_idx, total_chunks):
        """Обработка точек по одной с логом прогресса."""
        logger.info(
            "Valhalla: chunk %d/%d → per-point mode (%d точек)",
            chunk_idx + 1,
            total_chunks,
            len(chunk),
        )
        successful = 0
        failed = 0
        for j, pt in enumerate(chunk):
            try:
                resp1 = _call_iso_api(
                    [pt],
                    costing,
                    time_minutes=time_minutes,
                    distance_km=distance_km,
                    polygons=True,
                    timeout=timeout,
                    base_url=base,
                )
                g1 = _features_to_gdf(resp1)
                if not g1.empty:
                    parts.append(g1[["geometry"]])
                    successful += 1
                else:
                    logger.warning(
                        "  Пустой результат для точки %d/%d в chunk %d",
                        j + 1, len(chunk), chunk_idx + 1
                    )
                    failed += 1
            except (requests.HTTPError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e1:
                logger.warning(
                    "  Valhalla error for point %d/%d in chunk %d: %s",
                    j + 1,
                    len(chunk),
                    chunk_idx + 1,
                    e1,
                )
                failed += 1
        logger.info(
            "Valhalla: chunk %d/%d обработан: успешно %d, ошибок %d",
            chunk_idx + 1, total_chunks, successful, failed
        )

    n = len(pts)
    n_chunks = max(1, (n + chunk_size - 1) // chunk_size)

    for i in range(0, n, chunk_size):
        chunk_idx = i // chunk_size
        chunk = pts[i:i + chunk_size]

        if server_single_location_mode or len(chunk) == 1:
            # уже знаем, что сервер принимает только по одной точке
            _process_points_individually(chunk, chunk_idx, n_chunks)
            continue

        logger.info(
            "Valhalla: processing chunk %d/%d (size=%d)",
            chunk_idx + 1,
            n_chunks,
            len(chunk),
        )

        try:
            resp = _call_iso_api(
                chunk,
                costing,
                time_minutes=time_minutes,
                distance_km=distance_km,
                polygons=True,
                timeout=timeout,
                base_url=base,
                retries=0,  # таймаут/ошибка чанка → сразу фоллбек per-point без доп. ожидания
            )
            gpart = _features_to_gdf(resp)
            
            # Проверяем, получены ли изохроны для всех точек
            n_points_in_chunk = len(chunk)
            n_geometries = len(gpart) if not gpart.empty else 0
            
            logger.info(
                "Valhalla: chunk %d/%d ответ получен: %d геометрий в GDF, %d точек отправлено",
                chunk_idx + 1, n_chunks, n_geometries, n_points_in_chunk
            )
            
            # Проверяем: если получено меньше геометрий, чем отправлено точек - обрабатываем по одной
            if n_geometries < n_points_in_chunk:
                # Получено меньше изохрон, чем отправлено точек
                # Обрабатываем весь чанк по одной, чтобы гарантировать обработку всех точек
                logger.warning(
                    "Valhalla вернул %d изохрон для %d точек в чанке %d/%d. "
                    "Обрабатываем точки по одной для гарантии обработки всех точек.",
                    n_geometries, n_points_in_chunk, chunk_idx + 1, n_chunks
                )
                # НЕ сохраняем частичные результаты - обработаем весь чанк по одной
                # Это гарантирует, что все точки будут обработаны и не будет дубликатов
                _process_points_individually(chunk, chunk_idx, n_chunks)
                continue
            
            # Все точки обработаны успешно
            if not gpart.empty:
                logger.info(
                    "Valhalla: chunk %d/%d успешно обработан (%d изохрон для %d точек)",
                    chunk_idx + 1, n_chunks, n_geometries, n_points_in_chunk
                )
                parts.append(gpart[["geometry"]])
            else:
                logger.warning(
                    "Valhalla: chunk %d/%d вернул пустой результат для %d точек",
                    chunk_idx + 1, n_chunks, n_points_in_chunk
                )
            continue

        except requests.HTTPError as e:
            # если лимит на количество точек — переключаемся на режим по одной точке
            txt = e.response.text if e.response is not None else ""
            if "Exceeded max locations" in txt or '"error_code":150' in txt:
                if not server_single_location_mode:
                    logger.info(
                        "Valhalla: сервер ограничивает число точек в запросе → "
                        "переключаемся на per-point режим"
                    )
                    server_single_location_mode = True

                _process_points_individually(chunk, chunk_idx, n_chunks)
                continue

            logger.warning(
                "Valhalla error for chunk %d/%d (size=%d): %s. "
                "Пытаемся обработать точки по одной.",
                chunk_idx + 1, n_chunks, len(chunk), e
            )
            _process_points_individually(chunk, chunk_idx, n_chunks)
            continue

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.warning(
                "Valhalla network error for chunk %d/%d (size=%d): %s. "
                "Переходим к per-point режиму.",
                chunk_idx + 1, n_chunks, len(chunk), e
            )
            _process_points_individually(chunk, chunk_idx, n_chunks)
            continue

    if not parts:
        logger.warning("get_iso: не получено ни одной изохроны из %d точек", len(pts))
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    out = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        geometry="geometry",
        crs=4326,
    )

    n_isochrones = len(out)
    logger.info(
        "get_iso: получено %d изохрон из %d точек (dissolve=%s)",
        n_isochrones, len(pts), dissolve
    )

    if dissolve:
        result = gpd.GeoDataFrame({"geometry": [unary_union(out.geometry)]}, crs=4326)
        logger.info("get_iso: изохроны объединены в один полигон")
        return result

    return out
