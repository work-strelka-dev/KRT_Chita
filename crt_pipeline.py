"""
КРТ-оценка на гексагональной сетке. Табличный (конфиг-управляемый) движок.

Лист с критериями задаётся в run_config.txt параметром criteria_sheet.
Основная логика управляется таблицей; отдельные методологические исключения
(например, фильтр высотности для приаэродромных территорий) явно зафиксированы
в коде.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import json
import ntpath
import pathlib
import re
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

TARGET_CRS = 32649   # UTM zone 49N (Чита)
HEX_ID = "hex_id"

DEFAULT_CRITERIA_SHEET = "7.2 Критерии (город)"
CRITERIA_SHEET_ALIASES = {
    "город": DEFAULT_CRITERIA_SHEET,
    "city": DEFAULT_CRITERIA_SHEET,
    "7.2": DEFAULT_CRITERIA_SHEET,
    "инвестор": "7.1 Критерии (инвестор)",
    "investor": "7.1 Критерии (инвестор)",
    "7.1": "7.1 Критерии (инвестор)",
}


@dataclass
class RunConfig:
    criteria: list[int] | None = None   # None = запустить все
    hex_h_spacing: float = 400.0        # ширина гексагона, м
    hex_v_spacing: float = 400.0        # высота гексагона, м
    criteria_sheet: str = DEFAULT_CRITERIA_SHEET


def normalize_criteria_sheet(value: str | None) -> str:
    """Нормализует имя листа критериев и поддерживает короткие алиасы."""
    raw = (value or "").strip()
    if not raw:
        return DEFAULT_CRITERIA_SHEET
    return CRITERIA_SHEET_ALIASES.get(raw.lower(), raw)


def profile_from_sheet(sheet_name: str) -> tuple[str, str]:
    """Возвращает машинный ключ и подпись HTML-вкладки для листа критериев."""
    normalized = normalize_criteria_sheet(sheet_name)
    lowered = normalized.lower()
    if "инвестор" in lowered:
        return "investor", "Инвестор"
    if "город" in lowered:
        return "city", "Город"

    # Для произвольного листа нужен стабильный ASCII-ключ, пригодный для HTML/JS.
    import hashlib
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
    return f"profile_{digest}", normalized


def _fix_filter_expr(expr: str) -> str:
    """Оборачивает голые строки в кавычки для pandas .query().

    Пример: «house_condition == Аварийный» → «house_condition == 'Аварийный'»
    Числа и уже закавыченные значения не трогает.
    """
    def _quote(m: re.Match) -> str:
        op, val = m.group(1), m.group(2)
        if val[0] in ('"', "'", "@"):
            return m.group(0)
        try:
            float(val)
            return m.group(0)
        except ValueError:
            return f"{op} '{val}'"
    return re.sub(r'(==|!=|<=|>=|<|>)\s*(\S+)', _quote, expr)

# Позиции машинных колонок в листе критериев (0-indexed, считая от строки-заголовка)
_C_CRIT_NUM   = 0   # №
_C_CRIT_NAME  = 1   # Наименование (критерий)
_C_CRIT_DIR   = 3   # Тип шкалы (критерий): Прямая/Обратная
_C_PP         = 4   # №, п/п
_C_METRIC_DIR = 8   # Тип шкалы (метрика): Прямая/Обратная
_C_SCALE_KIND = 9   # Относительный / Предзаданный
_C_GRAD_TEXT    = 10  # Градация оценки
_C_GRAD_SCORE   = 11  # Балл
_C_LAYER        = 13  # layer
_C_SPAT_METH    = 14  # spatial_method
_C_SPAT_PARAMS  = 15  # spatial_params (JSON: {"buffer_m": 50})
_C_MATH_OP      = 16  # math_op
_C_SF1        = 17  # source_field_1
_C_SF2        = 18  # source_field_2
_C_ATTR_FILT  = 19  # attr_filter
_C_OUT_NAME   = 20  # out_name


# --------------------------------------------------------------------------- #
# Декларативные спецификации
# --------------------------------------------------------------------------- #
@dataclass
class OverlayItemSpec:
    """Один тип пространственного ограничения для взвешенного наложения.

    Один тип может собираться из нескольких файлов (например, ГП и ЕГРН).
    В пределах одной ячейки его балл учитывается только один раз независимо от
    количества пересекающих объектов и количества исходных файлов.
    """

    name: str
    score: float
    layer_paths: list[str]
    attr_filter: str | None = None


@dataclass
class MetricSpec:
    pp: str
    layer: str                 # абсолютный или относительный путь к слою
    spatial_method: str        # "within" | "intersect" | "buffer" | ...
    math_op: str               # "sum" | "density" | "diff" | "count"
    source_fields: list[str]   # [field_1] или [field_1, field_2]
    scale_kind: str            # "relative" | "predefined"
    direction: str             # "direct" | "inverse"
    attr_filter: str | None = None
    out_name: str = ""
    spatial_params: dict = field(default_factory=dict)
    iso_gradations: list[dict] = field(default_factory=list)
    numeric_gradations: list[dict] = field(default_factory=list)
    cat_gradations: list[dict] = field(default_factory=list)  # [{"value": str, "score": int}]
    overlay_items: list[OverlayItemSpec] = field(default_factory=list)

    def __post_init__(self):
        if not self.out_name:
            self.out_name = "m_" + self.pp.replace(".", "_")


@dataclass
class CriterionSpec:
    cid: int
    name: str
    direction: str             # направление на уровне критерия
    metrics: list[MetricSpec]
    weights: list[float] | None = None   # None = равные веса


# --------------------------------------------------------------------------- #
# Загрузка слоёв
# --------------------------------------------------------------------------- #
def _read_layer(path: str, *, layer: str | None = None) -> gpd.GeoDataFrame:
    """Читает один векторный слой.

    ``layer`` используется для многослойных контейнеров (например, GeoPackage
    или FileGDB). Отдельно обрабатывается конфликт, когда в атрибутах уже есть
    поле с именем ``geometry``.
    """
    try:
        kwargs = {"layer": layer} if layer is not None else {}
        return gpd.read_file(path, **kwargs)
    except ValueError as e:
        if "multiple columns using the geometry column name" not in str(e):
            raise
    # Атрибутное поле с именем 'geometry' конфликтует с геометрией — читаем через fiona
    import fiona
    from shapely.geometry import shape as _shape
    records, geoms = [], []
    open_kwargs = {"layer": layer} if layer is not None else {}
    with fiona.open(path, **open_kwargs) as src:
        crs = src.crs
        for feat in src:
            props = dict(feat["properties"])
            props["geometry_attr"] = props.pop("geometry", None)  # переименовываем конфликт
            records.append(props)
            raw_geom = feat.get("geometry")
            geoms.append(_shape(raw_geom) if raw_geom else None)
    return gpd.GeoDataFrame(records, geometry=geoms, crs=crs)


_VECTOR_FILE_EXTENSIONS = {
    ".shp", ".gpkg", ".geojson", ".json", ".fgb", ".sqlite",
    ".kml", ".gml", ".tab",
}


def _directory_vector_sources(directory: pathlib.Path) -> list[pathlib.Path]:
    """Возвращает поддерживаемые векторные источники внутри папки.

    Поиск рекурсивный. Каталоги FileGDB считаются самостоятельными
    источниками; их внутренние служебные файлы отдельно не перебираются.
    """
    sources: list[pathlib.Path] = []
    gdb_dirs = sorted(
        (p for p in directory.rglob("*") if p.is_dir() and p.suffix.lower() == ".gdb"),
        key=lambda p: str(p).lower(),
    )
    sources.extend(gdb_dirs)

    for candidate in sorted(directory.rglob("*"), key=lambda p: str(p).lower()):
        if not candidate.is_file() or candidate.suffix.lower() not in _VECTOR_FILE_EXTENSIONS:
            continue
        # Не захватываем служебные файлы, лежащие внутри уже найденной FileGDB.
        if any(gdb == candidate.parent or gdb in candidate.parents for gdb in gdb_dirs):
            continue
        sources.append(candidate)
    return sources


def _read_datasource_layers(path: pathlib.Path) -> list[tuple[str, gpd.GeoDataFrame]]:
    """Читает все пространственные слои одного контейнера/файла."""
    import fiona

    try:
        layer_names = list(fiona.listlayers(str(path)))
    except Exception:
        layer_names = []

    # Для обычного однослойного файла сохраняем прежнее поведение.
    if len(layer_names) <= 1:
        label = layer_names[0] if layer_names else path.stem
        return [(label, _read_layer(str(path), layer=layer_names[0] if layer_names else None))]

    return [(name, _read_layer(str(path), layer=name)) for name in layer_names]


def _read_layer_source(path: str) -> gpd.GeoDataFrame:
    """Читает файл либо объединяет все векторные слои из указанной папки.

    Запись в Excel вида ``Смешанное\Расположение лесных территорий`` может
    указывать не на файл, а на каталог. В этом случае все поддерживаемые
    векторные файлы внутри каталога (включая вложенные папки и все слои
    GeoPackage/FileGDB) приводятся к ``TARGET_CRS`` и объединяются в один
    GeoDataFrame. Только после этого выполняется пространственная метрика.
    """
    source = pathlib.Path(path)
    if not source.is_dir():
        return _read_layer(path)

    data_sources = _directory_vector_sources(source)
    if not data_sources:
        raise FileNotFoundError(
            f"В папке {path!r} не найдено поддерживаемых векторных слоёв"
        )

    parts: list[gpd.GeoDataFrame] = []
    loaded_labels: list[str] = []
    failures: list[str] = []

    for datasource in data_sources:
        try:
            layers = _read_datasource_layers(datasource)
        except Exception as exc:
            failures.append(f"{datasource}: {exc}")
            continue

        for layer_name, gdf in layers:
            try:
                if gdf.crs is None:
                    raise ValueError("у слоя не задана CRS")
                projected = gdf.to_crs(TARGET_CRS)
                projected = projected.loc[
                    projected.geometry.notna() & ~projected.geometry.is_empty
                ].copy()
                parts.append(projected)
                loaded_labels.append(f"{datasource.name}:{layer_name}")
            except Exception as exc:
                failures.append(f"{datasource}:{layer_name}: {exc}")

    if failures:
        details = "\n - ".join(failures)
        raise RuntimeError(
            f"Не удалось полностью собрать папку {path!r}. Ошибки:\n - {details}"
        )
    if not parts:
        raise ValueError(f"В папке {path!r} нет непустых геометрий")

    merged = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True, sort=False),
        geometry="geometry",
        crs=TARGET_CRS,
    )
    print(
        f"   Источник-папка: {path} | объединено слоёв: {len(loaded_labels)} "
        f"| объектов: {len(merged)}"
    )
    return merged


# --------------------------------------------------------------------------- #
# Пространственная привязка
# --------------------------------------------------------------------------- #
def _clip_to_bbox(features: gpd.GeoDataFrame, bbox_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Быстрый bbox-фильтр перед sjoin: отсекает объекты вне охвата целевых ячеек."""
    minx, miny, maxx, maxy = bbox_gdf.total_bounds
    return features.cx[minx:maxx, miny:maxy]


def assign_within(features: gpd.GeoDataFrame, hexes: gpd.GeoDataFrame,
                  id_col: str = "", **_) -> gpd.GeoDataFrame:
    """Привязка по representative_point — каждая фича ровно в одной ячейке."""
    features = _clip_to_bbox(features, hexes)
    pts = features.copy()
    pts["geometry"] = features.geometry.representative_point()
    joined = gpd.sjoin(pts, hexes[[HEX_ID, "geometry"]], predicate="within", how="inner")
    if id_col and id_col in joined.columns:
        joined = joined.drop_duplicates(subset=[id_col], keep="first")
    return joined


def assign_intersects(features: gpd.GeoDataFrame, hexes: gpd.GeoDataFrame,
                      **_) -> gpd.GeoDataFrame:
    """Истинное геометрическое наложение: объект пересекает ячейку.

    В отличие от ``within`` здесь не используется representative_point. Метод
    предназначен для площадных ограничений, где важен сам факт наложения на
    любую часть гексагона.
    """
    features = _clip_to_bbox(features, hexes)
    if features.empty:
        return gpd.GeoDataFrame(
            columns=[HEX_ID, "geometry"],
            geometry="geometry",
            crs=hexes.crs,
        )
    return gpd.sjoin(
        features,
        hexes[[HEX_ID, "geometry"]],
        predicate="intersects",
        how="inner",
    )


def assign_buffer(features: gpd.GeoDataFrame, hexes: gpd.GeoDataFrame,
                  buffer_m: float = 50.0, **_) -> gpd.GeoDataFrame:
    """Привязка через буфер N м вокруг ячейки; одна фича может попасть в несколько ячеек."""
    hexes_buf = hexes[[HEX_ID, "geometry"]].copy()
    hexes_buf = hexes_buf.set_geometry(hexes_buf.geometry.buffer(buffer_m))
    features = _clip_to_bbox(features, hexes_buf)
    joined = gpd.sjoin(features, hexes_buf, predicate="intersects", how="inner")
    return joined


def assign_centroid_distance(features: gpd.GeoDataFrame, hexes: gpd.GeoDataFrame,
                              **_) -> gpd.GeoDataFrame:
    """Евклидово расстояние от центроида ячейки до ближайшего объекта в слое.

    Возвращает по одной строке на ячейку с колонкой _nearest_m (метры).
    Используется совместно с math_op = nearest_dist.
    """
    centroids = hexes[[HEX_ID, "geometry"]].copy()
    centroids = centroids.set_geometry(hexes.geometry.centroid)
    nearest = gpd.sjoin_nearest(
        centroids,
        features[["geometry"]].reset_index(drop=True),
        how="left",
        distance_col="_nearest_m",
    )
    # sjoin_nearest даёт несколько строк при равном расстоянии до нескольких объектов
    nearest = nearest.drop_duplicates(subset=[HEX_ID], keep="first")
    return nearest[[HEX_ID, "_nearest_m"]]


ASSIGNERS: dict[str, Callable] = {
    "within":            assign_within,
    "intersect":         assign_within,          # алиас: имя из листа критериев
    "intersects":        assign_intersects,      # реальное геометрическое пересечение
    "buffer":            assign_buffer,
    "centroid_distance": assign_centroid_distance,
    # "isochrone": ...  (позже)
}


# --------------------------------------------------------------------------- #
# Редукции
# --------------------------------------------------------------------------- #
def _grouped_sum(joined: gpd.GeoDataFrame, fld: str) -> pd.Series:
    return joined.groupby(HEX_ID)[fld].apply(
        lambda s: pd.to_numeric(s, errors="coerce").sum()
    )


def reduce_sum(joined, spec, hexes) -> pd.Series:
    return _grouped_sum(joined, spec.source_fields[0])


def reduce_density(joined, spec, hexes) -> pd.Series:
    """sum(field) / площадь_ячейки."""
    s = _grouped_sum(joined, spec.source_fields[0])
    area = hexes.set_index(HEX_ID).geometry.area
    return s / area.reindex(s.index)


def reduce_diff(joined, spec, hexes) -> pd.Series:
    """sum(field_1) − sum(field_2)."""
    f1, f2 = spec.source_fields[0], spec.source_fields[1]
    return _grouped_sum(joined, f1).sub(_grouped_sum(joined, f2), fill_value=0)


def reduce_count(joined, _spec, _hexes) -> pd.Series:
    """Количество объектов в ячейке (без учёта полей)."""
    return joined.groupby(HEX_ID).size().astype(float)


def reduce_count_density(joined, _spec, hexes) -> pd.Series:
    """Количество объектов в ячейке / площадь ячейки (объекты / м²)."""
    cnt = joined.groupby(HEX_ID).size().astype(float)
    area = hexes.set_index(HEX_ID).geometry.area
    return cnt / area.reindex(cnt.index)


def reduce_nearest_dist(joined, _spec, _hexes) -> pd.Series:
    """Расстояние до ближайшего объекта (из assign_centroid_distance), м."""
    return joined.groupby(HEX_ID)["_nearest_m"].first()


REDUCERS: dict[str, Callable] = {
    "sum":           reduce_sum,
    "density":       reduce_density,
    "diff":          reduce_diff,
    "count":         reduce_count,
    "count_density": reduce_count_density,
    "nearest_dist":  reduce_nearest_dist,
}


RASTER_MATH_OPS = {
    "raster_min",
    "raster_max",
    "raster_mean",
    "raster_median",
    "raster_sum",
    "raster_std",
    "raster_range",
    "raster_relief_slope",
}


def is_raster_metric(spec: MetricSpec) -> bool:
    """Возвращает True для метрик, которые агрегируют значения GeoTIFF по ячейкам."""
    return spec.spatial_method == "raster_zonal" or spec.math_op in RASTER_MATH_OPS


def _equivalent_diameter_m(hexes: gpd.GeoDataFrame) -> pd.Series:
    """Эквивалентный диаметр ячейки: диаметр круга той же площади, в метрах."""
    projected = hexes if hexes.crs and hexes.crs.is_projected else hexes.to_crs(TARGET_CRS)
    area = projected.set_index(HEX_ID).geometry.area.astype(float)
    return 2.0 * np.sqrt(area / np.pi)


def compute_raster_metric(
    spec: MetricSpec,
    raster_path: str,
    hexes: gpd.GeoDataFrame,
) -> pd.Series:
    """Считает зональную статистику растра внутри каждой ячейки.

    Поддерживаемые ``math_op``:
    ``raster_min``, ``raster_max``, ``raster_mean``, ``raster_median``,
    ``raster_sum``, ``raster_std``, ``raster_range`` (max − min) и
    ``raster_relief_slope``.

    ``raster_relief_slope`` сначала считает перепад высот ``Δh=max−min``,
    затем переводит его в оценочный угол уклона:
    ``degrees(atan(Δh / L))``. По умолчанию ``L`` — эквивалентный диаметр
    конкретной ячейки. Его можно заменить числом в метрах через
    ``spatial_params={"run_m": 200}``.

    Общие параметры:
    - ``band``: номер канала GeoTIFF, по умолчанию 1;
    - ``all_touched``: учитывать все пиксели, которых касается полигон;
    - ``run_m``: число или ``"cell_equivalent_diameter"``.

    Ячейки без валидных пикселей получают NaN, а не ноль.
    """
    if spec.math_op not in RASTER_MATH_OPS:
        raise ValueError(
            f"{spec.out_name}: неизвестный raster math_op={spec.math_op!r}; "
            f"доступны {sorted(RASTER_MATH_OPS)}"
        )
    if not raster_path:
        raise ValueError(f"{spec.out_name}: не задан путь к растру")

    try:
        import rasterio
        from rasterio.errors import WindowError
        from rasterio.features import geometry_mask, geometry_window
        from shapely.geometry import mapping
    except ImportError as exc:
        raise ImportError(
            "Для растровых метрик требуется пакет rasterio. "
            "Добавьте его в environment.yml и пересоздайте окружение."
        ) from exc

    band = int(spec.spatial_params.get("band", 1))
    all_touched = bool(spec.spatial_params.get("all_touched", False))

    result = pd.Series(
        np.nan,
        index=pd.Index(hexes[HEX_ID].values, name=HEX_ID),
        name=spec.out_name,
        dtype=float,
    )

    run_param = spec.spatial_params.get("run_m", "cell_equivalent_diameter")
    run_by_id: pd.Series | None = None
    fixed_run_m: float | None = None
    if spec.math_op == "raster_relief_slope":
        if isinstance(run_param, str):
            if run_param != "cell_equivalent_diameter":
                raise ValueError(
                    f"{spec.out_name}: run_m должен быть числом или "
                    "'cell_equivalent_diameter'"
                )
            run_by_id = _equivalent_diameter_m(hexes)
        else:
            fixed_run_m = float(run_param)
            if fixed_run_m <= 0:
                raise ValueError(f"{spec.out_name}: run_m должен быть > 0")

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"{spec.out_name}: у растра не задана CRS")
        if band < 1 or band > src.count:
            raise ValueError(
                f"{spec.out_name}: band={band}, но в растре каналов: {src.count}"
            )

        zonal = hexes[[HEX_ID, "geometry"]].to_crs(src.crs)

        for row in zonal.itertuples(index=False):
            hid = getattr(row, HEX_ID)
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            try:
                window = geometry_window(src, [mapping(geom)])
            except WindowError:
                continue

            data = src.read(band, window=window, masked=True)
            if data.size == 0:
                continue

            inside = geometry_mask(
                [mapping(geom)],
                out_shape=data.shape,
                transform=src.window_transform(window),
                invert=True,
                all_touched=all_touched,
            )
            mask = np.ma.getmaskarray(data)
            values = np.asarray(data.data, dtype=float)[inside & ~mask]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue

            if spec.math_op == "raster_min":
                value = float(values.min())
            elif spec.math_op == "raster_max":
                value = float(values.max())
            elif spec.math_op == "raster_mean":
                value = float(values.mean())
            elif spec.math_op == "raster_median":
                value = float(np.median(values))
            elif spec.math_op == "raster_sum":
                value = float(values.sum())
            elif spec.math_op == "raster_std":
                value = float(values.std())
            else:
                delta_h = float(values.max() - values.min())
                if spec.math_op == "raster_range":
                    value = delta_h
                else:
                    run_m = fixed_run_m
                    if run_m is None and run_by_id is not None:
                        run_m = float(run_by_id.loc[hid])
                    value = float(np.degrees(np.arctan2(delta_h, run_m)))

            result.loc[hid] = value

    return result


# --------------------------------------------------------------------------- #
# Движок одной метрики
# --------------------------------------------------------------------------- #
def compute_metric(spec: MetricSpec, layer: gpd.GeoDataFrame,
                   hexes: gpd.GeoDataFrame) -> pd.Series:
    """Сырое значение метрики на каждую ячейку.

    diff: ячейки без фичей → NaN (отсутствие данных, не ноль).
    sum/density: ячейки без фичей → 0 (осмысленный ноль).
    """
    gdf = layer.to_crs(TARGET_CRS)
    if spec.attr_filter:
        gdf = gdf.query(_fix_filter_expr(spec.attr_filter))
    joined = ASSIGNERS[spec.spatial_method](gdf, hexes, **spec.spatial_params)
    raw = REDUCERS[spec.math_op](joined, spec, hexes)
    raw = raw.reindex(hexes[HEX_ID])
    # Ноль осмыслен только для аддитивных метрик: в ячейке действительно нет
    # объектов/значений. Для nearest_dist отсутствие объектов означает
    # отсутствие данных, поэтому NaN сохраняется.
    if spec.math_op in {"sum", "density", "count", "count_density"}:
        raw = raw.fillna(0.0)
    raw.name = spec.out_name
    return raw


def _apply_overlay_item_filter(
    layer: gpd.GeoDataFrame,
    attr_filter: str | None,
) -> gpd.GeoDataFrame:
    """Применяет атрибутивное условие к отдельному типу ограничения.

    Для третьей и четвёртой подзон приаэродромной территории используется
    специальное условие ``_mean < 12``: балл начисляется только полигонам, где
    допустимая высота застройки ниже 12 м. Значения ``_mean`` приводятся к
    числу, поэтому строковые и пустые значения не вызывают ложных попаданий.
    Остальные фильтры, если появятся, выполняются через pandas ``query``.
    """
    if not attr_filter:
        return layer

    normalized = re.sub(r"\s+", "", attr_filter).lower()
    if normalized == "_mean<12":
        if "_mean" not in layer.columns:
            raise KeyError(
                "для приаэродромной территории отсутствует обязательное поле '_mean'"
            )
        mean_height = pd.to_numeric(layer["_mean"], errors="coerce")
        return layer.loc[mean_height < 12.0].copy()

    return layer.query(_fix_filter_expr(attr_filter)).copy()


def compute_weighted_overlap_metric(
    spec: MetricSpec,
    hexes: gpd.GeoDataFrame,
    *,
    layer_reader: Callable[[str], gpd.GeoDataFrame] | None = None,
) -> tuple[pd.Series, list[dict]]:
    """Суммирует фиксированные баллы ограничений, пересекающих ячейку.

    Для каждого ``OverlayItemSpec``:
    1. читаются все указанные слои;
    2. определяется множество гексов, пересекающих хотя бы один объект;
    3. фиксированный балл типа ограничения добавляется к этим гексам один раз.

    Если один тип представлен несколькими файлами, повторное пересечение той же
    ячейки не удваивает его балл. Возвращает суммарный сырой показатель и отчёт
    по каждому типу ограничения.
    """
    if spec.math_op != "weighted_overlap_sum":
        raise ValueError(
            f"{spec.out_name}: ожидался math_op=weighted_overlap_sum, "
            f"получено {spec.math_op!r}"
        )
    if not spec.overlay_items:
        raise ValueError(f"{spec.out_name}: не заданы overlay_items")

    reader = layer_reader or _read_layer_source
    target = hexes[[HEX_ID, "geometry"]].copy()
    if target.crs is None:
        raise ValueError("У гексагональной сетки не задана система координат")

    total = pd.Series(
        0.0,
        index=pd.Index(target[HEX_ID].values, name=HEX_ID),
        name=spec.out_name,
    )
    report: list[dict] = []

    for item in spec.overlay_items:
        hit_ids: set = set()
        loaded_paths: list[str] = []
        failed_paths: list[dict[str, str]] = []

        for path in item.layer_paths:
            try:
                layer = reader(path)
                if layer.crs is None:
                    raise ValueError("у слоя не задана CRS")

                layer = layer.to_crs(target.crs)
                layer = _apply_overlay_item_filter(layer, item.attr_filter)
                layer = layer.loc[
                    layer.geometry.notna() & ~layer.geometry.is_empty,
                    ["geometry"],
                ].copy()
                if layer.empty:
                    loaded_paths.append(path)
                    continue

                layer = _clip_to_bbox(layer, target)
                if layer.empty:
                    loaded_paths.append(path)
                    continue

                joined = gpd.sjoin(
                    target,
                    layer[["geometry"]],
                    predicate="intersects",
                    how="inner",
                )
                hit_ids.update(joined[HEX_ID].dropna().tolist())
                loaded_paths.append(path)
            except Exception as exc:
                failed_paths.append({"path": path, "error": str(exc)})

        if hit_ids:
            total.loc[list(hit_ids)] += float(item.score)

        report.append({
            "name": item.name,
            "score": float(item.score),
            "attr_filter": item.attr_filter,
            "hits": len(hit_ids),
            "loaded_paths": loaded_paths,
            "failed_paths": failed_paths,
        })

    return total, report


def compute_isochrone_metric(
    spec: MetricSpec,
    poi_layer: gpd.GeoDataFrame,
    hexes: gpd.GeoDataFrame,
    *,
    timeout: float = 60,
) -> pd.Series:
    """Предзаданная изохронная оценка: строит изохроны через Valhalla и
    скорит каждый гексагон согласно таблице градаций.

    Алгоритм: от НАИБОЛЬШЕЙ изохроны к наименьшей — меньшая перекрывает больший балл.
    Делегирует всю работу с API функции get_iso() из get_iso.py (батчинг,
    ретраи, фоллбек per-point, dissolve).
    """
    import time
    from shapely.ops import unary_union
    from get_iso import get_iso

    if not spec.iso_gradations:
        warnings.warn(f"{spec.out_name}: iso_gradations пустые, возвращаем 0")
        return pd.Series(0.0, index=hexes[HEX_ID].values, name=spec.out_name)

    mode = next((g["mode"] for g in spec.iso_gradations), "pedestrian")

    time_to_score: dict[float, int] = {
        g["max_min"]: g["score"]
        for g in spec.iso_gradations
        if g.get("max_min") is not None
    }
    iso_times = sorted(time_to_score)

    default_score: int = next(
        (g["score"] for g in spec.iso_gradations if g.get("max_min") is None),
        0,
    )

    result = pd.Series(
        float(default_score),
        index=hexes[HEX_ID].values,
        name=spec.out_name,
    )

    centroids = hexes[[HEX_ID, "geometry"]].copy()
    centroids = centroids.set_geometry(hexes.geometry.centroid)

    print(
        f"\n   [{spec.out_name}] СТАРТ: {len(poi_layer)} объектов | "
        f"изохроны {iso_times} мин | mode={mode}",
        flush=True,
    )

    for t in sorted(iso_times, reverse=True):
        t_wall = time.perf_counter()
        print(f"\n   [{spec.out_name}] --- изохрона t={t}мин ---", flush=True)

        iso_gdf = get_iso(
            poi_layer,
            iso_type=mode,
            iso_time=float(t),
            dissolve=True,
            chunk_size=20,
            timeout=timeout,
        )

        wall = time.perf_counter() - t_wall

        if iso_gdf is None or iso_gdf.empty:
            warnings.warn(f"{spec.out_name}: изохрона {t} мин — нет результатов")
            print(f"   [{spec.out_name}] t={t}мин — пустой результат ({wall:.1f}s)", flush=True)
            continue

        iso_gdf = iso_gdf.to_crs(TARGET_CRS)
        iso_union = gpd.GeoDataFrame(
            geometry=[unary_union(iso_gdf.geometry)], crs=TARGET_CRS
        )

        joined = gpd.sjoin(
            centroids, iso_union[["geometry"]], predicate="within", how="inner"
        )
        inside_ids = joined[HEX_ID].unique()
        result.loc[inside_ids] = float(time_to_score[t])
        print(
            f"   [{spec.out_name}] t={t}мин ГОТОВО: "
            f"балл {time_to_score[t]} → {len(inside_ids)} гексов | {wall:.1f}s",
            flush=True,
        )

    print(
        f"\n   [{spec.out_name}] ЗАВЕРШЕНО. "
        f"Баллы: {result.value_counts().to_dict()}",
        flush=True,
    )
    return result


def compute_category_metric(
    spec: MetricSpec,
    layer: gpd.GeoDataFrame,
    hexes: gpd.GeoDataFrame,
) -> pd.Series:
    """Предзаданная категориальная оценка.

    Для каждой ячейки: пространственно привязать объекты слоя, определить
    доминирующую категорию поля source_fields[0] по таблице cat_gradations.
    При равенстве частот берётся наибольший балл. Ячейки без объектов → 0.
    """
    cat_map: dict[str, float] = {g["value"]: float(g["score"]) for g in spec.cat_gradations}
    field = spec.source_fields[0]

    gdf = layer.to_crs(TARGET_CRS)
    if spec.attr_filter:
        gdf = gdf.query(_fix_filter_expr(spec.attr_filter))

    joined = ASSIGNERS[spec.spatial_method](gdf, hexes, **spec.spatial_params)

    def _dominant_score(vals: pd.Series) -> float:
        scores = vals.map(cat_map).dropna()
        if scores.empty:
            return 0.0
        return float(scores.mode().max())  # при ничьей — наибольший балл

    raw = joined.groupby(HEX_ID)[field].apply(_dominant_score)
    raw = raw.reindex(hexes[HEX_ID], fill_value=0.0)
    raw.name = spec.out_name
    return raw


def score_numeric_predefined(
    raw: pd.Series,
    gradations: list[dict],
    *,
    default_score: float = 0.0,
) -> pd.Series:
    """Присваивает баллы числовым значениям по предзаданным интервалам.

    Интервалы трактуются как полуоткрытые, чтобы границы не пересекались:
    ``до 3`` → ``x < 3``; ``от 3 до 8`` → ``3 <= x < 8``;
    ``от 8`` → ``x >= 8``. Значения вне описанных диапазонов получают
    ``default_score``. NaN остаётся NaN.
    """
    result = pd.Series(
        float(default_score),
        index=raw.index,
        name=raw.name + "_score",
        dtype=float,
    )
    result.loc[raw.isna()] = np.nan

    assigned = pd.Series(False, index=raw.index)
    for grad in gradations:
        lower = grad.get("min")
        upper = grad.get("max")
        mask = raw.notna() & ~assigned
        if lower is not None:
            mask &= raw >= float(lower)
        if upper is not None:
            mask &= raw < float(upper)
        result.loc[mask] = float(grad["score"])
        assigned.loc[mask] = True

    return result


# --------------------------------------------------------------------------- #
# Нормировка
# --------------------------------------------------------------------------- #
def normalize_relative(raw: pd.Series, direction: str,  skip_name=False) -> pd.Series:
    """Линейный min-max → [0, 10].
    «Прямая» (direct): больше значение — выше балл.
    «Обратная» (inverse): больше значение — ниже балл.
    """
    valid = raw.dropna()
    if valid.empty:
        name = "test_agg" if skip_name else raw.name + "_score"
        return pd.Series(np.nan, index=raw.index, name=name, dtype=float)
    lo, hi = float(valid.min()), float(valid.max())
    if hi == lo:
        warnings.warn(f"{raw.name}: все ячейки равны ({lo:.4g}); балл = 0")
        return pd.Series(0.0, index=raw.index, name=raw.name + "_score")
    z = (raw - lo) / (hi - lo)
    if direction == "inverse":
        z = 1.0 - z
    out = z * 10.0
    if skip_name==False:
        out.name = raw.name + "_score"
    else:
        out.name = "test_agg"
    return out


# --------------------------------------------------------------------------- #
# Балл критерия из метрик (колонка D)
# --------------------------------------------------------------------------- #
def aggregate_criterion_score(
    scores: list[pd.Series],
    direction: str,
    weights: list[float] | None = None,
) -> pd.Series:
    """Взвешенное среднее баллов метрик → балл критерия с направлением из колонки D.

    NaN-ячейки в отдельных метриках усредняются по доступным (nanmean).
    Если ни одной метрики нет → NaN.
    """
    w = np.array(weights, dtype=float) if weights else np.ones(len(scores), dtype=float)
    w /= w.sum()

    arr = np.column_stack([s.values for s in scores]).astype(float)
    mask = ~np.isnan(arr)
    w_sum = (mask * w).sum(axis=1)
    numerator = np.where(mask, arr * w, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        agg = np.where(w_sum > 0, numerator / w_sum, np.nan)

    result = pd.Series(agg, index=scores[0].index)
    if direction == "inverse":
        result = 10.0 - result
    return result


# --------------------------------------------------------------------------- #
# Свёртка метрик в балл критерия
# --------------------------------------------------------------------------- #
def score_criterion(cspec: CriterionSpec, layers: dict[str, gpd.GeoDataFrame],
                    hexes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = hexes[[HEX_ID, "geometry"]].copy()
    score_cols = []

    for spec in cspec.metrics:
        # --- зональная статистика GeoTIFF ---
        if is_raster_metric(spec):
            if not spec.layer:
                warnings.warn(f"{spec.out_name}: путь к растру не задан, метрика пропущена")
                continue
            raw = compute_raster_metric(spec, spec.layer, hexes)
            out = out.merge(raw.reset_index(), on=HEX_ID, how="left")

            raw_indexed = out.set_index(HEX_ID)[spec.out_name]
            if spec.scale_kind == "predefined":
                if not spec.numeric_gradations:
                    warnings.warn(
                        f"{spec.out_name}: растровая predefined-метрика без "
                        "числовых градаций, пропущена"
                    )
                    continue
                sc = score_numeric_predefined(
                    raw_indexed,
                    spec.numeric_gradations,
                    default_score=float(spec.spatial_params.get("default_score", 0.0)),
                )
            else:
                sc = normalize_relative(raw_indexed, spec.direction)

            out = out.merge(sc.reset_index(), on=HEX_ID, how="left")
            score_cols.append(spec.out_name + "_score")
            continue

        # --- взвешенное наложение ограничений ---
        if spec.math_op == "weighted_overlap_sum":
            raw, _report = compute_weighted_overlap_metric(spec, hexes)
            out = out.merge(raw.reset_index(), on=HEX_ID, how="left")
            sc = normalize_relative(
                out.set_index(HEX_ID)[spec.out_name],
                spec.direction,
            )
            out = out.merge(sc.reset_index(), on=HEX_ID, how="left")
            score_cols.append(spec.out_name + "_score")
            continue

        # --- предзаданная изохронная метрика ---
        if spec.scale_kind == "predefined" and spec.iso_gradations:
            if not spec.layer or spec.layer not in layers:
                warnings.warn(f"{spec.out_name}: слой не загружен, метрика пропущена")
                continue
            raw = compute_isochrone_metric(spec, layers[spec.layer], hexes)
            out = out.merge(raw.reset_index(), on=HEX_ID, how="left")
            # Предзаданный балл уже на шкале 0–10, дублируем как _score
            score_series = out.set_index(HEX_ID)[spec.out_name].rename(spec.out_name + "_score")
            out = out.merge(score_series.reset_index(), on=HEX_ID, how="left")
            score_cols.append(spec.out_name + "_score")
            continue

        # --- предзаданная числовая векторная метрика (в т.ч. nearest_dist) ---
        if spec.scale_kind == "predefined" and spec.numeric_gradations:
            if not spec.layer or spec.layer not in layers:
                warnings.warn(f"{spec.out_name}: слой не загружен, метрика пропущена")
                continue
            raw = compute_metric(spec, layers[spec.layer], hexes)
            out = out.merge(raw.reset_index(), on=HEX_ID, how="left")
            sc = score_numeric_predefined(
                out.set_index(HEX_ID)[spec.out_name],
                spec.numeric_gradations,
                default_score=float(spec.spatial_params.get("default_score", 0.0)),
            )
            out = out.merge(sc.reset_index(), on=HEX_ID, how="left")
            score_cols.append(spec.out_name + "_score")
            continue

        # --- предзаданная категориальная метрика ---
        if spec.scale_kind == "predefined" and spec.cat_gradations:
            if not spec.layer or spec.layer not in layers:
                warnings.warn(f"{spec.out_name}: слой не загружен, метрика пропущена")
                continue
            raw = compute_category_metric(spec, layers[spec.layer], hexes)
            out = out.merge(raw.reset_index(), on=HEX_ID, how="left")
            score_series = out.set_index(HEX_ID)[spec.out_name].rename(spec.out_name + "_score")
            out = out.merge(score_series.reset_index(), on=HEX_ID, how="left")
            score_cols.append(spec.out_name + "_score")
            continue

        # --- предзаданная не-изохронная (нет градаций) — пропускаем ---
        if spec.scale_kind == "predefined":
            warnings.warn(f"{spec.out_name}: predefined без распознанных градаций, пропущено")
            continue

        # --- стандартная относительная метрика ---
        raw = compute_metric(spec, layers[spec.layer], hexes)
        out = out.merge(raw.reset_index(), on=HEX_ID, how="left")

        sc = normalize_relative(out.set_index(HEX_ID)[spec.out_name], spec.direction)
        out = out.merge(sc.reset_index(), on=HEX_ID, how="left")

        score_cols.append(spec.out_name + "_score")

    if not score_cols:
        out[f"crit_{cspec.cid}_score"] = np.nan
        return out

    w = np.array(cspec.weights) if cspec.weights else np.ones(len(score_cols))
    w = w / w.sum()
    crit_raw = pd.Series(
        (out[score_cols].to_numpy() * w).sum(axis=1), index=out.index
    )
    # Направление уже применено при расчёте каждой метрики. Повторная инверсия
    # на уровне критерия дала бы двойной разворот шкалы.
    out[f"crit_{cspec.cid}_score"] = crit_raw
    return out


# --------------------------------------------------------------------------- #
# Загрузка конфига из Критерии.xlsx
# --------------------------------------------------------------------------- #
def _val(x) -> str:
    """Возвращает строку или '' для NaN/пустых значений."""
    s = str(x).strip()
    return '' if s in ('nan', 'None', '') else s


def _resolve_layer_path(value: str, input_data_root: str) -> str:
    """Возвращает путь к слою из Excel в пригодном для чтения виде.

    ``pathlib.Path.is_absolute()`` зависит от ОС, на которой запущен код:
    например, на Linux он не считает ``G:\\...`` абсолютным путём. Поэтому
    Windows-пути Google Drive проверяются отдельно через ``ntpath``. К корню
    ``input_data`` добавляются только действительно относительные значения.
    """
    raw = value.strip().strip('"').strip("'").strip()
    if not raw:
        return ""

    if pathlib.Path(raw).is_absolute():
        return str(pathlib.Path(raw))
    if ntpath.isabs(raw):
        return ntpath.normpath(raw)
    return str(pathlib.Path(input_data_root) / raw)


def _parse_layer_paths(value, input_data_root: str) -> list[str]:
    """Разбирает одну или несколько ссылок на слои из ячейки Excel.

    Поддерживаются отдельные строки, строки в кавычках и разделение точкой с
    запятой. Это нужно, например, когда один тип ЗОУИТ представлен слоями ГП и
    ЕГРН: они образуют один тип ограничения и дают один балл на ячейку.
    """
    text = _val(value)
    if not text:
        return []

    quoted = re.findall(r'"([^"\r\n]+)"', text)
    if quoted:
        raw_paths = quoted
    else:
        raw_paths = [
            part.strip()
            for line in text.splitlines()
            for part in line.split(";")
            if part.strip()
        ]

    paths: list[str] = []
    for raw in raw_paths:
        resolved = _resolve_layer_path(raw, input_data_root)
        if resolved and resolved not in paths:
            paths.append(resolved)
    return paths


def _parse_optional_float(value) -> float | None:
    """Читает числовой балл из Excel, включая запись с запятой."""
    s = _val(value).replace(",", ".")
    if not s or s == "—":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_direction(x) -> str:
    return "inverse" if "Обратная" in _val(x) else "direct"


def _parse_spatial_params(x) -> dict:
    s = _val(x)
    if not s:
        return {}
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return {}


def _parse_scale_kind(x) -> str:
    return "predefined" if "Предзаданный" in _val(x) else "relative"


_ISO_MODE_MAP: dict[str, str] = {
    "пешком":    "pedestrian",
    "ходьб":     "pedestrian",
    "велосипед": "bicycle",
    "авто":      "auto",
    "мультим":   "multimodal",
}

_RE_BELOW  = re.compile(r"До\s+(\d+(?:[.,]\d+)?)\s+минут", re.IGNORECASE)
_RE_RANGE  = re.compile(r"От\s+(\d+(?:[.,]\d+)?)\s+до\s+(\d+(?:[.,]\d+)?)\s+минут", re.IGNORECASE)
_RE_ABOVE  = re.compile(r"От\s+(\d+(?:[.,]\d+)?)\s+минут", re.IGNORECASE)
_NUM_TOKEN = r"[-+]?\d+(?:[.,]\d+)?"
_NUM_UNIT = r"(?:\s*(?:м|метр(?:а|ов)?|км|минут(?:а|ы)?|градус(?:а|ов)?|°|%))?"
_RE_NUM_RANGE = re.compile(
    rf"(?:от\s+)?({_NUM_TOKEN}){_NUM_UNIT}\s+(?:до|[-–—])\s*({_NUM_TOKEN})",
    re.IGNORECASE,
)
_RE_NUM_BELOW = re.compile(
    rf"(?:до|менее|меньше|не\s+более)\s*({_NUM_TOKEN})",
    re.IGNORECASE,
)
_RE_NUM_ABOVE = re.compile(
    rf"(?:от|более|свыше|не\s+менее)\s*({_NUM_TOKEN})",
    re.IGNORECASE,
)


def _parse_iso_grad(text: str) -> dict | None:
    """Разбирает одну строку градации на словарь {max_min, mode}.

    max_min — верхняя граница изохроны в минутах (None = «за пределами»).
    Возвращает None, если текст не соответствует шаблону изохроны.
    """
    t = text.strip()
    if not t or t in ("—", "nan", "None"):
        return None

    mode = "pedestrian"
    for kw, m in _ISO_MODE_MAP.items():
        if kw.lower() in t.lower():
            mode = m
            break

    m2 = _RE_RANGE.match(t)
    if m2:
        return {"max_min": float(m2.group(2).replace(",", ".")), "mode": mode}

    m1 = _RE_BELOW.match(t)
    if m1:
        return {"max_min": float(m1.group(1).replace(",", ".")), "mode": mode}

    m3 = _RE_ABOVE.match(t)
    if m3:
        return {"max_min": None, "mode": mode}

    return None


def _parse_numeric_grad(text: str) -> dict | None:
    """Разбирает числовую градацию вида «до 3», «от 3 до 8», «от 8»."""
    t = " ".join(text.strip().lower().replace("ё", "е").split())
    if not t or t in ("—", "nan", "none"):
        return None

    m_range = _RE_NUM_RANGE.search(t)
    if m_range:
        return {
            "min": float(m_range.group(1).replace(",", ".")),
            "max": float(m_range.group(2).replace(",", ".")),
        }

    m_below = _RE_NUM_BELOW.search(t)
    if m_below:
        return {
            "min": None,
            "max": float(m_below.group(1).replace(",", ".")),
        }

    m_above = _RE_NUM_ABOVE.search(t)
    if m_above:
        return {
            "min": float(m_above.group(1).replace(",", ".")),
            "max": None,
        }

    return None


def _append_predefined_gradation(
    spec: MetricSpec,
    grad_text: str,
    score: float,
) -> None:
    """Классифицирует строку градации как изохронную, числовую или категориальную."""
    parsed_iso = _parse_iso_grad(grad_text)
    if parsed_iso is not None:
        parsed_iso["score"] = float(score)
        spec.iso_gradations.append(parsed_iso)
        return

    # Числовые интервалы используются не только для растров, но и для
    # nearest_dist и других количественных метрик. Изохронные формулировки
    # уже отсеяны выше, поэтому «Свыше 1000 м» корректно станет интервалом,
    # а текстовая категория останется категорией.
    parsed_numeric = _parse_numeric_grad(grad_text)
    if parsed_numeric is not None:
        parsed_numeric["score"] = float(score)
        parsed_numeric["label"] = grad_text
        spec.numeric_gradations.append(parsed_numeric)
        return

    spec.cat_gradations.append({"value": grad_text, "score": float(score)})


def load_criteria(
    xlsx_path: str,
    input_data_root: str = "input_data",
    sheet_name: str = DEFAULT_CRITERIA_SHEET,
) -> list[CriterionSpec]:
    """Читает критерии из выбранного листа книги.

    Обрабатывает четыре вида метрик:
    - Относительный: строка с заполненным out_name → стандартный pipeline.
    - Предзаданный/изохронный: строка с pp и Предзаданный, за которой идут
      строки-градации (без pp, без out_name) с текстом вида «До 5 минут пешком».
    - Растровый: ``spatial_method=raster_zonal`` и один из ``raster_*`` math_op;
      для predefined-шкалы числовые интервалы читаются из градаций Excel.
    - Взвешенное наложение: строка с ``math_op=weighted_overlap_sum``, после
      которой идут дочерние строки с путём к слою в колонке N и баллом в L.
    - Расстояние до ближайшего объекта: ``centroid_distance/nearest_dist``;
      предзаданные интервалы в метрах переводятся в баллы без min-max.
    Путь в колонке ``layer`` может быть как файлом, так и папкой с несколькими
    векторными слоями.
    """
    sheet_name = normalize_criteria_sheet(sheet_name)
    df_raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=0)

    # Находим строку-заголовок (там, где первый столбец == '№')
    mask = df_raw.iloc[:, 0].astype(str).str.strip() == "№"
    if not mask.any():
        raise ValueError(
            f"Строка-заголовок '№' не найдена в листе {sheet_name!r} файла {xlsx_path}"
        )
    header_row = int(df_raw[mask].index[0])

    # Оставляем только строки после заголовка
    df = df_raw.iloc[header_row + 1:].reset_index(drop=True)

    # Форвард-филл колонок критерия (значения только в первой строке блока)
    for ci in (_C_CRIT_NUM, _C_CRIT_NAME, _C_CRIT_DIR):
        df.iloc[:, ci] = df.iloc[:, ci].ffill()

    criteria: dict[int, CriterionSpec] = {}
    current_iso_spec: MetricSpec | None = None  # открытая предзаданная метрика
    current_overlay_spec: MetricSpec | None = None  # блок взвешенного наложения

    for _, row in df.iterrows():
        out_name = _val(row.iloc[_C_OUT_NAME])
        pp       = _val(row.iloc[_C_PP])
        scale_str = _val(row.iloc[_C_SCALE_KIND])
        grad_text = _val(row.iloc[_C_GRAD_TEXT])

        # --- дочерняя строка weighted_overlap_sum ---
        # Блок продолжается до следующей строки свойства (pp) или новой метрики
        # (out_name). Групповые заголовки без слоя просто пропускаются.
        if current_overlay_spec is not None and not out_name and not pp:
            layer_paths = _parse_layer_paths(row.iloc[_C_LAYER], input_data_root)
            fixed_score = _parse_optional_float(row.iloc[_C_GRAD_SCORE])
            if layer_paths and fixed_score is not None:
                item_name = _val(row.iloc[6]) or f"Ограничение {len(current_overlay_spec.overlay_items) + 1}"
                normalized_item_name = " ".join(
                    item_name.lower().replace("ё", "е").split()
                )
                attr_filter = None
                if current_overlay_spec.pp == "6.2" and (
                    normalized_item_name.startswith(
                        "третья подзона приаэродромной территории"
                    )
                    or normalized_item_name.startswith(
                        "четвертая подзона приаэродромной территории"
                    )
                ):
                    # Методологическое исключение для критерия 6.2:
                    # учитываются только участки с ограничением высоты < 12 м.
                    attr_filter = "_mean < 12"

                current_overlay_spec.overlay_items.append(
                    OverlayItemSpec(
                        name=item_name,
                        score=fixed_score,
                        layer_paths=layer_paths,
                        attr_filter=attr_filter,
                    )
                )
            continue

        # --- строка-градация: продолжение предзаданного блока ---
        if not out_name and not pp and grad_text and grad_text != "—":
            if current_iso_spec is not None:
                try:
                    score_val = int(float(row.iloc[_C_GRAD_SCORE]))
                except (ValueError, TypeError):
                    score_val = 0
                _append_predefined_gradation(current_iso_spec, grad_text, score_val)
            continue

        # --- любая строка с pp или out_name закрывает открытые дочерние блоки ---
        current_iso_spec = None
        current_overlay_spec = None

        # --- метрика (есть out_name) ---
        if out_name:
            try:
                cid = int(float(row.iloc[_C_CRIT_NUM]))
            except (ValueError, TypeError):
                continue
            _ensure_crit(criteria, cid, row)

            sf1 = _val(row.iloc[_C_SF1])
            sf2 = _val(row.iloc[_C_SF2])
            layer_val = _val(row.iloc[_C_LAYER])
            layer_path = _resolve_layer_path(layer_val, input_data_root) if layer_val else ""
            math_op = _val(row.iloc[_C_MATH_OP]) or "sum"

            scale_kind = _parse_scale_kind(row.iloc[_C_SCALE_KIND])
            spec = MetricSpec(
                pp=pp,
                layer=layer_path,
                spatial_method=_val(row.iloc[_C_SPAT_METH]) or "within",
                spatial_params=_parse_spatial_params(row.iloc[_C_SPAT_PARAMS]),
                math_op=math_op,
                source_fields=[f for f in [sf1, sf2] if f],
                scale_kind=scale_kind,
                direction=_parse_direction(row.iloc[_C_METRIC_DIR]),
                attr_filter=_val(row.iloc[_C_ATTR_FILT]) or None,
                out_name=out_name,
            )
            criteria[cid].metrics.append(spec)

            if math_op == "weighted_overlap_sum":
                current_overlay_spec = spec
                continue

            # Предзаданная метрика — начинаем сбор строк-градаций
            if scale_kind == "predefined":
                if grad_text and grad_text != "—":
                    try:
                        score_val = int(float(row.iloc[_C_GRAD_SCORE]))
                    except (ValueError, TypeError):
                        score_val = 0
                    _append_predefined_gradation(spec, grad_text, score_val)
                current_iso_spec = spec

            continue

        # --- метрика с pp, но без out_name ---
        # Такие строки встречаются, в частности, в критерии 15 инвестора.
        # Для них формируем стабильное имя автоматически. Изохронные шкалы
        # распознаются по минутам, числовые шкалы расстояния — как nearest_dist.
        if pp:
            try:
                cid = int(float(row.iloc[_C_CRIT_NUM]))
            except (ValueError, TypeError):
                continue

            layer_val = _val(row.iloc[_C_LAYER])
            layer_path = _resolve_layer_path(layer_val, input_data_root) if layer_val else ""
            explicit_spatial_method = _val(row.iloc[_C_SPAT_METH])
            explicit_math_op = _val(row.iloc[_C_MATH_OP])
            scale_kind = _parse_scale_kind(row.iloc[_C_SCALE_KIND])
            iso_first_grad = _parse_iso_grad(grad_text) if grad_text else None
            numeric_first_grad = _parse_numeric_grad(grad_text) if grad_text else None

            # Относительные строки без машинных настроек раньше считались
            # незаполненными. Исключение — критерий 15, где сама методология
            # задаёт расстояние до ближайшего объекта.
            should_infer_nearest = (
                cid == 15
                or (scale_kind == "predefined" and numeric_first_grad is not None)
            )
            if (
                not layer_path
                or (
                    scale_kind != "predefined"
                    and not explicit_spatial_method
                    and not explicit_math_op
                    and not should_infer_nearest
                )
            ):
                continue

            _ensure_crit(criteria, cid, row)
            sf1 = _val(row.iloc[_C_SF1])
            sf2 = _val(row.iloc[_C_SF2])

            if explicit_spatial_method or explicit_math_op:
                spatial_method = explicit_spatial_method or "within"
                math_op = explicit_math_op or "sum"
            elif iso_first_grad is not None:
                spatial_method = "isochrone"
                math_op = "isochrone"
            elif should_infer_nearest:
                spatial_method = "centroid_distance"
                math_op = "nearest_dist"
            else:
                # Обратная совместимость для прежних предзаданных строк,
                # которые в книге задавались как изохроны без машинных колонок.
                spatial_method = "isochrone"
                math_op = "isochrone"

            spec = MetricSpec(
                pp=pp,
                layer=layer_path,
                spatial_method=spatial_method,
                spatial_params=_parse_spatial_params(row.iloc[_C_SPAT_PARAMS]),
                math_op=math_op,
                source_fields=[f for f in [sf1, sf2] if f],
                scale_kind=scale_kind,
                direction=_parse_direction(row.iloc[_C_METRIC_DIR]),
                attr_filter=_val(row.iloc[_C_ATTR_FILT]) or None,
                out_name="m_" + pp.replace(".", "_"),
            )

            # Первая строка предзаданного блока тоже может содержать градацию.
            if scale_kind == "predefined" and grad_text and grad_text != "—":
                try:
                    score_val = int(float(row.iloc[_C_GRAD_SCORE]))
                except (ValueError, TypeError):
                    score_val = 0
                _append_predefined_gradation(spec, grad_text, score_val)

            criteria[cid].metrics.append(spec)
            if scale_kind == "predefined":
                current_iso_spec = spec

    return list(criteria.values())


def _ensure_crit(criteria: dict, cid: int, row) -> None:
    """Создаёт CriterionSpec если ещё нет."""
    if cid not in criteria:
        criteria[cid] = CriterionSpec(
            cid=cid,
            name=_val(row.iloc[_C_CRIT_NAME]),
            direction=_parse_direction(row.iloc[_C_CRIT_DIR]),
            metrics=[],
        )


# --------------------------------------------------------------------------- #
# Утилиты запуска
# --------------------------------------------------------------------------- #
def load_run_config(path: str = "run_config.txt") -> RunConfig:
    """Читает run_config.txt: лист, список критериев и размер гекс-сетки."""
    cfg = pathlib.Path(path)
    criteria: list[int] | None = None
    h = 400.0
    v = 400.0
    criteria_sheet = DEFAULT_CRITERIA_SHEET
    if cfg.exists():
        for raw_line in cfg.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#")[0].strip()
            if not line or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip().lower()
            val = val.strip()
            if key == "criteria" and val:
                criteria = [int(x.strip()) for x in val.split(",") if x.strip()]
            elif key == "hex_h_spacing" and val:
                h = float(val)
            elif key == "hex_v_spacing" and val:
                v = float(val)
            elif key in ("criteria_sheet", "sheet") and val:
                criteria_sheet = normalize_criteria_sheet(val)
    return RunConfig(
        criteria=criteria,
        hex_h_spacing=h,
        hex_v_spacing=v,
        criteria_sheet=criteria_sheet,
    )


def load_hexes(path: str) -> gpd.GeoDataFrame:
    """Читает сетку и нормализует идентификатор ячейки в hex_id."""
    hexes = gpd.read_file(path).to_crs(TARGET_CRS)
    if HEX_ID not in hexes.columns:
        col = "id" if "id" in hexes.columns else None
        hexes = hexes.rename(columns={col: HEX_ID}) if col else hexes.assign(
            **{HEX_ID: range(len(hexes))}
        )
    return hexes


_SQRT3 = 3 ** 0.5


def build_hex_grid(
    bound_path: str,
    h_spacing: float,
    v_spacing: float,
) -> gpd.GeoDataFrame:
    """Строит сетку плоско-верхних (flat-top) гексагонов по слою границы.

    h_spacing = v_spacing = D → правильный шестиугольник с flat-to-flat = D м.
    D = расстояние между центрами соседних ячеек = диаметр вписанной окружности.
    Ячейки фильтруются через Extract by Location (intersects) — clip не применяется.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    bound = gpd.read_file(bound_path).to_crs(TARGET_CRS)
    bound_union = unary_union(bound.geometry.values)
    xmin, ymin, xmax, ymax = bound_union.bounds

    # r_h — горизонтальный описанный радиус (центр → правая вершина)
    r_h = h_spacing / _SQRT3
    col_stride = 1.5 * r_h    # горизонтальный шаг между центрами колонок
    row_stride = v_spacing     # вертикальный шаг (= flat-to-flat по вертикали)

    xmin -= h_spacing
    ymin -= v_spacing
    xmax += h_spacing
    ymax += v_spacing

    polygons: list = []
    col = 0
    cx = xmin
    while cx <= xmax:
        y_offset = (col % 2) * (v_spacing / 2.0)
        cy = ymin + y_offset
        while cy <= ymax:
            # Вершины flat-top гексагона. При h_spacing = v_spacing — правильный.
            pts = [
                (cx + r_h,          cy),
                (cx + r_h / 2,  cy + v_spacing / 2),
                (cx - r_h / 2,  cy + v_spacing / 2),
                (cx - r_h,          cy),
                (cx - r_h / 2,  cy - v_spacing / 2),
                (cx + r_h / 2,  cy - v_spacing / 2),
            ]
            polygons.append(Polygon(pts))
            cy += row_stride
        cx += col_stride
        col += 1

    grid = gpd.GeoDataFrame(
        {HEX_ID: range(len(polygons)), "geometry": polygons},
        crs=TARGET_CRS,
    )

    # Extract by location: оставляем только ячейки, пересекающие границу
    grid = grid[grid.geometry.intersects(bound_union)].reset_index(drop=True)
    grid[HEX_ID] = range(len(grid))
    return grid


def run_criterion(cspec: CriterionSpec, hexes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Запускает критерий, загружая все слои с диска по путям из MetricSpec."""
    layers: dict[str, gpd.GeoDataFrame] = {}
    for spec in cspec.metrics:
        if (
            spec.layer
            and not is_raster_metric(spec)
            and spec.layer not in layers
        ):
            layers[spec.layer] = _read_layer_source(spec.layer)
    hexes = hexes.to_crs(TARGET_CRS).copy()
    if HEX_ID not in hexes.columns:
        hexes[HEX_ID] = range(len(hexes))
    return score_criterion(cspec, layers, hexes)
