"""Тест метрик на реальных данных.

Грид:     01_vector data/grid_400.gpkg
Критерии: Критерии.xlsx

Правила:
- Метрика без source_field_1 -> SKIP (данные не готовы)
- Метрика без layer          -> SKIP (слой не указан)
- Ошибка загрузки/вычисления -> ERR (падает с сообщением, не прерывает прогон)
- Relative шкала: балл должен быть в [0, 10]
- diff: ячейки без фичей → NaN (не 0)
- sum/density: ячейки без фичей → 0
"""
import pathlib
import warnings
import crt_pipeline as P
import generate_map

BOUND_PATH   = r"01_vector data\bound_chita.gpkg"
WATER_PATH   = pathlib.Path("01_vector data/natural_water.gpkg")
XLSX_PATH    = "Критерии.xlsx"
RUN_CONFIG   = "run_config.txt"
RESULTS_DIR  = pathlib.Path("results")

def remove_water_hexes(hexes):
    """Удаляет ячейки, центр которых попадает в полигон водной поверхности."""

    if not WATER_PATH.exists():
        raise FileNotFoundError(
            f"Не найден слой водных объектов: {WATER_PATH}"
        )

    water = P._read_layer(str(WATER_PATH))

    if water.crs is None:
        raise ValueError(
            f"У слоя {WATER_PATH} не задана система координат"
        )

    water = water.to_crs(hexes.crs)

    # Убираем пустые геометрии и оставляем только геометрию.
    water = water.loc[
        water.geometry.notna() & ~water.geometry.is_empty,
        ["geometry"],
    ].copy()

    # Определяем принадлежность к воде по центру гексагона.
    # Так береговые ячейки не удаляются из-за минимального пересечения с водой.
    hex_centers = hexes[["hex_id", "geometry"]].copy()
    hex_centers["geometry"] = hexes.geometry.centroid

    water_hits = hex_centers.sjoin(
        water,
        how="inner",
        predicate="intersects",
    )

    water_hex_ids = water_hits["hex_id"].unique()

    filtered = hexes.loc[
        ~hexes["hex_id"].isin(water_hex_ids)
    ].copy()

    print(
        f"Водная маска: удалено {len(hexes) - len(filtered)} "
        f"из {len(hexes)} гексов"
    )

    return filtered

# --- загрузка базовых данных ---
run_cfg = P.load_run_config(RUN_CONFIG)
profile_key, profile_label = P.profile_from_sheet(run_cfg.criteria_sheet)
out_gpkg = RESULTS_DIR / f"krt_scores_{profile_key}.gpkg"

all_criteria = P.load_criteria(
    XLSX_PATH,
    sheet_name=run_cfg.criteria_sheet,
)

print(
    f"Профиль: {profile_label} | лист: {run_cfg.criteria_sheet} | "
    f"результат: {out_gpkg}"
)

hexes = P.build_hex_grid(
    BOUND_PATH,
    run_cfg.hex_h_spacing,
    run_cfg.hex_v_spacing,
)

hexes = remove_water_hexes(hexes)
print(f"Сетка: {len(hexes)} ячеек, EPSG:{hexes.crs.to_epsg()}")
print(f"Размер ячейки: {run_cfg.hex_h_spacing} × {run_cfg.hex_v_spacing} м")

if run_cfg.criteria is not None:
    criteria = [c for c in all_criteria if c.cid in run_cfg.criteria]
    print(f"Критериев загружено: {len(all_criteria)} | запускаем: {run_cfg.criteria}\n")
else:
    criteria = all_criteria
    print(f"Критериев загружено: {len(criteria)} | запускаем все\n")

passed = skipped = errors = 0

# Финальный слой: начинаем с сетки, добавляем колонки по мере вычисления
result = hexes.copy()

for cspec in criteria:
    print(f"--- Критерий {cspec.cid} [{cspec.direction}]: {cspec.name}")

    criterion_scores: list = []   # баллы метрик для агрегации в балл критерия

    for spec in cspec.metrics:
        prefix = f"   {spec.pp}"

        # --- зональная статистика GeoTIFF ---
        if P.is_raster_metric(spec):
            if not spec.layer:
                print(f"{prefix}  SKIP: путь к растру не заполнен")
                skipped += 1
                continue
            if spec.scale_kind == "predefined" and not spec.numeric_gradations:
                print(f"{prefix}  SKIP: predefined-растр без числовых градаций")
                skipped += 1
                continue

            try:
                raw = P.compute_raster_metric(spec, spec.layer, hexes)
            except Exception as e:
                print(f"{prefix}  ERR:  ошибка растровой метрики — {e}")
                errors += 1
                continue

            result[spec.out_name] = raw.values
            if spec.scale_kind == "predefined":
                score = P.score_numeric_predefined(
                    raw,
                    spec.numeric_gradations,
                    default_score=float(spec.spatial_params.get("default_score", 0.0)),
                )
            else:
                score = P.normalize_relative(raw, spec.direction)

            result[spec.out_name + "_score"] = score.values
            criterion_scores.append(score)

            valid_raw = raw.dropna()
            valid_score = score.dropna()
            null_count = int(raw.isna().sum())
            null_note = f" | NULL={null_count}" if null_count else ""
            if valid_raw.empty:
                raw_note = "raw=[нет данных]"
                score_note = "score=[нет данных]"
            else:
                raw_note = f"raw=[{valid_raw.min():.4g}, {valid_raw.max():.4g}]"
                score_note = (
                    f"score=[{valid_score.min():.2f}, {valid_score.max():.2f}]"
                    if not valid_score.empty else "score=[нет данных]"
                )
            grad_note = ""
            if spec.numeric_gradations:
                grad_note = " | grades=" + "; ".join(
                    f"{g.get('label', '')}→{g['score']:g}"
                    for g in spec.numeric_gradations
                )
            print(
                f"{prefix}  OK    {spec.math_op}/{spec.scale_kind} | "
                f"{raw_note} | {score_note}{null_note}{grad_note}"
            )
            passed += 1
            continue

        # --- взвешенное наложение пространственных ограничений ---
        if spec.math_op == "weighted_overlap_sum":
            if not spec.overlay_items:
                print(f"{prefix}  SKIP: не заданы дочерние строки ограничений")
                skipped += 1
                continue

            try:
                raw, overlay_report = P.compute_weighted_overlap_metric(spec, hexes)
            except Exception as e:
                print(f"{prefix}  ERR:  ошибка взвешенного наложения — {e}")
                errors += 1
                continue

            result[spec.out_name] = raw.values
            score = P.normalize_relative(raw, spec.direction)
            result[spec.out_name + "_score"] = score.values
            criterion_scores.append(score)

            failed_count = 0
            for item in overlay_report:
                loaded = len(item["loaded_paths"])
                failed = len(item["failed_paths"])
                failed_count += failed
                status = "OK" if failed == 0 else "WARN"
                filter_note = (
                    f" | фильтр={item['attr_filter']}"
                    if item.get("attr_filter")
                    else ""
                )
                print(
                    f"      {status} +{item['score']:g} | {item['name']} | "
                    f"гексов={item['hits']} | слоёв={loaded}/{loaded + failed}"
                    f"{filter_note}"
                )
                for failure in item["failed_paths"]:
                    print(
                        f"           не загружен: {failure['path']} — "
                        f"{failure['error']}"
                    )

            errors += failed_count
            valid = score.dropna()
            print(
                f"{prefix}  OK    weighted_overlap_sum/{spec.direction} | "
                f"types={len(spec.overlay_items)} | "
                f"raw=[{raw.min():.4g}, {raw.max():.4g}] | "
                f"score=[{valid.min():.2f}, {valid.max():.2f}]"
            )
            passed += 1
            continue

        # --- предзаданная изохронная метрика ---
        if spec.scale_kind == "predefined" and spec.iso_gradations:
            if not spec.layer:
                print(f"{prefix}  SKIP: layer не заполнен (изохрона)")
                skipped += 1
                continue

            try:
                poi_layer = P._read_layer_source(spec.layer)
            except Exception as e:
                print(f"{prefix}  ERR:  слой ПОИ не загружен — {e}")
                errors += 1
                continue

            try:
                score = P.compute_isochrone_metric(spec, poi_layer, hexes)
            except Exception as e:
                print(f"{prefix}  ERR:  ошибка изохрон — {e}")
                errors += 1
                continue

            result[spec.out_name] = score.values
            result[spec.out_name + "_score"] = score.values
            criterion_scores.append(score.rename(spec.out_name + "_score"))
            valid = score.dropna()
            times = sorted(
                g["max_min"] for g in spec.iso_gradations if g.get("max_min") is not None
            )
            print(
                f"{prefix}  OK    isochrone/predefined | times={times} | "
                f"score=[{valid.min():.0f}, {valid.max():.0f}] | "
                f"unique={sorted(valid.unique())}"
            )
            passed += 1
            continue

        # --- предзаданная числовая векторная метрика (nearest_dist и др.) ---
        if spec.scale_kind == "predefined" and spec.numeric_gradations:
            if not spec.layer:
                print(f"{prefix}  SKIP: layer не заполнен (числовая predefined)")
                skipped += 1
                continue

            try:
                numeric_layer = P._read_layer_source(spec.layer)
            except Exception as e:
                print(f"{prefix}  ERR:  слой не загружен — {e}")
                errors += 1
                continue

            try:
                raw = P.compute_metric(spec, numeric_layer, hexes)
                score = P.score_numeric_predefined(
                    raw,
                    spec.numeric_gradations,
                    default_score=float(spec.spatial_params.get("default_score", 0.0)),
                )
            except Exception as e:
                print(f"{prefix}  ERR:  ошибка числовой predefined-метрики — {e}")
                errors += 1
                continue

            result[spec.out_name] = raw.values
            result[spec.out_name + "_score"] = score.values
            criterion_scores.append(score)

            valid_raw = raw.dropna()
            valid_score = score.dropna()
            null_count = int(raw.isna().sum())
            null_note = f" | NULL={null_count}" if null_count else ""
            grad_note = "; ".join(
                f"{g.get('label', '')}→{g['score']:g}"
                for g in spec.numeric_gradations
            )
            raw_note = (
                f"raw=[{valid_raw.min():.4g}, {valid_raw.max():.4g}]"
                if not valid_raw.empty else "raw=[нет данных]"
            )
            score_note = (
                f"score=[{valid_score.min():.2f}, {valid_score.max():.2f}]"
                if not valid_score.empty else "score=[нет данных]"
            )
            print(
                f"{prefix}  OK    {spec.math_op}/predefined | "
                f"{raw_note} | {score_note}{null_note} | grades={grad_note}"
            )
            passed += 1
            continue

        # --- предзаданная категориальная метрика ---
        if spec.scale_kind == "predefined" and spec.cat_gradations:
            if not spec.layer:
                print(f"{prefix}  SKIP: layer не заполнен (категориальная)")
                skipped += 1
                continue
            if not spec.source_fields:
                print(f"{prefix}  SKIP: source_field_1 не заполнен (категориальная)")
                skipped += 1
                continue

            try:
                cat_layer = P._read_layer_source(spec.layer)
            except Exception as e:
                print(f"{prefix}  ERR:  слой не загружен — {e}")
                errors += 1
                continue

            try:
                score = P.compute_category_metric(spec, cat_layer, hexes)
            except Exception as e:
                print(f"{prefix}  ERR:  ошибка категориальной метрики — {e}")
                errors += 1
                continue

            result[spec.out_name] = score.values
            result[spec.out_name + "_score"] = score.values
            criterion_scores.append(score.rename(spec.out_name + "_score"))
            valid = score.dropna()
            cats = [g["value"][:30] for g in spec.cat_gradations]
            print(
                f"{prefix}  OK    category/predefined | field={spec.source_fields[0]} | "
                f"cats={cats} | score=[{valid.min():.0f}, {valid.max():.0f}] | "
                f"unique={sorted(valid.unique())}"
            )
            passed += 1
            continue

        # --- предзаданная без градаций — пропускаем ---
        if spec.scale_kind == "predefined":
            print(f"{prefix}  SKIP: predefined без градаций (нет iso/numeric/cat_gradations)")
            skipped += 1
            continue

        # --- стандартная относительная метрика ---
        needs_field = spec.math_op not in ("count", "count_density", "nearest_dist")
        if needs_field and not spec.source_fields:
            print(f"{prefix}  SKIP: source_field_1 не заполнен")
            skipped += 1
            continue

        if not spec.layer:
            print(f"{prefix}  SKIP: layer не заполнен")
            skipped += 1
            continue

        # загрузка слоя
        try:
            layer = P._read_layer_source(spec.layer)
        except Exception as e:
            print(f"{prefix}  ERR:  не удалось загрузить слой — {e}")
            errors += 1
            continue

        # вычисление метрики
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                raw = P.compute_metric(spec, layer, hexes)
        except Exception as e:
            print(f"{prefix}  ERR:  ошибка вычисления — {e}")
            errors += 1
            continue

        null_count = raw.isna().sum()
        result[spec.out_name] = raw.values

        score = P.normalize_relative(raw, spec.direction)
        valid = score.dropna()
        assert valid.between(0, 10).all(), (
            f"{spec.pp}: балл вне [0,10]: min={valid.min():.3f} max={valid.max():.3f}"
        )
        result[spec.out_name + "_score"] = score.values
        criterion_scores.append(score)
        null_note = f" | NULL={null_count}" if null_count else ""
        warn_note = f"  [все ячейки = {raw.min():.4g}]" if caught else ""
        print(
            f"{prefix}  OK    {spec.math_op}/{spec.direction} | "
            f"raw=[{raw.min():.4g}, {raw.max():.4g}] | "
            f"score=[{valid.min():.2f}, {valid.max():.2f}]{null_note}{warn_note}"
        )
        passed += 1

    # --- балл критерия (колонка D) ---
    if criterion_scores:
        # Направление уже применено внутри каждой метрики. Повторная передача
        # cspec.direction здесь инвертировала бы обратные критерии второй раз.
        crit_score = P.aggregate_criterion_score(
            criterion_scores, "direct", cspec.weights
        )
        crit_score = P.normalize_relative(crit_score, 'direct', skip_name=True)
        result[f"crit_{cspec.cid}_score"] = crit_score.values
        valid_cs = crit_score.dropna()
        null_c = crit_score.isna().sum()
        null_note = f" | NULL={null_c}" if null_c else ""
        print(
            f"   → балл критерия {cspec.cid} [{cspec.direction}]: "
            f"[{valid_cs.min():.2f}, {valid_cs.max():.2f}]{null_note}"
        )

print(f"\n{'=' * 55}")
print(f"OK: {passed}   SKIP: {skipped}   ERR: {errors}")

# --- сохранение финального слоя ---
if passed > 0:
    RESULTS_DIR.mkdir(exist_ok=True)
    if out_gpkg.exists():
        out_gpkg.unlink()
    result.to_file(out_gpkg, driver="GPKG", layer="krt_scores")
    print(f"\nСлой сохранён: {out_gpkg}  ({passed} метрик, {len(result)} ячеек)")
    print()
    generate_map.main(
        results_gpkg=out_gpkg,
        criteria_sheet=run_cfg.criteria_sheet,
        profile_key=profile_key,
        profile_label=profile_label,
    )
