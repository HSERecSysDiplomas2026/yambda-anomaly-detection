"""
Поиск пользователей с «нечеловеческим» поведением в Yambda-50M.

Для каждого пользователя считаем набор признаков, по каждому выводим
самых крайних, а в конце — сводный рейтинг: у кого больше всего признаков
сильно выбиваются из нормы.

Запуск из корня проекта:
    python src/bot_hunt.py > bot_hunt_report.txt
"""

from pathlib import Path

import numpy as np
import polars as pl

DATA = Path(__file__).resolve().parent.parent / "data"
if not (DATA / "listens.parquet").exists():
    raise SystemExit("Нет данных в data/. Сначала запустите: python src/download_data.py")

MIN_LISTENS = 200        # пользователей с меньшим числом прослушиваний не рассматриваем
SESSION_GAP = 30 * 60    # пауза больше 30 минут = начало новой сессии
TOP = 10                 # сколько крайних пользователей показывать по каждому признаку
Z_FLAG = 5               # с какого робастного z-score признак считается «сильно выбивающимся»

# настройки вывода таблиц в отчет, чтобы ничего не обрезалось
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)
pl.Config.set_fmt_str_lengths(40)


def section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def show(df: pl.DataFrame, col: str, descending: bool = True) -> None:
    """Распределение признака и самые крайние пользователи."""
    print(f"\nРаспределение «{col}»:")
    print(df[col].describe())
    print(f"\nТоп-{TOP} по «{col}»:")
    print(df.sort(col, descending=descending, nulls_last=True).head(TOP))


# ---------------------------------------------------------------------------
# Загрузка и подготовка
# ---------------------------------------------------------------------------
listens = (
    pl.read_parquet(DATA / "listens.parquet")
    .with_columns(pl.col("timestamp").cast(pl.Int64))
    .sort(["uid", "timestamp"])
    # новые колонки
    .with_columns(
        (pl.col("track_length_seconds") * pl.col("played_ratio_pct") / 100).alias("played_sec"),
        ((pl.col("timestamp") % 86400) // 3600).alias("hour"),
        (pl.col("timestamp") // 86400).alias("day"),
        pl.col("timestamp").diff().over("uid").alias("gap"),
    )
    .with_columns(
        # номер сессии: 1 там, где начинается новая сессия (первое событие или пауза > 30 мин),
        # накопленная сумма дает всем событиям одной сессии одинаковый номер
        (pl.col("gap").is_null() | (pl.col("gap") > SESSION_GAP))
        .cast(pl.Int32).cum_sum().over("uid").alias("session"),
        # номер серии: так же, но новая серия начинается, когда сменился трек
        (pl.col("item_id") != pl.col("item_id").shift(1).over("uid"))
        .fill_null(True).cast(pl.Int32).cum_sum().over("uid").alias("run"),
    )
)
active = (
    listens.group_by("uid").len().rename({"len": "n_listens"})
    .filter(pl.col("n_listens") >= MIN_LISTENS)
)
listens = listens.join(active.select("uid"), on="uid")
print(f"Пользователей с {MIN_LISTENS}+ прослушиваниями: {active.height}")

# ---------------------------------------------------------------------------
# 1. Нет «времени сна» (не зависит от часового пояса)
# ---------------------------------------------------------------------------
section("1. Доля прослушиваний в самом тихом 3-часовом окне пользователя\n"
        "   (у человека в любом часовом поясе мала; у бота 24/7 около 0,125)")

uh = (
    listens.group_by(["uid", "hour"]).len()
    .pivot(on="hour", index="uid", values="len")
    .fill_null(0)
)
hour_cols = [str(h) for h in range(24) if str(h) in uh.columns]
counts = uh.select(hour_cols).to_numpy().astype(float)
shares = counts / counts.sum(axis=1, keepdims=True)
windows = sum(np.roll(shares, -k, axis=1) for k in range(3))
quiet = pl.DataFrame({
    "uid": uh["uid"],
    "quiet_3h_share": windows.min(axis=1),
    "active_hours": (counts > 0).sum(axis=1),
})
show(quiet, "quiet_3h_share")

# ---------------------------------------------------------------------------
# 2. Очень длинные непрерывные сессии и очень много музыки в сутки
# ---------------------------------------------------------------------------
section("2. Длинные сессии без пауз больше 30 минут и часы прослушивания в сутки")

sessions = listens.group_by(["uid", "session"]).agg(
    ((pl.col("timestamp").max() - pl.col("timestamp").min()
      + pl.col("played_sec").last()) / 3600).alias("session_hours"),
)
sess = sessions.group_by("uid").agg(
    pl.col("session_hours").max().alias("max_session_h"),
    (pl.col("session_hours") > 12).sum().alias("sessions_over_12h"),
)
daily = listens.group_by(["uid", "day"]).agg((pl.col("played_sec").sum() / 3600).alias("hours"))
day_stats = daily.group_by("uid").agg(
    pl.col("hours").max().alias("max_hours_per_day"),
    (pl.col("hours") > 16).sum().alias("days_over_16h"),
)
show(sess, "max_session_h")
show(day_stats, "max_hours_per_day")

# ---------------------------------------------------------------------------
# 3. Слишком регулярные паузы между прослушиваниями
# ---------------------------------------------------------------------------
section("3. Регулярность пауз внутри сессий\n"
        "   (mode_gap_share — доля пауз, равных самой частой паузе пользователя)")

gaps = listens.filter(pl.col("gap").is_between(1, SESSION_GAP))
gap_mode = (
    gaps.group_by(["uid", "gap"]).len()
    .sort("len", descending=True)
    .group_by("uid").agg(
        pl.col("gap").first().alias("mode_gap_sec"),
        pl.col("len").first().alias("mode_gap_count"),
        pl.col("len").sum().alias("n_gaps"),
    )
    .with_columns((pl.col("mode_gap_count") / pl.col("n_gaps")).alias("mode_gap_share"))
)
gap_cv = gaps.group_by("uid").agg(
    (pl.col("gap").std() / pl.col("gap").mean()).alias("gap_cv"),
)
regular = gap_mode.join(gap_cv, on="uid")
show(regular, "mode_gap_share")

# ---------------------------------------------------------------------------
# 4. Повторы: один трек по кругу
# ---------------------------------------------------------------------------
section("4. Повторы одного трека")

runs = listens.group_by(["uid", "run"]).len()
rep = runs.group_by("uid").agg(pl.col("len").max().alias("max_same_in_a_row"))
top_item = (
    listens.group_by(["uid", "item_id"]).len()
    .sort("len", descending=True)
    .group_by("uid").agg(
        pl.col("item_id").first().alias("top_item"),
        pl.col("len").first().alias("top_item_plays"),
    )
)
rep = rep.join(top_item, on="uid").join(active, on="uid").with_columns(
    (pl.col("top_item_plays") / pl.col("n_listens")).alias("top_item_share"),
)
uniq = listens.group_by("uid").agg(pl.col("item_id").n_unique().alias("n_unique_items"))
rep = rep.join(uniq, on="uid").with_columns(
    (pl.col("n_listens") / pl.col("n_unique_items")).alias("plays_per_item"),
)
show(rep, "max_same_in_a_row")
show(rep, "top_item_share")

# ---------------------------------------------------------------------------
# 5. Оценки без прослушиваний
# ---------------------------------------------------------------------------
section("5. Лайки от пользователей, у которых нет ни одного прослушивания")

all_listeners = pl.read_parquet(DATA / "listens.parquet", columns=["uid"]).unique()
likes = pl.read_parquet(DATA / "likes.parquet").with_columns(pl.col("timestamp").cast(pl.Int64))
nl = likes.join(all_listeners, on="uid", how="anti").with_columns(
    (pl.col("timestamp") // 3600).alias("hour_abs")
)
nl_stats = nl.group_by("uid").agg(
    pl.len().alias("n_likes"),
    ((pl.col("timestamp").max() - pl.col("timestamp").min()) / 86400).round(1).alias("period_days"),
    pl.col("item_id").n_unique().alias("n_items"),
).join(
    nl.group_by(["uid", "hour_abs"]).len().group_by("uid").agg(pl.col("len").max().alias("max_likes_per_hour")),
    on="uid",
)
# для сравнения — те же показатели у пользователей, которые слушают
lk = likes.join(all_listeners, on="uid").with_columns((pl.col("timestamp") // 3600).alias("hour_abs"))
lk_stats = lk.group_by(["uid", "hour_abs"]).len().group_by("uid").agg(pl.col("len").max().alias("max_likes_per_hour"))
print("\nДля сравнения: макс. лайков за час у слушающих пользователей:")
print(lk_stats["max_likes_per_hour"].describe())
show(nl_stats, "max_likes_per_hour")
show(nl_stats, "n_likes")

# ---------------------------------------------------------------------------
# 6. Сводный рейтинг
# ---------------------------------------------------------------------------
section(f"6. Сводный рейтинг: сколько признаков у пользователя выбивается сильнее, чем на {Z_FLAG} робастных z")

features = (
    active
    .join(quiet, on="uid", how="left")
    .join(sess, on="uid", how="left")
    .join(day_stats, on="uid", how="left")
    .join(regular.select("uid", "mode_gap_share", "mode_gap_sec"), on="uid", how="left")
    .join(rep.select("uid", "max_same_in_a_row", "top_item_share", "plays_per_item"), on="uid", how="left")
)

check = ["quiet_3h_share", "max_session_h", "max_hours_per_day",
         "mode_gap_share", "max_same_in_a_row", "top_item_share", "plays_per_item"]


def robust_z(col: str) -> pl.Expr:
    """Отклонение от медианы в единицах MAD (устойчиво к выбросам и скошенности)."""
    med = pl.col(col).median()
    mad = (pl.col(col) - med).abs().median() * 1.4826
    # если у большинства значения одинаковые, MAD = 0; тогда берём стандартное отклонение
    scale = pl.when(mad > 0).then(mad).otherwise(pl.col(col).std())
    return ((pl.col(col) - med) / scale).fill_nan(0).fill_null(0).alias(f"z_{col}")


features = features.with_columns([robust_z(c) for c in check])
z_cols = [f"z_{c}" for c in check]
features = features.with_columns(
    pl.sum_horizontal([(pl.col(z) > Z_FLAG).cast(pl.Int32) for z in z_cols]).alias("n_flags"),
    pl.concat_str(
        [pl.when(pl.col(f"z_{c}") > Z_FLAG).then(pl.lit(c)).otherwise(pl.lit(None)) for c in check],
        separator=", ", ignore_nulls=True,
    ).alias("flags"),
)

print("\nСколько пользователей с данным числом сильных отклонений:")
print(features.group_by("n_flags").len().sort("n_flags"))

print("\nПользователи с наибольшим числом сильных отклонений:")
print(
    features.filter(pl.col("n_flags") >= 2)
    .sort(["n_flags", "n_listens"], descending=True)
    .select(["uid", "n_listens", "n_flags", "flags"] + check)
    .head(30)
)

features.write_parquet(DATA / "user_features_hunt.parquet")
print(f"\nТаблица признаков сохранена: {DATA / 'user_features_hunt.parquet'}")
