"""
Utilitas akuisisi dan integrasi dataset bulanan DBD & cuaca DKI Jakarta (2010-2024).

Prioritas pencarian sumber raw dataset:
1. Lokal di `project/data/raw/`
2. Lokal di `../dataset/raw/` (folder kerja utama)
3. Unduh otomatis via `kagglehub` & GitHub OpenDengue (berguna saat dijalankan di Colab
   setelah clone repositori yang tidak meng-upload folder `data/`).
"""
from __future__ import annotations

import glob
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OPENDENGUE_URL = (
    "https://github.com/OpenDengue/master-repo/"
    "raw/main/data/releases/V1.3.1/Temporal_extract_V1_3_1.zip"
)


def get_project_root() -> Path:
    """Mengembalikan path root folder `project/`."""
    return Path(__file__).resolve().parent.parent


def ensure_raw_datasets(raw_dir: Path | None = None, allow_download: bool = True) -> dict[str, Path]:
    """
    Memastikan ketiga file raw dataset tersedia di `data/raw/`.
    Menyalin dari `../dataset/raw/` bila tersedia secara lokal, atau mengunduh dari sumber asli.
    """
    project_root = get_project_root()
    if raw_dir is None:
        raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    fallback_raw_dir = project_root.parent / "dataset" / "raw"
    expected_files = {
        "climate": raw_dir / "climate_data.csv",
        "dengue_weather_2021_2024": raw_dir / "jakarta-dengue-weather_2021-2024.csv",
        "opendengue": raw_dir / "dengue_dki_jakarta_bulanan_2004_2024.csv",
    }

    # 1. Copy from ../dataset/raw/ bila belum ada di project/data/raw/
    for key, target_path in expected_files.items():
        if not target_path.exists() and fallback_raw_dir.exists():
            candidate = fallback_raw_dir / target_path.name
            if candidate.exists():
                shutil.copy2(candidate, target_path)

    missing = [k for k, p in expected_files.items() if not p.exists()]
    if missing and not allow_download:
        raise FileNotFoundError(
            f"Raw dataset belum lengkap di {raw_dir}: {missing}. "
            "Aktifkan allow_download=True atau salin manual ke data/raw/."
        )

    if missing and allow_download:
        import kagglehub
        import requests

        if not expected_files["climate"].exists():
            clim_dl = Path(kagglehub.dataset_download("greegtitan/indonesia-climate"))
            matches = [Path(p) for p in glob.glob(str(clim_dl / "**" / "*.csv"), recursive=True)
                       if "climate_data" in Path(p).name.lower()]
            if not matches:
                raise FileNotFoundError("climate_data.csv tidak ditemukan dari unduhan Kaggle.")
            shutil.copy2(matches[0], expected_files["climate"])

        if not expected_files["dengue_weather_2021_2024"].exists():
            dw_dl = Path(kagglehub.dataset_download("yutakatarokusumah/jakarta-dengue-weather-2021-2024"))
            matches = [Path(p) for p in glob.glob(str(dw_dl / "**" / "*.csv"), recursive=True)
                       if "jakarta-dengue-weather" in Path(p).name.lower()]
            if not matches:
                raise FileNotFoundError("jakarta-dengue-weather_2021-2024.csv tidak ditemukan dari unduhan Kaggle.")
            shutil.copy2(matches[0], expected_files["dengue_weather_2021_2024"])

        if not expected_files["opendengue"].exists():
            zip_path = raw_dir / "opendengue.zip"
            resp = requests.get(OPENDENGUE_URL, timeout=180)
            resp.raise_for_status()
            zip_path.write_bytes(resp.content)
            with zipfile.ZipFile(zip_path, "r") as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                selected = next((n for n in csv_names if "temporal" in Path(n).name.lower()), csv_names[0])
                with zf.open(selected) as f:
                    od_raw = pd.read_csv(f, low_memory=False)
            od = od_raw[
                (od_raw["ISO_A0"] == "IDN")
                & (od_raw["adm_1_name"].str.upper() == "DKI JAKARTA")
                & (od_raw["T_res"] == "Month")
            ].copy()
            od["calendar_start_date"] = pd.to_datetime(od["calendar_start_date"], errors="coerce")
            od = od[
                (od["calendar_start_date"] >= "2004-01-01")
                & (od["calendar_start_date"] <= "2024-07-31")
            ].copy()
            od.to_csv(expected_files["opendengue"], index=False)

    return expected_files


def build_monthly_dataset(
    raw_dir: Path | None = None,
    out_path: Path | None = None,
    allow_download: bool = True,
) -> pd.DataFrame:
    """
    Mengintegrasikan cuaca harian BMKG Jakarta (Stasiun 96745 Kemayoran & 96741 Tanjung Priok)
    periode 2010-2024 ke agregasi bulanan dan menggabungkannya dengan OpenDengue DKI Jakarta
    (2010-01 s.d. 2024-07, total 175 baris x 15 kolom).
    """
    project_root = get_project_root()
    raw_paths = ensure_raw_datasets(raw_dir=raw_dir, allow_download=allow_download)

    if out_path is None:
        out_path = project_root / "data" / "merged" / "dki_jakarta_dengue_weather_2010_2024_monthly.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_cols = ["Tn", "Tx", "Tavg", "RH_avg", "RR", "ss", "ff_x", "ddd_x", "ff_avg"]

    # 1. BMKG 2010-2020 (Stasiun 96741 Tanjung Priok, 96745 Kemayoran)
    clim = pd.read_csv(raw_paths["climate"])
    c_jkt = clim[clim["station_id"].isin([96741, 96745])].copy()
    c_jkt["date_dt"] = pd.to_datetime(c_jkt["date"], format="%d-%m-%Y", errors="coerce")
    c_jkt["ym"] = c_jkt["date_dt"].dt.to_period("M")
    for col in num_cols:
        c_jkt[col] = pd.to_numeric(c_jkt[col], errors="coerce")
        c_jkt.loc[c_jkt[col] >= 8888, col] = np.nan
        c_jkt.loc[c_jkt[col] < 0, col] = np.nan

    # 2. BMKG 2021-2024 (jak-pus = Kemayoran 96745, jak-ut = Tanjung Priok 96741)
    orig = pd.read_csv(raw_paths["dengue_weather_2021_2024"])
    orig["date_dt"] = pd.to_datetime(orig["DATE"], format="%d-%m-%Y", errors="coerce")
    orig["ym"] = orig["date_dt"].dt.to_period("M")
    orig = orig.rename(
        columns={
            "TN": "Tn",
            "TX": "Tx",
            "TAVG": "Tavg",
            "RH_AVG": "RH_avg",
            "SS": "ss",
            "FF_X": "ff_x",
            "DDD_X": "ddd_x",
            "FF_AVG": "ff_avg",
        }
    )
    orig["station_id"] = orig["REG"].map({"jak-pus": 96745, "jak-ut": 96741})
    for col in num_cols:
        orig[col] = pd.to_numeric(orig[col], errors="coerce")
        orig.loc[orig[col] >= 8888, col] = np.nan
        orig.loc[orig[col] < 0, col] = np.nan

    daily_weather = pd.concat(
        [
            c_jkt[["date_dt", "ym", "station_id"] + num_cols],
            orig[["date_dt", "ym", "station_id"] + num_cols],
        ],
        ignore_index=True,
    )
    daily_weather["days_in_month"] = daily_weather["date_dt"].dt.days_in_month

    st_monthly = (
        daily_weather.groupby(["ym", "station_id"])
        .agg(
            Tn=("Tn", "mean"),
            Tx=("Tx", "mean"),
            Tavg=("Tavg", "mean"),
            RH_avg=("RH_avg", "mean"),
            RR_mean=("RR", "mean"),
            days=("days_in_month", "first"),
            ss=("ss", "mean"),
            ff_x=("ff_x", "max"),
            ff_avg=("ff_avg", "mean"),
            rain_days=("RR", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )
    st_monthly["RR_total"] = st_monthly["RR_mean"] * st_monthly["days"]

    w_monthly = st_monthly.groupby("ym").agg(
        Tn=("Tn", "mean"),
        Tx=("Tx", "mean"),
        Tavg=("Tavg", "mean"),
        RH_avg=("RH_avg", "mean"),
        RR_total=("RR_total", "mean"),
        RR_mean=("RR_mean", "mean"),
        rain_days=("rain_days", "mean"),
        ss=("ss", "mean"),
        ff_x=("ff_x", "mean"),
        ff_avg=("ff_avg", "mean"),
    )

    # 3. OpenDengue DKI Jakarta Monthly (2010-01 s.d. 2024-07)
    dengue = pd.read_csv(raw_paths["opendengue"])
    dengue["ym"] = pd.to_datetime(dengue["calendar_start_date"]).dt.to_period("M")

    idx = pd.period_range("2010-01", "2024-07", freq="M")
    df_final = pd.DataFrame(index=idx)
    df_final.index.name = "ym"
    df_final = df_final.join(dengue.set_index("ym")[["dengue_total"]]).join(w_monthly)
    df_final = df_final.reset_index()
    df_final["date"] = df_final["ym"].dt.to_timestamp().dt.strftime("%Y-%m-%d")
    df_final["year"] = df_final["ym"].dt.year
    df_final["month"] = df_final["ym"].dt.month
    df_final["ym"] = df_final["ym"].astype(str)

    cols = [
        "ym", "date", "year", "month",
        "Tn", "Tx", "Tavg", "RH_avg", "RR_total", "RR_mean",
        "rain_days", "ss", "ff_x", "ff_avg", "dengue_total",
    ]
    df_final = df_final[cols]
    df_final.to_csv(out_path, index=False, float_format="%.4f")
    return df_final


if __name__ == "__main__":
    df_res = build_monthly_dataset(allow_download=False)
    print(f"Dataset gabungan siap: {df_res.shape[0]} baris x {df_res.shape[1]} kolom")
