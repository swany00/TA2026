from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np


def parse_utc_timestamp(ts_yyyymmddhhmm: str) -> datetime:
    return datetime.strptime(ts_yyyymmddhhmm, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def encode_time_features(ts_yyyymmddhhmm: str, sza_center: float) -> np.ndarray:
    dt = parse_utc_timestamp(ts_yyyymmddhhmm)
    doy = dt.timetuple().tm_yday
    hour = dt.hour + dt.minute / 60.0
    month = dt.month
    doy_theta = 2.0 * math.pi * doy / 365.0
    hour_theta = 2.0 * math.pi * hour / 24.0
    month_theta = 2.0 * math.pi * month / 12.0
    return np.array(
        [
            math.cos(doy_theta),
            math.sin(doy_theta),
            math.cos(hour_theta),
            math.sin(hour_theta),
            math.cos(month_theta),
            math.sin(month_theta),
            float(sza_center),
        ],
        dtype=np.float32,
    )
