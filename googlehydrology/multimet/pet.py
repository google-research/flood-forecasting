# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np
import xarray as xr


def calculate_fao56_penman_monteith_pet(
    t2m_k: np.ndarray,
    d2m_k: np.ndarray,
    sp_pa: np.ndarray,
    ssr_jm2: np.ndarray,
    str_jm2: np.ndarray,
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
) -> np.ndarray:
  """Computes daily FAO-56 Penman-Monteith reference evapotranspiration.

  Reference: Allen et al. (1998) FAO Irrigation and Drainage Paper No. 56.

  Args:
    t2m_k: Daily mean 2m temperature in Kelvin.
    d2m_k: Daily mean 2m dewpoint temperature in Kelvin.
    sp_pa: Daily mean surface pressure in Pascals.
    ssr_jm2: Daily surface net solar radiation in J/m^2.
    str_jm2: Daily surface net thermal radiation in J/m^2.
    u10_ms: Daily mean 10m U-component of wind in m/s.
    v10_ms: Daily mean 10m V-component of wind in m/s.

  Returns:
    Daily potential evapotranspiration in mm/day.
  """
  # Temperature in Celsius
  t_c = t2m_k - 273.15
  d_c = d2m_k - 273.15

  # Pressure in kPa
  p_kpa = sp_pa / 1000.0

  # Wind speed at 2m (converted from 10m using logarithmic profile)
  # u2 = u10 * 4.87 / ln(67.8 * 10 - 5.42) = u10 * 0.748
  wind_speed_10m = np.hypot(u10_ms, v10_ms)
  u2 = wind_speed_10m * (4.87 / np.log(67.8 * 10.0 - 5.42))

  # Net radiation Rn in MJ/m^2/day (from J/m^2)
  # Net radiation = net solar + net thermal
  rn_mj = (ssr_jm2 + str_jm2) / 1e6

  # Soil heat flux density G (assumed ~0 for daily time steps)
  g_mj = 0.0

  # Saturation vapor pressure es (kPa)
  es = 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))

  # Actual vapor pressure ea (kPa) from dewpoint temperature
  ea = 0.6108 * np.exp((17.27 * d_c) / (d_c + 237.3))

  # Slope of vapor pressure curve delta (kPa / °C)
  delta = (4098.0 * es) / ((t_c + 237.3) ** 2)

  # Psychrometric constant gamma (kPa / °C)
  gamma = 0.000665 * p_kpa

  # FAO-56 Penman-Monteith equation for daily step:
  # ETo = (0.408 * delta * (Rn - G) + gamma * (900 / (T + 273)) * u2 * (es - ea)) /
  #       (delta + gamma * (1 + 0.34 * u2))
  numerator = 0.408 * delta * (rn_mj - g_mj) + gamma * (
      900.0 / (t_c + 273.15)
  ) * u2 * np.maximum(0.0, es - ea)
  denominator = delta + gamma * (1.0 + 0.34 * u2)

  pet = numerator / np.maximum(1e-6, denominator)
  pet = np.maximum(0.0, pet)  # non-negative

  return pet.astype(np.float32)
