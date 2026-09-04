"""Site constants and model parameters. No logic lives here."""

LATITUDE = 51.98
LONGITUDE = 5.80
SITE_TZ = "Europe/Amsterdam"

DATA_GLOB = "pvoutput_gethistory.*.parquet.xz"

# Geometric elevation of the sun at SPA sunrise/sunset, in degrees.
# Measured against pvlib 0.15.2: -0.8359 to -0.8350 across a year.
SUNRISE_ELEVATION_DEG = -0.8358

# Model parameters (spec section 4.2).
POOL_HALF_WIDTH_DAYS = 10
PERCENTILE = 5.0
N_HARMONICS = 2
