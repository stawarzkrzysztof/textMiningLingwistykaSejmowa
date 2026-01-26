import geopy.geocoders
import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

INPUT_CSV = "mp.csv"
OUTPUT_CSV = "mp_with_latlon.csv"
CACHE_CSV = "city_latlon_cache.csv"

# MUST be a real, identifying user-agent (not a placeholder).
USER_AGENT = "agh-sejm-geocoder/0.1 (contact: your_real_email@domain.tld)"
geopy.geocoders.options.default_user_agent = USER_AGENT  # optional, but safe

geolocator = Nominatim(user_agent=USER_AGENT, timeout=10)

# 1 req/sec max (policy) + do NOT retry/sleep on errors (prevents hanging on 403).
geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=1.1,
    max_retries=0,
    error_wait_seconds=0.0,
    swallow_exceptions=True,
    return_value_on_exception=None,
)

df = pd.read_csv(INPUT_CSV)

try:
    cache = pd.read_csv(CACHE_CSV)
except FileNotFoundError:
    cache = pd.DataFrame(columns=["miejsceUr", "miejsceUrLan", "miejsceUrLon"])

known = dict(zip(cache["miejsceUr"], zip(cache["miejsceUrLan"], cache["miejsceUrLon"])))

cities = df["miejsceUr"].astype(str).str.strip().replace({"": pd.NA}).dropna().unique()

new_rows = []
for city in cities:
    if city in known:
        continue

    # Prefer restricting to PL via Nominatim param (more robust than appending ", Poland")
    loc = geocode(city, country_codes="pl")
    lat = loc.latitude if loc else None
    lon = loc.longitude if loc else None
    new_rows.append((city, lat, lon))

if new_rows:
    cache = pd.concat(
        [
            cache,
            pd.DataFrame(
                new_rows, columns=["miejsceUr", "miejsceUrLan", "miejsceUrLon"]
            ),
        ],
        ignore_index=True,
    )
    cache.to_csv(CACHE_CSV, index=False)

df = df.drop(columns=[c for c in ["miejsceUrLan", "miejsceUrLon"] if c in df.columns])
df = df.merge(
    cache[["miejsceUr", "miejsceUrLan", "miejsceUrLon"]], on="miejsceUr", how="left"
)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")
