"""
India Crime Detection — Real India Map + Real Predictions
==========================================================
- Draws a detailed India map with state boundaries
- Predictions use per-city Linear Regression on REAL data trends
- Each city gets a different forecast based on its own trend
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CITY COORDINATES
# ─────────────────────────────────────────────
CITY_COORDS = {
    "Delhi":         (28.61, 77.21),
    "Mumbai":        (19.08, 72.88),
    "Bangalore":     (12.97, 77.59),
    "Hyderabad":     (17.39, 78.49),
    "Chennai":       (13.08, 80.27),
    "Kolkata":       (22.57, 88.36),
    "Pune":          (18.52, 73.86),
    "Ahmedabad":     (23.02, 72.57),
    "Lucknow":       (26.85, 80.95),
    "Jaipur":        (26.91, 75.79),
    "Surat":         (21.17, 72.83),
    "Nagpur":        (21.15, 79.09),
    "Kanpur":        (26.45, 80.33),
    "Agra":          (27.18, 78.01),
    "Indore":        (22.72, 75.86),
    "Patna":         (25.59, 85.14),
    "Visakhapatnam": (17.69, 83.22),
    "Ludhiana":      (30.90, 75.86),
    "Bhopal":        (23.26, 77.41),
    "Thane":         (19.22, 72.98),
    "Ghaziabad":     (28.67, 77.45),
    "Nashik":        (20.00, 73.79),
    "Meerut":        (28.98, 77.71),
    "Srinagar":      (34.08, 74.80),
    "Faridabad":     (28.41, 77.32),
    "Varanasi":      (25.32, 82.97),
    "Kalyan":        (19.24, 73.14),
    "Vasai":         (19.39, 72.84),
    "Rajkot":        (22.30, 70.80),
}

# ─────────────────────────────────────────────
# INDIA STATE BOUNDARY POLYGONS (simplified)
# Format: list of (lon, lat) tuples per state
# ─────────────────────────────────────────────
STATES = {
    "Jammu & Kashmir": [(74.0,37.0),(80.0,37.0),(80.5,36.0),(79.0,34.5),(77.5,35.5),(75.5,36.5),(74.0,37.0)],
    "Himachal Pradesh": [(75.5,33.0),(77.5,33.5),(79.0,33.0),(78.5,31.5),(76.5,31.5),(75.5,33.0)],
    "Punjab":          [(73.8,32.5),(76.5,32.5),(76.5,31.5),(75.5,30.0),(73.8,30.5),(73.8,32.5)],
    "Haryana":         [(74.5,30.5),(77.5,31.0),(77.5,29.5),(76.0,29.0),(74.5,29.5),(74.5,30.5)],
    "Uttarakhand":     [(78.0,31.5),(81.0,31.0),(81.5,30.0),(79.5,29.5),(78.0,30.0),(78.0,31.5)],
    "Rajasthan":       [(69.5,29.5),(77.5,29.5),(77.5,28.0),(76.0,24.5),(74.0,23.5),(70.0,24.5),(68.5,27.5),(69.5,29.5)],
    "Uttar Pradesh":   [(77.5,31.0),(84.5,28.0),(84.5,24.0),(80.0,24.0),(77.5,26.0),(77.5,31.0)],
    "Bihar":           [(83.5,28.0),(88.0,27.5),(88.0,24.5),(84.5,24.0),(83.5,26.0),(83.5,28.0)],
    "Jharkhand":       [(83.5,26.0),(87.5,25.5),(87.5,22.5),(84.0,21.5),(83.0,23.0),(83.5,26.0)],
    "West Bengal":     [(85.5,27.5),(89.0,27.0),(89.5,22.0),(87.0,21.5),(85.5,22.0),(85.5,27.5)],
    "Odisha":          [(81.5,22.5),(87.0,22.5),(87.0,18.5),(84.0,17.5),(80.5,18.5),(81.5,22.5)],
    "Chhattisgarh":    [(80.0,24.0),(84.5,24.0),(84.0,21.5),(82.0,18.0),(79.5,18.5),(80.0,24.0)],
    "Madhya Pradesh":  [(74.0,23.5),(80.0,26.5),(84.5,24.0),(84.0,21.5),(80.5,18.5),(76.0,18.0),(73.0,21.0),(74.0,23.5)],
    "Gujarat":         [(68.0,24.5),(74.0,24.5),(74.5,22.0),(73.5,20.0),(72.5,20.0),(68.5,22.0),(68.0,24.5)],
    "Maharashtra":     [(73.0,21.0),(80.5,18.5),(80.0,17.0),(77.5,15.5),(74.0,15.5),(72.5,20.0),(73.0,21.0)],
    "Telangana":       [(77.5,19.5),(82.0,19.5),(82.0,16.5),(79.5,16.0),(77.5,17.0),(77.5,19.5)],
    "Andhra Pradesh":  [(77.5,15.5),(80.0,17.0),(82.0,16.5),(84.5,14.0),(80.5,12.5),(77.5,13.5),(77.5,15.5)],
    "Karnataka":       [(74.0,15.5),(77.5,17.0),(77.5,13.5),(80.5,12.5),(78.5,10.5),(75.5,10.0),(74.0,12.0),(74.0,15.5)],
    "Kerala":          [(75.5,12.5),(77.5,12.5),(78.0,10.5),(77.0,8.0),(76.0,8.5),(75.5,10.5),(75.5,12.5)],
    "Tamil Nadu":      [(77.5,13.5),(80.5,12.5),(80.5,8.5),(78.0,8.0),(77.0,8.0),(77.5,10.5),(77.5,13.5)],
    "Assam":           [(89.5,26.5),(92.5,27.5),(96.0,27.0),(95.5,25.0),(92.0,24.5),(89.5,25.0),(89.5,26.5)],
    "Northeast":       [(91.5,27.5),(97.5,28.5),(97.0,24.0),(93.5,23.5),(91.5,25.0),(91.5,27.5)],
    "Sikkim":          [(88.0,28.0),(89.5,28.5),(89.5,27.0),(88.0,27.5),(88.0,28.0)],
}

STATE_COLORS = [
    "#1a3a5c","#1e4d7a","#16324d","#1b4570","#153050",
    "#1a3a5c","#1e4d7a","#16324d","#1b4570","#153050",
    "#1a3a5c","#1e4d7a","#16324d","#1b4570","#153050",
    "#1a3a5c","#1e4d7a","#16324d","#1b4570","#153050",
    "#1a3a5c","#1e4d7a","#16324d",
]

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 62)
print("   🇮🇳  INDIA CRIME DETECTION & 5-YEAR REAL FORECAST")
print("=" * 62)

df = pd.read_csv("crime_dataset_india.csv")
df["Date of Occurrence"] = pd.to_datetime(df["Date of Occurrence"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Date of Occurrence"])
df["Year"]  = df["Date of Occurrence"].dt.year
df["Month"] = df["Date of Occurrence"].dt.month

print(f"\n📂 Loaded: {len(df):,} records | {df['City'].nunique()} cities | {df['Year'].min()}–{df['Year'].max()}")

yearly_city  = df.groupby(["Year","City"]).size().reset_index(name="crime_count")
yearly_total = df.groupby("Year").size().reset_index(name="crime_count")
hist_years   = sorted(df["Year"].unique())
future_years = [2025, 2026, 2027, 2028, 2029]

# ─────────────────────────────────────────────
# 2. PER-CITY REAL PREDICTIONS
#    Uses Linear Regression on actual city yearly counts
#    Each city gets a different slope → different forecasts
# ─────────────────────────────────────────────
city_predictions = {}

for city in df["City"].unique():
    sub  = yearly_city[yearly_city["City"] == city].sort_values("Year")
    X    = sub["Year"].values.reshape(-1, 1)
    y    = sub["crime_count"].values

    # Linear regression on real data
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    r2    = r2_score(y, model.predict(X))

    future_preds = {}
    for yr in future_years:
        pred = model.predict([[yr]])[0]
        # Bound predictions realistically
        pred = max(pred, y.min() * 0.5)   # won't go below 50% of historical min
        pred = min(pred, y.max() * 1.4)   # won't exceed 40% above historical max
        future_preds[yr] = round(float(pred))

    city_predictions[city] = {
        "hist_years":  list(sub["Year"].values),
        "hist_counts": list(y),
        "future":      future_preds,
        "slope":       slope,
        "r2":          r2,
        "trend":       "↑ Rising" if slope > 2 else ("↓ Falling" if slope < -2 else "→ Stable"),
    }

# National forecast = sum of all city predictions
national_future = {}
for yr in future_years:
    national_future[yr] = int(sum(city_predictions[c]["future"][yr] for c in df["City"].unique()))

# ─────────────────────────────────────────────
# 3. CITY RISK CLASSIFICATION
# ─────────────────────────────────────────────
city_total = df.groupby("City").size().reset_index(name="total_crimes")
q75 = city_total["total_crimes"].quantile(0.75)
q50 = city_total["total_crimes"].quantile(0.50)
q25 = city_total["total_crimes"].quantile(0.25)

def get_risk(c):
    if c >= q75:   return ("#e74c3c", "🔴 High Risk",   220)
    elif c >= q50: return ("#e67e22", "🟠 Medium-High", 140)
    elif c >= q25: return ("#f1c40f", "🟡 Medium-Low",   90)
    else:          return ("#2ecc71", "🟢 Low Risk",      55)

city_total["color"] = city_total["total_crimes"].apply(lambda x: get_risk(x)[0])
city_total["label"] = city_total["total_crimes"].apply(lambda x: get_risk(x)[1])
city_total["size"]  = city_total["total_crimes"].apply(lambda x: get_risk(x)[2])

print("\n🔮 CITY-WISE PREDICTIONS (2025–2029)")
print("-" * 62)
print(f"{'City':<18} {'2025':>6} {'2026':>6} {'2027':>6} {'2028':>6} {'2029':>6}  Trend")
print("-" * 62)
for city in city_total.sort_values("total_crimes", ascending=False)["City"]:
    cp = city_predictions[city]
    f  = cp["future"]
    print(f"{city:<18} {f[2025]:>6} {f[2026]:>6} {f[2027]:>6} {f[2028]:>6} {f[2029]:>6}  {cp['trend']}")

print("\n🇮🇳 NATIONAL FORECAST")
print("-" * 40)
for yr, cnt in national_future.items():
    print(f"  {yr}: {cnt:,} crimes")

# ─────────────────────────────────────────────
# 4. PLOT
# ─────────────────────────────────────────────
BG   = "#0d1117"
CARD = "#161b22"
TEXT = "#c9d1d9"
GRID = "#21262d"

fig = plt.figure(figsize=(24, 18))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

def style_ax(ax, title):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(color=GRID, linestyle="--", linewidth=0.4, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

# ── INDIA MAP ────────────────────────────────
ax_map = fig.add_subplot(gs[:2, :2])
ax_map.set_facecolor("#060e1e")

# Draw sea / ocean background
ocean = plt.Rectangle((65, 5), 37, 35, color="#060e1e", zorder=0)
ax_map.add_patch(ocean)

# Draw state polygons
for i, (state, coords) in enumerate(STATES.items()):
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    color = STATE_COLORS[i % len(STATE_COLORS)]
    ax_map.fill(lons, lats, color=color, alpha=0.9, zorder=1)
    ax_map.plot(lons, lats, color="#2d5a8e", linewidth=0.8, zorder=2)

# Sri Lanka
ax_map.fill([80.0,81.5,81.5,80.0],[10.0,10.0,8.0,8.0], color="#1a3a5c", alpha=0.7, zorder=1)

# Plot cities
city_lookup = city_total.set_index("City")
for _, row in city_total.iterrows():
    city = row["City"]
    if city not in CITY_COORDS:
        continue
    lat, lon = CITY_COORDS[city]
    ax_map.scatter(lon, lat, s=row["size"], c=row["color"],
                   alpha=0.92, zorder=5, edgecolors="white", linewidths=0.8)

    # Label high & medium-high cities
    if row["total_crimes"] >= q50:
        txt = ax_map.annotate(
            city, (lon, lat),
            textcoords="offset points",
            xytext=(7, 3),
            color="white", fontsize=7, fontweight="bold", zorder=6,
        )
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground="black")])

# Compass rose
ax_map.annotate("N", xy=(97.5, 36), fontsize=13, color="white", fontweight="bold", ha="center")
ax_map.annotate("▲", xy=(97.5, 35.2), fontsize=10, color="white", ha="center")

ax_map.set_xlim(66, 100)
ax_map.set_ylim(6, 38)
ax_map.set_xlabel("Longitude", color=TEXT, fontsize=9)
ax_map.set_ylabel("Latitude",  color=TEXT, fontsize=9)
ax_map.tick_params(colors=TEXT, labelsize=8)
for spine in ax_map.spines.values():
    spine.set_edgecolor("#30363d")
ax_map.set_title("🗺️  India Crime Risk Map — City Level (2020–2024)",
                 color=TEXT, fontsize=13, fontweight="bold", pad=12)

# Legend
legend_items = [
    mpatches.Patch(color="#e74c3c", label="🔴 High Risk  (>75th percentile)"),
    mpatches.Patch(color="#e67e22", label="🟠 Medium-High (50–75th)"),
    mpatches.Patch(color="#f1c40f", label="🟡 Medium-Low  (25–50th)"),
    mpatches.Patch(color="#2ecc71", label="🟢 Low Risk   (<25th percentile)"),
]
ax_map.legend(handles=legend_items, loc="lower left", fontsize=8.5,
              facecolor="#0d1a2d", labelcolor=TEXT, framealpha=0.95,
              edgecolor="#2d5a8e")

# ── CITY RANKING BAR ─────────────────────────
ax_bar = fig.add_subplot(gs[:2, 2])
style_ax(ax_bar, "🏙️ City Crime Ranking (2020–2024)")
city_sorted = city_total.sort_values("total_crimes", ascending=True)
bars = ax_bar.barh(city_sorted["City"], city_sorted["total_crimes"],
                   color=city_sorted["color"], alpha=0.85, height=0.7)
for bar, val in zip(bars, city_sorted["total_crimes"]):
    ax_bar.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                str(int(val)), va="center", color=TEXT, fontsize=6.5)
ax_bar.set_xlabel("Total Crimes")
ax_bar.tick_params(axis="y", labelsize=7)

# ── NATIONAL TREND + FORECAST ─────────────────
ax_nat = fig.add_subplot(gs[2, :2])
style_ax(ax_nat, "📈 National Crime Trend & 5-Year Real Forecast")

nat_x = yearly_total["Year"].values
nat_y = yearly_total["crime_count"].values

# Fit linear on historical
lr_nat = LinearRegression()
lr_nat.fit(nat_x.reshape(-1,1), nat_y)
fitted_y = lr_nat.predict(nat_x.reshape(-1,1))

fut_x = np.array(future_years)
fut_y = np.array([national_future[yr] for yr in future_years])

ax_nat.plot(nat_x, nat_y, "o", color="#58a6ff", markersize=10, zorder=5, label="Actual Data")
ax_nat.plot(nat_x, fitted_y, "-", color="#58a6ff", linewidth=2, alpha=0.6, label="Historical Trend")

# Connect last real to first forecast
ax_nat.plot([nat_x[-1], fut_x[0]], [nat_y[-1], fut_y[0]], "--", color="#f85149", linewidth=1.5, alpha=0.6)
ax_nat.plot(fut_x, fut_y, "s-", color="#f85149", linewidth=2.5, markersize=9, label="Forecast 2025–2029", zorder=5)
ax_nat.fill_between(fut_x, fut_y * 0.90, fut_y * 1.10, alpha=0.12, color="#f85149", label="±10% Confidence Band")

for yr, val in zip(fut_x, fut_y):
    ax_nat.annotate(f"{val:,}", (yr, val), textcoords="offset points",
                    xytext=(0, 12), ha="center", color="#f85149", fontsize=9, fontweight="bold")

for yr, val in zip(nat_x, nat_y):
    ax_nat.annotate(f"{val:,}", (yr, val), textcoords="offset points",
                    xytext=(0, 10), ha="center", color="#58a6ff", fontsize=8)

ax_nat.axvline(x=2024.5, color="#8b949e", linestyle=":", linewidth=1.5)
ax_nat.text(2024.6, nat_y.max() * 0.97, "← Historical | Forecast →", color="#8b949e", fontsize=8.5)
ax_nat.set_xlabel("Year")
ax_nat.set_ylabel("Total Crimes")
ax_nat.legend(facecolor=CARD, labelcolor=TEXT, fontsize=8.5)
ax_nat.set_xticks(list(nat_x) + future_years)
ax_nat.tick_params(axis="x", rotation=30)

# ── TOP 6 CITIES FORECAST ────────────────────
ax_city = fig.add_subplot(gs[2, 2])
style_ax(ax_city, "🔮 Top 6 Cities — Real Predictions 2025–2029")

top6   = city_total.nlargest(6, "total_crimes")["City"].tolist()
ccolors= ["#f85149","#58a6ff","#3fb950","#d29922","#bc8cff","#39d353"]

for i, city in enumerate(top6):
    cp     = city_predictions[city]
    hyrs   = cp["hist_years"]
    hvals  = cp["hist_counts"]
    fyrs   = list(cp["future"].keys())
    fvals  = list(cp["future"].values())

    ax_city.plot(hyrs, hvals, "o-", color=ccolors[i], linewidth=1.8,
                 markersize=5, alpha=0.7)
    ax_city.plot([hyrs[-1]] + fyrs, [hvals[-1]] + fvals,
                 "s--", color=ccolors[i], linewidth=2.2,
                 markersize=6, label=f"{city} ({cp['trend']})")

ax_city.axvline(x=2024.5, color="#8b949e", linestyle=":", linewidth=1.2)
ax_city.set_xlabel("Year")
ax_city.set_ylabel("Crime Count")
ax_city.legend(facecolor=CARD, labelcolor=TEXT, fontsize=7, loc="upper right")
ax_city.set_xticks(list(hist_years) + future_years)
ax_city.tick_params(axis="x", labelsize=7, rotation=30)

fig.suptitle("🇮🇳  INDIA CRIME DETECTION — REAL MAP + REAL 5-YEAR PREDICTIONS",
             color=TEXT, fontsize=15, fontweight="bold", y=0.998)

plt.savefig("crime_india_final.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("\n✅ Saved: crime_india_final.png")
print("=" * 62)
