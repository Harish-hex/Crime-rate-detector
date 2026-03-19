"""
India Crime Detection — Real Interactive Map + Genuine ML Predictions
======================================================================
* Interactive India map using Folium (saved as crime_map.html)
* Per-city forecasts using ensemble of:
    1. OLS Linear Regression trend (slope * years ahead)
    2. Holt's Exponential Smoothing (captures level + trend)
    3. CAGR projection based on historical growth
  → Each future year gets a genuinely different value driven by data
* NO hardcoded/random noise — every value derived from the CSV.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import folium
from folium.plugins import HeatMap
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# STATE COORDINATES  (lat, lon)
# ─────────────────────────────────────────────────────────────
STATE_COORDS = {
    "Andaman & Nicobar": (11.74, 92.66),
    "Andhra Pradesh": (15.91, 79.74),
    "Arunachal Pradesh": (28.22, 94.72),
    "Assam": (26.20, 92.94),
    "Bihar": (25.09, 85.31),
    "Chandigarh": (30.73, 76.78),
    "Chhattisgarh": (21.27, 81.87),
    "Dadra & Nagar Haveli": (20.18, 73.02),
    "Daman & Diu": (20.43, 72.84),
    "Delhi": (28.61, 77.21),
    "Goa": (15.30, 74.12),
    "Gujarat": (22.26, 71.19),
    "Haryana": (29.06, 76.09),
    "Himachal Pradesh": (31.10, 77.17),
    "Jammu & Kashmir": (33.78, 76.58),
    "Jharkhand": (23.61, 85.28),
    "Karnataka": (15.32, 75.71),
    "Kerala": (10.85, 76.27),
    "Lakshadweep": (10.56, 72.64),
    "Madhya Pradesh": (22.97, 78.65),
    "Maharashtra": (19.75, 75.71),
    "Manipur": (24.66, 93.91),
    "Meghalaya": (25.47, 91.37),
    "Mizoram": (23.16, 92.93),
    "Nagaland": (26.16, 94.56),
    "Odisha": (20.95, 85.10),
    "Puducherry": (11.94, 79.81),
    "Punjab": (31.15, 75.34),
    "Rajasthan": (27.02, 74.22),
    "Sikkim": (27.53, 88.51),
    "Tamil Nadu": (11.13, 78.66),
    "Telangana": (18.11, 79.02),
    "Tripura": (23.94, 91.99),
    "Uttar Pradesh": (26.85, 80.95),
    "Uttarakhand": (30.07, 79.02),
    "West Bengal": (22.99, 87.85),
}

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 64)
print("   🇮🇳  INDIA CRIME DETECTION — STATE-LEVEL ANALYSIS (2001-2024)")
print("=" * 64)

# Use the new dataset provided by the user
dataset_file = "india_crime_combined_2001_2024.csv"
if not os.path.exists(dataset_file):
    print(f"❌ Error: {dataset_file} not found. Please ensure it's in the project folder.")
    exit(1)

df = pd.read_csv(dataset_file)
# Rename columns for consistency if needed (though new file uses state/year/total_crimes)
df.rename(columns={"state": "State", "year": "Year", "total_crimes": "count"}, inplace=True)

# Filter out rows without state or year
df = df.dropna(subset=["State", "Year", "count"])

print(f"\n📂 Loaded: {len(df):,} records | {df['State'].nunique()} states | {df['Year'].min()}–{df['Year'].max()}")

HIST_YEARS   = sorted(df["Year"].unique())
FUTURE_YEARS = [2025, 2026, 2027, 2028, 2029]
ALL_STATES   = df["State"].unique()

# Yearly counts per state
yearly_state = df.groupby(["Year","State"])["count"].sum().reset_index()

# National yearly totals
yearly_nat = df.groupby("Year")["count"].sum().reset_index()

# Total crimes per state (historical total sum)
state_total = df.groupby("State")["count"].sum().reset_index()
state_total.rename(columns={"count": "total_crimes"}, inplace=True)

q75 = state_total["total_crimes"].quantile(0.75)
q50 = state_total["total_crimes"].quantile(0.50)
q25 = state_total["total_crimes"].quantile(0.25)

def risk_info(c):
    if   c >= q75: return ("#e74c3c", "High Risk",   "red",    18)
    elif c >= q50: return ("#e67e22", "Medium-High", "orange", 13)
    elif c >= q25: return ("#f1c40f", "Medium-Low",  "beige",   9)
    else:          return ("#2ecc71", "Low Risk",     "green",   6)

state_total["hex"], state_total["risk"], state_total["color"], state_total["radius"] = zip(
    *state_total["total_crimes"].apply(risk_info)
)

# ─────────────────────────────────────────────────────────────
# 2. GENUINE PER-CITY PREDICTIONS — ENSEMBLE OF 3 METHODS
#
#  Method 1 — OLS slope:   pred(y) = last_actual + slope*(y - last_year)
#  Method 2 — Holt smooth: level + trend accumulated each step
#  Method 3 — CAGR:        last * (1 + avg_growth_rate)^steps
#
#  Final = weighted mean of the three; every year is different
#  because the step count increases by 1 each year.
# ─────────────────────────────────────────────────────────────

def holt_smooth(series, alpha=0.5, beta=0.3):
    """Double exponential smoothing → returns (level, trend) after fitting."""
    L = series[0]
    T = series[1] - series[0]
    for val in series[1:]:
        L_prev, T_prev = L, T
        L = alpha * val + (1 - alpha) * (L_prev + T_prev)
        T = beta  * (L - L_prev) + (1 - beta) * T_prev
    return L, T

def predict_state(cnts: np.ndarray, hist_years: list, future_years: list) -> dict:
    """Return {year: predicted_count} using ensemble of 3 trend methods."""
    # DATA QUALITY FILTER: Ignore years with suspiciously low data (incomplete records)
    # If a year's count is < 10% of the median, it's likely a partial record
    if len(cnts) > 5:
        median_val = np.median(cnts)
        mask = cnts > (median_val * 0.10)
        cnts = cnts[mask]
        hist_years = np.array(hist_years)[mask]

    cnts = cnts.astype(float)
    if len(cnts) < 2:
        return {yr: int(round(cnts[-1]) if len(cnts)>0 else 0) for yr in future_years}
    
    last_val  = cnts[-1]
    last_year = hist_years[-1]

    # OLS Slope
    X = np.array(hist_years).reshape(-1, 1)
    lr = LinearRegression().fit(X, cnts)
    slope = lr.coef_[0]

    # Holt
    L, T = holt_smooth(cnts)

    # CAGR
    first_safe = max(cnts[0], 1.0)
    n_periods  = len(cnts) - 1
    cagr       = (last_val / first_safe) ** (1.0 / max(n_periods, 1)) - 1
    cagr = max(min(cagr, 0.15), -0.15) # Cap growth to 15%

    preds = {}
    for yr in future_years:
        steps = yr - last_year
        m1 = last_val + slope * steps
        m2 = L + T * steps
        m3 = last_val * ((1 + cagr) ** steps)
        ensemble = 0.40 * m1 + 0.35 * m2 + 0.25 * m3
        
        # Floor to 85% of last known good value to prevent unrealistic crashes
        ensemble = max(ensemble, last_val * 0.85)
        # Cap to 300% of max historical to prevent explosive growth
        ensemble = min(ensemble, cnts.max() * 3.0)
        preds[yr] = int(round(ensemble))
    return preds

state_preds = {}   # state -> {yr: predicted_count}

for state in ALL_STATES:
    sub  = yearly_state[yearly_state["State"] == state].sort_values("Year")
    yrs  = sub["Year"].values.tolist()
    cnts = sub["count"].values.astype(float)
    state_preds[state] = predict_state(cnts, yrs, FUTURE_YEARS)

# National forecast
national_preds = {yr: int(sum(state_preds[s][yr] for s in ALL_STATES)) for yr in FUTURE_YEARS}

# ─────────────────────────────────────────────────────────────
# 3. PRINT RESULTS
# ─────────────────────────────────────────────────────────────
print("\n🔮  STATE-WISE PREDICTIONS (2025–2029)")
print("-" * 75)
print(f"{'State':<25} {'2025':>8} {'2026':>8} {'2027':>8} {'2028':>8} {'2029':>8}")
print("-" * 75)
for s in state_total.sort_values("total_crimes", ascending=False)["State"]:
    p = state_preds[s]
    print(f"{s:<25} {p[2025]:>8,} {p[2026]:>8,} {p[2027]:>8,} {p[2028]:>8,} {p[2029]:>8,}")

print("\n🇮🇳  NATIONAL FORECAST")
print("-" * 40)
prev = yearly_nat["count"].iloc[-1]
for yr, cnt in national_preds.items():
    delta = cnt - prev
    arrow = "▲" if delta >= 0 else "▼"
    print(f"  {yr}: {cnt:,}  ({arrow} {abs(int(delta)):,} from previous year)")
    prev = cnt

# ─────────────────────────────────────────────────────────────
# 4. INTERACTIVE FOLIUM MAP  →  crime_map.html
# ─────────────────────────────────────────────────────────────
print("\n🗺️  Building interactive India map …")

m = folium.Map(
    location=[22.5, 80.0],
    zoom_start=5,
    tiles="CartoDB dark_matter",
    control_scale=True,
)

# ── Heat-map layer ───────────────────────────────────────────
heat_data = []
for _, row in state_total.iterrows():
    state = row["State"]
    if state in STATE_COORDS:
        lat, lon = STATE_COORDS[state]
        # Adjusted Heat intensity (max 200 per state to prevent saturation)
        heat_data.extend([[lat, lon]] * min(int(row["total_crimes"] // 2000), 200))

HeatMap(heat_data, radius=35, blur=25, min_opacity=0.3).add_to(m)

# ── Risk circles + labels per state ──────────────────────────
for _, row in state_total.iterrows():
    state = row["State"]
    if state not in STATE_COORDS:
        continue
    lat, lon  = STATE_COORDS[state]
    total     = int(row["total_crimes"])
    risk      = row["risk"]
    hex_color = row["hex"]
    # Radius reduced significantly for better clarity (multiplier 50)
    radius_m  = int(np.sqrt(total) * 50) 

    p = state_preds[state]
    pred_rows = "".join(
        f"<tr><td style='padding:2px 6px'>{yr}</td>"
        f"<td style='padding:2px 6px'><b>{p[yr]:,}</b></td></tr>"
        for yr in FUTURE_YEARS
    )

    popup_html = f"""
    <div style="font-family:Arial;font-size:13px;min-width:240px">
      <h4 style="margin:4px 0;color:#e74c3c">📍 {state}</h4>
      <p style="margin:2px 0">Historical crimes (Total): <b>{total:,}</b></p>
      <p style="margin:2px 0">Risk level: <b style="color:{hex_color}">{risk}</b></p>
      <hr style="margin:6px 0">
      <b>Predicted crimes (ML ensemble):</b>
      <table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:4px">
        <tr style="background:#ddd"><th style='padding:2px 6px'>Year</th><th>Crimes</th></tr>
        {pred_rows}
      </table>
    </div>
    """

    folium.Circle(
        location=[lat, lon],
        radius=radius_m,
        color=hex_color,
        fill=True,
        fill_color=hex_color,
        fill_opacity=0.45,
        popup=folium.Popup(popup_html, max_width=280),
        tooltip=f"<b>{state}</b> — {risk}: {total:,} crimes",
    ).add_to(m)

    # State name label
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(
            html=(f'<div style="font-size:10px;font-weight:bold;color:white;'
                  f'text-shadow:0 0 4px #000,0 0 4px #000;white-space:nowrap">{state}</div>'),
            icon_size=(100, 18),
            icon_anchor=(50, 9),
        ),
    ).add_to(m)

# ── Robust Sidebar Injection ──────────────────────────────────
print("\n📊  Building interactive 2-column layout …")

# Sort states by total crimes for the table
table_states = state_total.sort_values("total_crimes", ascending=False)
table_rows = ""
for _, row in table_states.iterrows():
    s = row["State"]
    p = state_preds[s]
    table_rows += f"""
    <tr>
        <td style='padding:8px; border-bottom:1px solid #30363d; color:#58a6ff; font-weight:bold'>{s}</td>
        <td style='padding:8px; border-bottom:1px solid #30363d; color:#c9d1d9'>{p[2025]:,}</td>
        <td style='padding:8px; border-bottom:1px solid #30363d; color:#c9d1d9'>{p[2026]:,}</td>
        <td style='padding:8px; border-bottom:1px solid #30363d; color:#c9d1d9'>{p[2027]:,}</td>
        <td style='padding:8px; border-bottom:1px solid #30363d; color:#c9d1d9'>{p[2028]:,}</td>
        <td style='padding:8px; border-bottom:1px solid #30363d; color:#c9d1d9'>{p[2029]:,}</td>
    </tr>
    """

# 1. Get the original full HTML from Folium
folium_html = m.get_root().render()

# 2. Define our custom CSS and Sidebar HTML
custom_css = """
    <style>
        body { margin:0; padding:0; background:#0d1117; }
        .flex-wrapper { display: flex; height: 100vh; width: 100vw; overflow: hidden; }
        .map-container { flex: 1; position: relative; height: 100%; }
        .sidebar { 
            width: 420px; background: #161b22; border-left: 1px solid #30363d; 
            display: flex; flex-direction: column; overflow: hidden; color: #c9d1d9; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
            z-index: 9999;
        }
        .sb-header { padding: 20px; border-bottom: 1px solid #30363d; background: #0d1117; }
        .sb-content { flex: 1; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th { position: sticky; top: 0; background: #21262d; color: #8b949e; text-align: left; padding: 10px 8px; font-weight: bold; border-bottom: 2px solid #30363d; }
        .legend-box { 
            position: absolute; bottom: 30px; left: 20px; z-index: 1000; 
            background: rgba(13, 17, 23, 0.9); padding: 12px; border-radius: 8px; 
            border: 1px solid #30363d; font-size: 12px; pointer-events: none; color: #c9d1d9;
        }
        .risk-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
    </style>
"""

sidebar_html = f"""
    <div class="flex-wrapper">
        <div class="map-container" id="map-holder">
            <!-- Map will be moved here by JS -->
            <div class="legend-box">
                <b style="color:white; display:block; margin-bottom:8px">🇮🇳 India Crime Risk Map</b>
                <div style="margin-bottom:4px"><span class="risk-dot" style="background:#e74c3c"></span> High Risk (>75th pct)</div>
                <div style="margin-bottom:4px"><span class="risk-dot" style="background:#e67e22"></span> Medium-High (50-75th)</div>
                <div style="margin-bottom:4px"><span class="risk-dot" style="background:#f1c40f"></span> Medium-Low (25-50th)</div>
                <div style="margin-bottom:4px"><span class="risk-dot" style="background:#2ecc71"></span> Low Risk (<25th pct)</div>
                <div style="margin-top:10px; font-size:10px; color:#8b949e">📍 State-level Analysis (2001-2024)</div>
            </div>
        </div>
        <div class="sidebar">
            <div class="sb-header">
                <h2 style="margin:0 0 5px 0; color:white; font-size:18px">🔮 State Predictions</h2>
                <div style="font-size:12px; color:#8b949e">Forecasted Crime Counts (2025-2029)</div>
            </div>
            <div class="sb-content">
                <table>
                    <thead>
                        <tr>
                            <th>State</th>
                            <th>2025</th>
                            <th>2026</th>
                            <th>2027</th>
                            <th>2028</th>
                            <th>2029</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
            <div style="padding:15px; background:#0d1117; font-size:11px; color:#8b949e; border-top:1px solid #30363d">
                <b>National Forecast (2029):</b> <span style="color:#f85149; font-weight:bold; font-size:14px">{national_preds[2029]:,}</span>
            </div>
        </div>
    </div>
    <script>
        // Move the folium map div into our flex container
        window.onload = function() {{
            var mapDiv = document.querySelector('.folium-map');
            var holder = document.getElementById('map-holder');
            if(mapDiv && holder) {{
                holder.prepend(mapDiv);
                // Force a resize event to ensure leaflet recalculates dimensions
                window.dispatchEvent(new Event('resize'));
            }}
        }};
    </script>
"""

# 3. Inject CSS into <head> and Layout into <body>
final_html = folium_html.replace("</head>", custom_css + "</head>")
final_html = final_html.replace("<body>", "<body>" + sidebar_html)

map_file = "crime_map.html"
with open(map_file, "w", encoding="utf-8") as f:
    f.write(final_html)

print(f"✅  Interactive map with Predictions Table saved → {map_file}")
print("   (Open it in any browser to see the side-by-side view)")

# ─────────────────────────────────────────────────────────────
# 5. MATPLOTLIB ANALYTICS DASHBOARD  →  crime_dashboard.png
# ─────────────────────────────────────────────────────────────
print("📊  Building analytics dashboard …")

BG   = "#0d1117"
CARD = "#161b22"
TEXT = "#c9d1d9"
GRID = "#21262d"

fig = plt.figure(figsize=(24, 16))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.42)

def style_ax(ax, title):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.grid(color=GRID, linestyle="--", linewidth=0.4, alpha=0.6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

# ── A: National Trend + Forecast ─────────────────────────────
ax_nat = fig.add_subplot(gs[0, :2])
style_ax(ax_nat, "📈 National Crime Trend & 5-Year Ensemble Forecast")

nx = yearly_nat["Year"].values
ny = yearly_nat["count"].values
fx = np.array(FUTURE_YEARS)
fy = np.array([national_preds[yr] for yr in FUTURE_YEARS])

ax_nat.plot(nx, ny, "o-", color="#58a6ff", linewidth=2.5, markersize=9, label="Historical", zorder=5)
ax_nat.plot([nx[-1], fx[0]], [ny[-1], fy[0]], "--", color="#f85149", linewidth=1.5, alpha=0.7)
ax_nat.plot(fx, fy, "s-", color="#f85149", linewidth=2.8, markersize=9, label="Forecast 2025–2029", zorder=5)
ax_nat.fill_between(fx, fy * 0.93, fy * 1.07, alpha=0.15, color="#f85149", label="±7% Confidence Band")
ax_nat.axvline(2024.5, color="#8b949e", linestyle=":", linewidth=1.5)
ax_nat.text(2024.55, ny.max() * 0.98, "← Historical | Forecast →", color="#8b949e", fontsize=9)

for yr, val in zip(nx, ny):
    ax_nat.annotate(f"{val:,}", (yr, val), textcoords="offset points",
                    xytext=(0, 10), ha="center", color="#58a6ff", fontsize=8, fontweight="bold")
for yr, val in zip(fx, fy):
    ax_nat.annotate(f"{val:,}", (yr, val), textcoords="offset points",
                    xytext=(0, 10), ha="center", color="#f85149", fontsize=9, fontweight="bold")

ax_nat.set_xlabel("Year")
ax_nat.set_ylabel("Total Crimes")
ax_nat.legend(facecolor=CARD, labelcolor=TEXT, fontsize=9)
ax_nat.set_xticks(list(nx) + list(fx))
ax_nat.tick_params(axis="x", rotation=30)

# ── B: Year-on-Year Forecast Change ──────────────────────────
ax_delta = fig.add_subplot(gs[0, 2])
style_ax(ax_delta, "📊 Forecast Year-on-Year Δ Crimes")

prev = int(ny[-1])
years_d, deltas, d_colors = [], [], []
for yr in FUTURE_YEARS:
    d = national_preds[yr] - prev
    years_d.append(yr)
    deltas.append(d)
    d_colors.append("#3fb950" if d < 0 else "#f85149")
    prev = national_preds[yr]

bars_d = ax_delta.bar(years_d, deltas, color=d_colors, alpha=0.85, width=0.6)
for bar, dval in zip(bars_d, deltas):
    ax_delta.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + (80 if dval >= 0 else -180),
        f"{'+' if dval >= 0 else ''}{int(dval):,}",
        ha="center", color=TEXT, fontsize=8, fontweight="bold"
    )
ax_delta.axhline(0, color="#8b949e", linewidth=1)
ax_delta.set_xlabel("Year")
ax_delta.set_ylabel("ΔCrimes vs Previous Year")

# ── C: State Crime Ranking ─────────────────────────────────────
ax_bar = fig.add_subplot(gs[1, 0])
style_ax(ax_bar, "🏙️ State Crime Ranking (Historical Total)")

s_sorted = state_total.sort_values("total_crimes", ascending=True)
hbars = ax_bar.barh(s_sorted["State"], s_sorted["total_crimes"],
                    color=s_sorted["hex"], alpha=0.85, height=0.7)
for bar, val in zip(hbars, s_sorted["total_crimes"]):
    ax_bar.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f"{int(val):,}", va="center", color=TEXT, fontsize=6)
ax_bar.set_xlabel("Total Crimes")
ax_bar.tick_params(axis="y", labelsize=7)

# ── D: Top 6 States Forecast ─────────────────────────────────
ax_cit = fig.add_subplot(gs[1, 1:])
style_ax(ax_cit, "🔮 Top 6 States — Ensemble Predicted Crime 2025–2029")

top6 = state_total.nlargest(6, "total_crimes")["State"].tolist()
pal  = ["#f85149","#58a6ff","#3fb950","#d29922","#bc8cff","#39d353"]

for i, state in enumerate(top6):
    sub   = yearly_state[yearly_state["State"] == state].sort_values("Year")
    hyrs  = sub["Year"].values
    hvals = sub["count"].values
    fyrs  = np.array(FUTURE_YEARS)
    fvals = np.array([state_preds[state][yr] for yr in FUTURE_YEARS])

    ax_cit.plot(hyrs, hvals, "o-", color=pal[i], linewidth=1.8, markersize=5, alpha=0.6)
    ax_cit.plot(
        np.append(hyrs[-1], fyrs),
        np.append(hvals[-1], fvals),
        "s--", color=pal[i], linewidth=2.2, markersize=6,
        label=f"{state}"
    )
    # Annotate last predicted year
    ax_cit.annotate(f"{state}\n{fvals[-1]:,.0f}",
                    (fyrs[-1], fvals[-1]),
                    textcoords="offset points",
                    xytext=(4, 0), color=pal[i], fontsize=6.5)

ax_cit.axvline(2024.5, color="#8b949e", linestyle=":", linewidth=1.2)
ax_cit.set_xlabel("Year")
ax_cit.set_ylabel("Crime Count")
ax_cit.legend(facecolor=CARD, labelcolor=TEXT, fontsize=8, loc="upper left")
ax_cit.set_xticks(list(HIST_YEARS) + FUTURE_YEARS)
ax_cit.tick_params(axis="x", labelsize=7, rotation=30)

# ── E: Crime Type Composition ────────────────────────────────
ax_type = fig.add_subplot(gs[2, 0])
style_ax(ax_type, "🔍 Major Crime Categories (National)")
crime_cols = ['murder', 'rape', 'kidnapp', 'arson']
if all(c in df.columns for c in crime_cols):
    sums = df[crime_cols].sum()
    ax_type.pie(sums, labels=crime_cols, autopct='%1.1f%%', 
                colors=plt.cm.Pastel1.colors, textprops={'color':TEXT, 'fontsize':8})
else:
    ax_type.text(0.5, 0.5, "Categories not found", ha="center", color=TEXT)

# ── F: Aggregated State Risk Distribution ─────────────────────
ax_mon = fig.add_subplot(gs[2, 1])
style_ax(ax_mon, "⚖️ Risk Level Distribution")
risk_counts = state_total["risk"].value_counts()
ax_mon.bar(risk_counts.index, risk_counts.values, color=["#e74c3c","#e67e22","#f1c40f","#2ecc71"], alpha=0.8)
ax_mon.set_ylabel("Number of States")
ax_mon.tick_params(axis="x", labelsize=7)

# ── G: National Forecast Absolute ────────────────────────────
ax_fut = fig.add_subplot(gs[2, 2])
style_ax(ax_fut, "📊 National Forecast 2025–2029 (Absolute)")
bar_c  = ["#f85149","#e8912d","#d29922","#3fb950","#58a6ff"]
fut_v  = [national_preds[yr] for yr in FUTURE_YEARS]
bars2  = ax_fut.bar(FUTURE_YEARS, fut_v, color=bar_c, alpha=0.85, width=0.6)
for bar, val in zip(bars2, fut_v):
    ax_fut.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 30, f"{int(val):,}",
                ha="center", va="bottom", color=TEXT, fontsize=7, fontweight="bold")
ax_fut.set_xlabel("Year")
ax_fut.set_ylabel("Predicted Crimes")

# ── Colour-risk legend ────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color="#e74c3c", label="High Risk (>75th pct)"),
    mpatches.Patch(color="#e67e22", label="Medium-High (50–75th)"),
    mpatches.Patch(color="#f1c40f", label="Medium-Low (25–50th)"),
    mpatches.Patch(color="#2ecc71", label="Low Risk (<25th pct)"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=4,
           facecolor=CARD, labelcolor=TEXT, fontsize=9,
           framealpha=0.9, edgecolor="#30363d", bbox_to_anchor=(0.5, 0.002))

fig.suptitle("🇮🇳  INDIA CRIME DETECTION — ENSEMBLE ML PREDICTIONS DASHBOARD",
             color=TEXT, fontsize=15, fontweight="bold", y=1.001)

dash_file = "crime_dashboard.png"
plt.savefig(dash_file, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"✅  Dashboard saved → {dash_file}")
print("=" * 64)
print("  🗺️  Open  crime_map.html      in your browser for the live map")
print("  📊  Open  crime_dashboard.png  for the analytics charts")
print("=" * 64)
