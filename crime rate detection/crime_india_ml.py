"""
India Crime Rate Detection & Future Prediction ML Model
========================================================
- Uses real Indian crime dataset
- Trains ML model to predict crime counts
- Forecasts crime for next 5 years (2025–2029)
- Visualizes trends by city and crime type
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("   INDIA CRIME RATE DETECTION & FUTURE PREDICTION")
print("=" * 60)

df = pd.read_csv("crime_dataset_india.csv")

# Parse dates
df["Date of Occurrence"] = pd.to_datetime(df["Date of Occurrence"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Date of Occurrence"])

df["Year"]        = df["Date of Occurrence"].dt.year
df["Month"]       = df["Date of Occurrence"].dt.month
df["Day"]         = df["Date of Occurrence"].dt.day
df["DayOfWeek"]   = df["Date of Occurrence"].dt.dayofweek
df["Quarter"]     = df["Date of Occurrence"].dt.quarter

print(f"\n📂 Dataset Loaded: {df.shape[0]:,} records")
print(f"   Cities  : {df['City'].nunique()} Indian cities")
print(f"   Years   : {df['Year'].min()} – {df['Year'].max()}")
print(f"   Crimes  : {df['Crime Description'].nunique()} types")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
le_city   = LabelEncoder()
le_crime  = LabelEncoder()
le_domain = LabelEncoder()
le_weapon = LabelEncoder()

df["City_enc"]        = le_city.fit_transform(df["City"])
df["CrimeType_enc"]   = le_crime.fit_transform(df["Crime Description"])
df["Domain_enc"]      = le_domain.fit_transform(df["Crime Domain"])
df["Weapon_enc"]      = le_weapon.fit_transform(df["Weapon Used"].fillna("None"))
df["CaseClosed_enc"]  = (df["Case Closed"] == "Yes").astype(int)

# Aggregate monthly crime counts per city
monthly = df.groupby(["Year", "Month", "City", "City_enc", "Crime Domain", "Domain_enc"]).agg(
    crime_count     = ("Report Number", "count"),
    avg_police      = ("Police Deployed", "mean"),
    case_close_rate = ("CaseClosed_enc", "mean"),
    avg_victim_age  = ("Victim Age", "mean"),
).reset_index()

features = [
    "Year", "Month", "City_enc", "Domain_enc",
    "avg_police", "case_close_rate", "avg_victim_age"
]

X = monthly[features]
y = monthly["crime_count"]

# ─────────────────────────────────────────────
# 3. TRAIN ML MODEL
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    results[name] = {
        "model": model, "preds": preds,
        "MAE":  mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2":   r2_score(y_test, preds),
    }

print("\n📈 MODEL PERFORMANCE")
print("-" * 50)
for name, r in results.items():
    print(f"  {name}")
    print(f"    MAE: {r['MAE']:.2f} | RMSE: {r['RMSE']:.2f} | R²: {r['R2']:.3f}")

best_name = max(results, key=lambda k: results[k]["R2"])
best_model = results[best_name]["model"]
print(f"\n✅ Best Model: {best_name} (R² = {results[best_name]['R2']:.3f})")

# ─────────────────────────────────────────────
# 4. FUTURE PREDICTIONS: 2025–2029
# ─────────────────────────────────────────────
future_years = [2025, 2026, 2027, 2028, 2029]
cities       = df["City"].unique()
domains      = df["Crime Domain"].unique()

future_rows = []
for year in future_years:
    for month in range(1, 13):
        for city in cities:
            for domain in domains:
                city_enc   = le_city.transform([city])[0]
                domain_enc = le_domain.transform([domain])[0]
                future_rows.append({
                    "Year": year, "Month": month,
                    "City": city, "City_enc": city_enc,
                    "Crime Domain": domain, "Domain_enc": domain_enc,
                    "avg_police":      monthly["avg_police"].mean(),
                    "case_close_rate": monthly["case_close_rate"].mean(),
                    "avg_victim_age":  monthly["avg_victim_age"].mean(),
                })

future_df = pd.DataFrame(future_rows)
X_future   = future_df[features]
X_future_sc = scaler.transform(X_future)
future_df["predicted_crime"] = best_model.predict(X_future_sc).clip(0)

# Yearly city summary
yearly_city = future_df.groupby(["Year", "City"])["predicted_crime"].sum().reset_index()
# Historical yearly city
hist_city   = df.groupby(["Year", "City"]).size().reset_index(name="crime_count")

# Yearly total
yearly_total_hist   = df.groupby("Year").size().reset_index(name="crime_count")
yearly_total_future = future_df.groupby("Year")["predicted_crime"].sum().reset_index()
yearly_total_future.columns = ["Year", "crime_count"]

print("\n🔮 FUTURE CRIME PREDICTIONS (Total — All Cities)")
print("-" * 50)
for _, row in yearly_total_future.iterrows():
    print(f"  {int(row['Year'])}: {int(row['crime_count']):,} predicted crimes")

# ─────────────────────────────────────────────
# 5. CRIME ZONE CLUSTERING
# ─────────────────────────────────────────────
city_stats = df.groupby("City").agg(
    total_crimes   = ("Report Number", "count"),
    avg_police     = ("Police Deployed", "mean"),
    close_rate     = ("CaseClosed_enc", "mean") if "CaseClosed_enc" in df.columns else ("Case Closed", "count"),
).reset_index()

km = KMeans(n_clusters=4, random_state=42, n_init=10)
city_stats["cluster"] = km.fit_predict(city_stats[["total_crimes", "avg_police"]])
cluster_avg = city_stats.groupby("cluster")["total_crimes"].mean().sort_values(ascending=False)
risk_map = {
    cluster_avg.index[0]: ("🔴 High Risk",    "#e74c3c"),
    cluster_avg.index[1]: ("🟠 Medium-High",  "#e67e22"),
    cluster_avg.index[2]: ("🟡 Medium-Low",   "#f1c40f"),
    cluster_avg.index[3]: ("🟢 Low Risk",     "#2ecc71"),
}
city_stats["risk_label"] = city_stats["cluster"].map(lambda c: risk_map[c][0])
city_stats["risk_color"] = city_stats["cluster"].map(lambda c: risk_map[c][1])

print("\n🗺️  CITY RISK LEVELS")
print("-" * 50)
for _, row in city_stats.sort_values("total_crimes", ascending=False).iterrows():
    print(f"  {row['risk_label']}  {row['City']:20s} — {int(row['total_crimes']):,} crimes")

# ─────────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#0d1117")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

BG   = "#161b22"
TEXT = "#c9d1d9"
ACC  = "#58a6ff"

def style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

# --- Plot 1: Historical + Future Total Crime Trend ---
ax1 = fig.add_subplot(gs[0, :2])
style_ax(ax1, "📈 Historical & Predicted Crime Trend (All India)")
ax1.plot(yearly_total_hist["Year"], yearly_total_hist["crime_count"],
         "o-", color=ACC, linewidth=2.5, markersize=7, label="Historical")
ax1.plot(yearly_total_future["Year"], yearly_total_future["crime_count"],
         "s--", color="#f85149", linewidth=2.5, markersize=7, label="Predicted (2025–2029)")
ax1.axvline(x=2024.5, color="#8b949e", linestyle=":", linewidth=1.5)
ax1.text(2024.6, ax1.get_ylim()[0], "Forecast →", color="#8b949e", fontsize=8)
ax1.fill_between(yearly_total_future["Year"], yearly_total_future["crime_count"],
                 alpha=0.15, color="#f85149")
ax1.set_xlabel("Year")
ax1.set_ylabel("Total Crimes")
ax1.legend(facecolor=BG, labelcolor=TEXT, fontsize=9)

# --- Plot 2: City Risk Levels ---
ax2 = fig.add_subplot(gs[0, 2])
style_ax(ax2, "🏙️ City Risk Levels")
city_sorted = city_stats.sort_values("total_crimes", ascending=True).tail(15)
ax2.barh(city_sorted["City"], city_sorted["total_crimes"],
         color=city_sorted["risk_color"], alpha=0.85)
ax2.set_xlabel("Total Crimes")

# --- Plot 3: Top 5 Cities Historical Trend ---
ax3 = fig.add_subplot(gs[1, :2])
style_ax(ax3, "🏙️ Top 6 Cities — Historical Crime Trend")
top_cities = hist_city.groupby("City")["crime_count"].sum().nlargest(6).index
colors_city = ["#58a6ff","#f85149","#3fb950","#d29922","#bc8cff","#39d353"]
for i, city in enumerate(top_cities):
    sub = hist_city[hist_city["City"] == city]
    ax3.plot(sub["Year"], sub["crime_count"], "o-", label=city,
             color=colors_city[i], linewidth=2, markersize=5)
ax3.set_xlabel("Year")
ax3.set_ylabel("Crime Count")
ax3.legend(facecolor=BG, labelcolor=TEXT, fontsize=8, ncol=2)

# --- Plot 4: Future Predictions by Top City ---
ax4 = fig.add_subplot(gs[1, 2])
style_ax(ax4, "🔮 2025–2029 Prediction by City")
top5 = yearly_city.groupby("City")["predicted_crime"].sum().nlargest(5).index
for i, city in enumerate(top5):
    sub = yearly_city[yearly_city["City"] == city]
    ax4.plot(sub["Year"], sub["predicted_crime"], "o-", label=city,
             color=colors_city[i], linewidth=2, markersize=5)
ax4.set_xlabel("Year")
ax4.set_ylabel("Predicted Crimes")
ax4.legend(facecolor=BG, labelcolor=TEXT, fontsize=8)

# --- Plot 5: Crime Type Distribution ---
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, "🔍 Crime Type Distribution")
crime_counts = df["Crime Description"].value_counts().head(10)
bar_colors   = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(crime_counts)))
ax5.barh(crime_counts.index, crime_counts.values, color=bar_colors, alpha=0.85)
ax5.set_xlabel("Count")

# --- Plot 6: Monthly Crime Pattern ---
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, "📅 Monthly Crime Pattern")
monthly_avg = df.groupby("Month").size()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ax6.fill_between(range(1, 13), monthly_avg.values, alpha=0.3, color=ACC)
ax6.plot(range(1, 13), monthly_avg.values, "o-", color=ACC, linewidth=2, markersize=5)
ax6.set_xticks(range(1, 13))
ax6.set_xticklabels(month_names, fontsize=7)
ax6.set_ylabel("Crime Count")

# --- Plot 7: Predicted Crime 2025–2029 Bar ---
ax7 = fig.add_subplot(gs[2, 2])
style_ax(ax7, "📊 Total Predicted Crimes Per Year")
bar_c = ["#f85149","#e8912d","#d29922","#3fb950","#58a6ff"]
bars  = ax7.bar(yearly_total_future["Year"].astype(int),
                yearly_total_future["crime_count"].astype(int),
                color=bar_c, alpha=0.85, width=0.6)
for bar, val in zip(bars, yearly_total_future["crime_count"]):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f"{int(val):,}", ha="center", va="bottom", color=TEXT, fontsize=8, fontweight="bold")
ax7.set_xlabel("Year")
ax7.set_ylabel("Predicted Crimes")

fig.suptitle("🇮🇳  INDIA CRIME DETECTION & 5-YEAR FORECAST DASHBOARD",
             color=TEXT, fontsize=15, fontweight="bold", y=0.99)

plt.savefig("crime_india_dashboard.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("\n✅ Dashboard saved: crime_india_dashboard.png")
print("=" * 60)
