import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
import warnings, os
warnings.filterwarnings("ignore")

df = pd.read_csv("crime_dataset_india.csv")
df["Date"] = pd.to_datetime(df["Date of Occurrence"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Date"])
df["Year"] = df["Date"].dt.year
yearly_city = df.groupby(["Year","City"]).size().reset_index(name="count")
FUTURE = [2025,2026,2027,2028,2029]

def holt_smooth(series, alpha=0.5, beta=0.3):
    L = series[0]; T = series[1] - series[0]
    for val in series[1:]:
        Lp,Tp = L,T
        L = alpha * val + (1-alpha)*(Lp+Tp)
        T = beta*(L-Lp)+(1-beta)*Tp
    return L, T

def predict_city(cnts, hist_yrs, future_yrs):
    cnts = cnts.astype(float)
    last_val = cnts[-1]; last_yr = hist_yrs[-1]
    X = np.array(hist_yrs).reshape(-1,1)
    lr = LinearRegression().fit(X, cnts); slope = lr.coef_[0]
    L, T = holt_smooth(cnts)
    cagr = (last_val/max(cnts[0],1))**(1/max(len(cnts)-1,1))-1
    cagr = max(min(cagr,0.20),-0.20)
    preds = {}
    for yr in future_yrs:
        s = yr - last_yr
        m1 = last_val + slope*s; m2 = L+T*s; m3 = last_val*((1+cagr)**s)
        pred = 0.40*m1 + 0.35*m2 + 0.25*m3
        pred = max(pred, cnts.min()*0.70); pred = min(pred, cnts.max()*1.70)
        preds[yr] = round(pred)
    return preds

top5 = df.groupby("City").size().nlargest(5).index.tolist()
print(f"{'City':<20} {'2025':>7} {'2026':>7} {'2027':>7} {'2028':>7} {'2029':>7}  Status")
print("-"*75)
for city in top5:
    sub = yearly_city[yearly_city["City"]==city].sort_values("Year")
    p = predict_city(sub["count"].values, sub["Year"].values.tolist(), FUTURE)
    vals = [p[y] for y in FUTURE]
    status = "ALL SAME! ❌" if len(set(vals))==1 else "DIFFERENT ✅"
    print(f"{city:<20} {vals[0]:>7,} {vals[1]:>7,} {vals[2]:>7,} {vals[3]:>7,} {vals[4]:>7,}  {status}")

print()
print("Output files:")
for f in ["crime_map.html","crime_dashboard.png"]:
    if os.path.exists(f):
        sz = os.path.getsize(f)
        print(f"  {f}  ({sz:,} bytes) ✅")
    else:
        print(f"  {f}  MISSING ❌")
