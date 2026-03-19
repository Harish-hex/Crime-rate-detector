import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
import warnings, os, sys
warnings.filterwarnings("ignore")

df = pd.read_csv("crime_dataset_india.csv")
df["Year"] = pd.to_datetime(df["Date of Occurrence"], dayfirst=True, errors="coerce").dt.year
df = df.dropna(subset=["Year"])
yc = df.groupby(["Year","City"]).size().reset_index(name="count")
F = [2025,2026,2027,2028,2029]

def holt(s, a=0.5, b=0.3):
    L,T = s[0], s[1]-s[0]
    for v in s[1:]: Lp,Tp=L,T; L=a*v+(1-a)*(Lp+Tp); T=b*(L-Lp)+(1-b)*Tp
    return L,T

def pred(cnts, yrs, fut):
    c=cnts.astype(float); lv=c[-1]; ly=yrs[-1]
    sl=LinearRegression().fit(np.array(yrs).reshape(-1,1),c).coef_[0]
    L,T=holt(c); gr=(lv/max(c[0],1))**(1/max(len(c)-1,1))-1; gr=max(min(gr,.2),-.2)
    return {yr:round(max(min(.4*(lv+sl*(yr-ly))+.35*(L+T*(yr-ly))+.25*(lv*(1+gr)**(yr-ly)),c.min()*.7),c.max()*1.7)) for yr in fut}

top5=df.groupby("City").size().nlargest(5).index.tolist()
sys.stdout.write(f"{'City':<20} {'2025':>6} {'2026':>6} {'2027':>6} {'2028':>6} {'2029':>6}  Status\n")
sys.stdout.write("-"*70+"\n")
for cy in top5:
    s=yc[yc["City"]==cy].sort_values("Year")
    p=pred(s["count"].values,s["Year"].values.tolist(),F)
    v=[p[y] for y in F]; st="ALL SAME!" if len(set(v))==1 else "OK-different"
    sys.stdout.write(f"{cy:<20} {v[0]:>6,} {v[1]:>6,} {v[2]:>6,} {v[3]:>6,} {v[4]:>6,}  {st}\n")
sys.stdout.flush()
for f in ["crime_map.html","crime_dashboard.png"]:
    print(f"{f}: {'EXISTS '+str(os.path.getsize(f))+'B' if os.path.exists(f) else 'MISSING'}")
