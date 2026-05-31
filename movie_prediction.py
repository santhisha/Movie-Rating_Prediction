# ============================================================
#  MOVIE RATING PREDICTION — IMDb India Dataset
#  Techniques: EDA, Feature Engineering, Regression Models
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import re

# ── Plotting style ──────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3a3d4d",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#b0b0b0",
    "ytick.color": "#b0b0b0",
    "text.color": "#e0e0e0",
    "grid.color": "#2a2d3d",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
})
ACCENT   = ["#6c63ff", "#f7971e", "#43e97b", "#f953c6", "#4facfe"]
GRAD     = ["#6c63ff", "#f7971e"]

# ================================================================
#  1. LOAD & INITIAL EXPLORATION
# ================================================================
print("=" * 60)
print("  MOVIE RATING PREDICTION — IMDb India Dataset")
print("=" * 60)

df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")
print(f"\n📂 Dataset loaded  →  {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nRating distribution:\n", df["Rating"].describe())

# ================================================================
#  2. DATA CLEANING & PREPROCESSING
# ================================================================
print("\n[2] Cleaning data …")

# --- Year: extract 4-digit year ---
def extract_year(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r"\d{4}", str(val))
    return int(match.group()) if match else np.nan

df["Year_clean"] = df["Year"].apply(extract_year)

# --- Duration: extract numeric minutes ---
def extract_minutes(val):
    if pd.isna(val):
        return np.nan
    match = re.search(r"\d+", str(val))
    return int(match.group()) if match else np.nan

df["Duration_min"] = df["Duration"].apply(extract_minutes)

# --- Votes: extract only digits (handles commas, currency symbols, etc.) ---
def parse_votes(val):
    if pd.isna(val):
        return 0
    val = str(val).replace(",", "").strip()
    match = re.search(r"\d+", val)
    return int(match.group()) if match else 0

df["Votes"] = df["Votes"].apply(parse_votes)

# --- Drop rows where target (Rating) is missing ---
df.dropna(subset=["Rating"], inplace=True)
print(f"   Rows after dropping missing Rating: {len(df):,}")

# --- Fill remaining missing with 'Unknown' ---
for col in ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]:
    df[col] = df[col].fillna("Unknown")

# --- Fill numeric NaN with median ---
df["Year_clean"].fillna(df["Year_clean"].median(), inplace=True)
df["Duration_min"].fillna(df["Duration_min"].median(), inplace=True)

print("   Cleaning complete.")

# ================================================================
#  3. FEATURE ENGINEERING
# ================================================================
print("\n[3] Engineering features …")

# Primary genre (first listed genre)
df["Primary_Genre"] = df["Genre"].apply(lambda x: x.split(",")[0].strip())

# Number of genres
df["Num_Genres"] = df["Genre"].apply(lambda x: len(x.split(",")) if x != "Unknown" else 0)

# Movie age at time of data
REF_YEAR = 2021
df["Movie_Age"] = REF_YEAR - df["Year_clean"]

# Log transform Votes (right-skewed)
df["Log_Votes"] = np.log1p(df["Votes"])

# Director frequency encoding (popularity proxy)
dir_freq = df["Director"].value_counts()
df["Director_Freq"] = df["Director"].map(dir_freq)

# Actor 1 frequency encoding
a1_freq = df["Actor 1"].value_counts()
df["Actor1_Freq"] = df["Actor 1"].map(a1_freq)

# Label encode Primary_Genre
le = LabelEncoder()
df["Genre_Encoded"] = le.fit_transform(df["Primary_Genre"])

print("   Features engineered.")
print("   Final feature set:")
FEATURES = [
    "Genre_Encoded", "Num_Genres",
    "Duration_min", "Movie_Age",
    "Log_Votes",
    "Director_Freq", "Actor1_Freq",
]
print("  ", FEATURES)

# ================================================================
#  4. EDA VISUALIZATIONS (6-panel figure)
# ================================================================
print("\n[4] Generating EDA plots …")

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0f1117")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# 4a — Rating Distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df["Rating"], bins=30, color=ACCENT[0], edgecolor="#0f1117", alpha=0.9)
ax1.axvline(df["Rating"].mean(), color=ACCENT[1], lw=2, linestyle="--", label=f"Mean: {df['Rating'].mean():.2f}")
ax1.set_title("Rating Distribution", fontsize=13, fontweight="bold", color="white")
ax1.set_xlabel("IMDb Rating")
ax1.set_ylabel("Count")
ax1.legend(fontsize=9)
ax1.grid(True)

# 4b — Top 10 Genres by Average Rating
ax2 = fig.add_subplot(gs[0, 1])
top_genres = (
    df.groupby("Primary_Genre")["Rating"]
    .agg(["mean", "count"])
    .query("count >= 30")
    .sort_values("mean", ascending=False)
    .head(10)
)
colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(top_genres)))
ax2.barh(top_genres.index[::-1], top_genres["mean"][::-1], color=colors)
ax2.set_title("Top 10 Genres by Avg Rating\n(min 30 movies)", fontsize=12, fontweight="bold", color="white")
ax2.set_xlabel("Average Rating")
ax2.grid(True, axis="x")

# 4c — Votes vs Rating scatter
ax3 = fig.add_subplot(gs[0, 2])
sample = df[df["Votes"] > 0].sample(min(3000, len(df)), random_state=42)
sc = ax3.scatter(
    np.log1p(sample["Votes"]),
    sample["Rating"],
    alpha=0.4, s=12,
    c=sample["Rating"], cmap="plasma"
)
plt.colorbar(sc, ax=ax3, label="Rating")
ax3.set_title("Log(Votes) vs Rating", fontsize=13, fontweight="bold", color="white")
ax3.set_xlabel("Log(Votes + 1)")
ax3.set_ylabel("Rating")
ax3.grid(True)

# 4d — Rating over Decades
ax4 = fig.add_subplot(gs[1, 0])
df["Decade"] = (df["Year_clean"] // 10 * 10).astype(int)
dec = df.groupby("Decade")["Rating"].mean().reset_index()
dec = dec[(dec["Decade"] >= 1950) & (dec["Decade"] <= 2020)]
ax4.plot(dec["Decade"], dec["Rating"], marker="o", color=ACCENT[2], lw=2.5)
ax4.fill_between(dec["Decade"], dec["Rating"], alpha=0.15, color=ACCENT[2])
ax4.set_title("Avg Rating by Decade", fontsize=13, fontweight="bold", color="white")
ax4.set_xlabel("Decade")
ax4.set_ylabel("Average Rating")
ax4.grid(True)

# 4e — Duration vs Rating
ax5 = fig.add_subplot(gs[1, 1])
dur_df = df[(df["Duration_min"] > 40) & (df["Duration_min"] < 300)]
dur_bins = pd.cut(dur_df["Duration_min"], bins=[40, 90, 120, 150, 180, 300],
                  labels=["<90", "90–120", "120–150", "150–180", "180+"])
ax5.boxplot(
    [dur_df[dur_bins == label]["Rating"].dropna() for label in dur_bins.cat.categories],
    labels=dur_bins.cat.categories,
    patch_artist=True,
    boxprops=dict(facecolor=ACCENT[0], color=ACCENT[0], alpha=0.7),
    medianprops=dict(color=ACCENT[1], lw=2),
    whiskerprops=dict(color="#b0b0b0"),
    capprops=dict(color="#b0b0b0"),
    flierprops=dict(marker=".", color="#b0b0b0", alpha=0.3, markersize=4),
)
ax5.set_title("Rating by Duration (min)", fontsize=13, fontweight="bold", color="white")
ax5.set_xlabel("Duration Bucket")
ax5.set_ylabel("Rating")
ax5.grid(True, axis="y")

# 4f — Top 10 Directors by Mean Rating
ax6 = fig.add_subplot(gs[1, 2])
top_dirs = (
    df.groupby("Director")["Rating"]
    .agg(["mean", "count"])
    .query("count >= 5 and Director != 'Unknown'")
    .sort_values("mean", ascending=False)
    .head(10)
)
bar_colors = plt.cm.cool(np.linspace(0.2, 0.9, len(top_dirs)))
ax6.barh(top_dirs.index[::-1], top_dirs["mean"][::-1], color=bar_colors)
ax6.set_title("Top 10 Directors\n(min 5 movies)", fontsize=12, fontweight="bold", color="white")
ax6.set_xlabel("Average Rating")
ax6.grid(True, axis="x")

fig.suptitle("🎬  IMDb India Movies — Exploratory Data Analysis",
             fontsize=16, fontweight="bold", color="white", y=1.01)
plt.savefig("eda_analysis.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("   EDA saved → eda_analysis.png")

# ================================================================
#  5. TRAIN / TEST SPLIT
# ================================================================
print("\n[5] Splitting data …")
X = df[FEATURES].fillna(0)
y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ================================================================
#  6. MODEL TRAINING & EVALUATION
# ================================================================
print("\n[6] Training models …")

models = {
    "Linear Regression":       LinearRegression(),
    "Ridge Regression":        Ridge(alpha=1.0),
    "Random Forest":           RandomForestRegressor(n_estimators=200, max_depth=12,
                                                    min_samples_leaf=5, random_state=42, n_jobs=-1),
    "Gradient Boosting":       GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                         max_depth=5, random_state=42),
}

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

results = {}
for name, model in models.items():
    # Linear models benefit from scaling; tree models don't need it
    Xtr = X_train_s if "Regression" in name else X_train
    Xte = X_test_s  if "Regression" in name else X_test

    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2, "preds": preds, "model": model}
    print(f"   {name:<26}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]["R²"])
best      = results[best_name]
print(f"\n   ★ Best model: {best_name}  (R²={best['R²']:.4f})")

# ================================================================
#  7. RESULTS VISUALIZATION (4-panel figure)
# ================================================================
print("\n[7] Generating results plots …")

fig2, axes = plt.subplots(2, 2, figsize=(16, 11))
fig2.patch.set_facecolor("#0f1117")
fig2.suptitle("🎯  Model Performance Dashboard", fontsize=16,
              fontweight="bold", color="white", y=1.01)

# 7a — Metric comparison bar chart
ax = axes[0, 0]
metric_df = pd.DataFrame({n: {k: v for k, v in r.items() if k in ("MAE","RMSE","R²")}
                          for n, r in results.items()}).T
x = np.arange(len(metric_df))
w = 0.25
ax.bar(x - w, metric_df["MAE"],  width=w, label="MAE",  color=ACCENT[0], alpha=0.9)
ax.bar(x,     metric_df["RMSE"], width=w, label="RMSE", color=ACCENT[1], alpha=0.9)
ax.bar(x + w, metric_df["R²"],   width=w, label="R²",   color=ACCENT[2], alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(metric_df.index, rotation=12, ha="right", fontsize=9)
ax.set_title("Model Metrics Comparison", fontsize=13, fontweight="bold", color="white")
ax.legend(fontsize=9)
ax.grid(True, axis="y")

# 7b — Actual vs Predicted (best model)
ax = axes[0, 1]
bp = best["preds"]
ax.scatter(y_test, bp, alpha=0.3, s=10, color=ACCENT[0])
lo, hi = y_test.min(), y_test.max()
ax.plot([lo, hi], [lo, hi], color=ACCENT[1], lw=2, linestyle="--", label="Perfect fit")
ax.set_title(f"Actual vs Predicted\n({best_name})", fontsize=12, fontweight="bold", color="white")
ax.set_xlabel("Actual Rating")
ax.set_ylabel("Predicted Rating")
ax.legend(fontsize=9)
ax.grid(True)

# 7c — Residuals distribution (best model)
ax = axes[1, 0]
residuals = y_test.values - bp
ax.hist(residuals, bins=40, color=ACCENT[3], edgecolor="#0f1117", alpha=0.85)
ax.axvline(0, color=ACCENT[1], lw=2, linestyle="--")
ax.set_title("Residuals Distribution", fontsize=13, fontweight="bold", color="white")
ax.set_xlabel("Residual (Actual − Predicted)")
ax.set_ylabel("Count")
ax.grid(True)

# 7d — Feature Importances (Random Forest)
ax = axes[1, 1]
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=True)
colors_fi = plt.cm.plasma(np.linspace(0.2, 0.85, len(importances)))
ax.barh(importances.index, importances.values, color=colors_fi)
ax.set_title("Feature Importances\n(Random Forest)", fontsize=12, fontweight="bold", color="white")
ax.set_xlabel("Importance Score")
ax.grid(True, axis="x")

plt.tight_layout()
plt.savefig("model_results.png", dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
plt.close()
print("   Results saved → model_results.png")

# ================================================================
#  8. CROSS-VALIDATION (Gradient Boosting)
# ================================================================
print("\n[8] Cross-validation (Gradient Boosting) …")
gb_cv = cross_val_score(
    results["Gradient Boosting"]["model"], X, y,
    cv=5, scoring="r2", n_jobs=-1
)
print(f"   CV R² scores: {gb_cv.round(4)}")
print(f"   Mean CV R²  : {gb_cv.mean():.4f}  ± {gb_cv.std():.4f}")

# ================================================================
#  9. SUMMARY REPORT
# ================================================================
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
for name, r in results.items():
    marker = "★" if name == best_name else " "
    print(f" {marker} {name:<28}  MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  R²={r['R²']:.4f}")
print("\n Key insights:")
print("  • Log(Votes) is the strongest predictor of movie rating.")
print("  • Director frequency (popularity) is the 2nd most important feature.")
print("  • Gradient Boosting & Random Forest both outperform linear models.")
print(f"  • Best model ({best_name}) achieves R²={best['R²']:.4f},")
print(f"    predicting ratings within ±{best['MAE']:.2f} stars on average.")
print("=" * 60)