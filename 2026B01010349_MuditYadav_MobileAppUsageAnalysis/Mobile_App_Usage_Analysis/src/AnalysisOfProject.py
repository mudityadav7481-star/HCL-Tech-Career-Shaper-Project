from pathlib import Path
import numpy as np
import pandas as pd
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)
here = Path(__file__).resolve().parent
root = here.parent
csv_name = "24_mobile_app_usage.csv"
folders = [root / "data", here / "data", here, root, Path.cwd() / "data", Path.cwd()]
csv_path = None
for f in folders:
    if (f / csv_name).exists():
        csv_path = f / csv_name
        break
if csv_path is None:
    for f in folders:
        found = sorted(f.glob("*mobile_app_usage*.csv")) if f.exists() else []
        if found:
            csv_path = found[0]
            break
if csv_path is None:
    raise FileNotFoundError(f"Could not find {csv_name}. Put it in the same folder as analysis.py ({here}) or in a 'data' folder next to it.")
out_dir = root / "outputs"
out_dir.mkdir(exist_ok=True)
def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
def avg_duration_for(data, feature):
    rows = data[data["Feature_Used"] == feature]
    if rows.empty:
        return None
    return round(rows["Session_Duration_Mins"].mean(), 2)
def make_table(counts, total, name):
    t = pd.DataFrame({"Sessions": counts, "Share_%": (counts / total * 100).round(2)})
    t.index.name = name
    t.insert(0, "Rank", range(1, len(t) + 1))
    return t
class AppUser:
    def __init__(self, user_id, device_type):
        self.user_id = user_id
        self.device_type = device_type
    def describe(self):
        return f"AppUser(user_id={self.user_id}, device_type={self.device_type})"
header("TASK 1: Load and inspect the app usage dataset")
raw = pd.read_csv(csv_path)
print(f"Dataset file : {csv_path.name}")
print(f"Shape        : {raw.shape[0]} rows x {raw.shape[1]} columns")
print("\nColumn names :", list(raw.columns))
print("\nData types:")
print(raw.dtypes)
print("\nFirst 5 rows:")
print(raw.head())
print("\nBasic statistics of numeric columns:")
print(raw.describe().round(2))
header("TASK 2: Missing values and duplicate Session_ID records")
print("Missing values per column (raw data):")
print(raw.isnull().sum())
dup_rows = raw[raw.duplicated(subset="Session_ID", keep=False)]
dup_count = int(raw.duplicated(subset="Session_ID", keep="first").sum())
identical_count = int(raw.duplicated(keep="first").sum())
print(f"\nExtra (repeated) Session_ID records : {dup_count}")
print(f"Of these, fully identical rows       : {identical_count}")
print("\nSession_IDs that appear more than once (first 6 rows shown):")
print(dup_rows.sort_values("Session_ID").head(6))
df = raw.drop_duplicates(subset="Session_ID", keep="first").copy()
df = df.reset_index(drop=True)
print(f"\nAfter removing duplicates: {len(raw)} -> {len(df)} rows")
header("TASK 3: Handling missing Session_Duration_Mins and User_Rating")
print("Missing values after removing duplicates:")
print(df[["Session_Duration_Mins", "User_Rating"]].isnull().sum())
df["Duration_Was_Missing"] = df["Session_Duration_Mins"].isnull()
df["Rating_Was_Missing"] = df["User_Rating"].isnull()
dur_fill = df.groupby("Feature_Used")["Session_Duration_Mins"].transform("median")
rat_fill = df.groupby("Feature_Used")["User_Rating"].transform("median")
df["Session_Duration_Mins"] = df["Session_Duration_Mins"].fillna(dur_fill).round(1)
df["User_Rating"] = df["User_Rating"].fillna(rat_fill).round(1)
print("\nMethod used : fill with the median of the same Feature_Used.")
print("Reason      : median resists extreme values; feature-wise values are more")
print("              realistic than one global value; no rows are lost.")
print(f"\nDuration values filled : {int(df['Duration_Was_Missing'].sum())}")
print(f"Rating values filled   : {int(df['Rating_Was_Missing'].sum())}")
print("\nMissing values after cleaning:")
print(df.isnull().sum())
header("TASK 4: Datetime conversion and feature extraction")
df["Session_Date"] = pd.to_datetime(df["Session_Date"])
df["Year"] = df["Session_Date"].dt.year
df["Month"] = df["Session_Date"].dt.month
df["Month_Name"] = df["Session_Date"].dt.month_name()
df["Weekday"] = df["Session_Date"].dt.day_name()
df["Hour"] = df["Session_Date"].dt.hour
print("Session_Date data type :", df["Session_Date"].dtype)
print(f"Date range             : {df['Session_Date'].min()}  to  {df['Session_Date'].max()}")
print("\nSample of the new columns:")
print(df[["Session_Date", "Year", "Month", "Month_Name", "Weekday", "Hour"]].head())
header("TASK 5: Key metrics")
total = len(df)
users = df["User_ID"].nunique()
avg_dur = df["Session_Duration_Mins"].mean()
print(f"Total sessions           : {total}")
print(f"Unique users             : {users}")
print(f"Average session duration : {avg_dur:.2f} minutes")
print(f"Average sessions per user: {total / users:.2f}")
header("TASK 6: Most frequently used app features")
feat_counts = df["Feature_Used"].value_counts()
feat_table = make_table(feat_counts, total, "Feature_Used")
n_feat = df["Feature_Used"].nunique()
print(f"Note: the dataset contains only {n_feat} distinct features, so the")
print(f"'top 10' table below shows all {n_feat} of them, ranked.\n")
print(feat_table.head(10))
header("TASK 7: Average session duration by device type")
dev_table = df.groupby("Device_Type")["Session_Duration_Mins"].agg(Sessions="count", Avg_Duration_Mins="mean").round(2).sort_values("Avg_Duration_Mins", ascending=False)
print(dev_table)
print(f"\nLongest average sessions : {dev_table.index[0]} ({dev_table['Avg_Duration_Mins'].iloc[0]} mins)")
print(f"Shortest average sessions: {dev_table.index[-1]} ({dev_table['Avg_Duration_Mins'].iloc[-1]} mins)")
header("TASK 8: Age groups ranked by total sessions")
age_table = make_table(df["Age_Group"].value_counts(), total, "Age_Group")
print(age_table)
header("TASK 9: Average user rating by app feature")
rating_table = df.groupby("Feature_Used")["User_Rating"].agg(Sessions="count", Avg_Rating="mean").round(2).sort_values("Avg_Rating", ascending=False)
rating_table.insert(0, "Rank", range(1, len(rating_table) + 1))
print(rating_table)
header("TASK 10: Peak usage hours and busiest weekdays")
hour_counts = df["Hour"].value_counts().sort_values(ascending=False)
print("Top 5 busiest hours of the day (24-hour clock):")
top_hours = hour_counts.head(5).rename("Sessions").to_frame()
top_hours.index.name = "Hour"
print(top_hours)
peak_hour = int(hour_counts.index[0])
quiet_hour = int(hour_counts.index[-1])
print(f"\nPeak hour     : {peak_hour}:00 ({hour_counts.iloc[0]} sessions)")
print(f"Quietest hour : {quiet_hour}:00 ({hour_counts.iloc[-1]} sessions)")
day_table = make_table(df["Weekday"].value_counts(), total, "Weekday")
print("\nWeekdays ranked by number of sessions:")
print(day_table)
busy_day = day_table.index[0]
quiet_day = day_table.index[-1]
print(f"\nBusiest weekday  : {busy_day}")
print(f"Quietest weekday : {quiet_day}")
month_table = df.groupby(["Month", "Month_Name"]).agg(Sessions=("Session_ID", "count"), Years_Covered=("Year", "nunique")).reset_index()
month_table["Sessions_per_Year_Covered"] = (month_table["Sessions"] / month_table["Years_Covered"]).round(1)
print("\nExtra: sessions by calendar month")
print(month_table[["Month_Name", "Sessions", "Years_Covered", "Sessions_per_Year_Covered"]].to_string(index=False))
print("Note: the data spans Jan 2025 - Oct 2026, so Nov and Dec appear in only one")
print("year. The last column makes the months fairly comparable.")
header("TASK 11: List of top features and tuple of device categories")
top_features = feat_counts.head(10).index.tolist()
print("List of most-used features (rank order):")
for pos, name in enumerate(top_features, start=1):
    print(f"  {pos}. {name}")
device_types = ("Android", "iOS", "Web")
print("\nTuple of fixed device categories:", device_types)
devices_found = set(df["Device_Type"].unique())
if devices_found == set(device_types):
    print("Check: the tuple matches the devices found in the data.")
else:
    print("Warning: data contains devices not in the tuple:", devices_found - set(device_types))
header("TASK 12: Unique users (set) and average duration function")
user_set = set(df["User_ID"])
print(f"Unique users using a set       : {len(user_set)}")
print(f"Unique users using nunique()   : {users}")
print("Both methods agree:", len(user_set) == users)
print("\nAverage session duration per feature (using the function):")
for name in top_features:
    print(f"  {name:<10}: {avg_duration_for(df, name)} mins")
chosen = top_features[0]
print(f"\nSelected feature -> {chosen}: {avg_duration_for(df, chosen)} mins")
print("Unknown feature test  -> 'Gaming':", avg_duration_for(df, "Gaming"))
header("TASK 13: NumPy statistics of session duration")
durations = df["Session_Duration_Mins"].to_numpy()
mean_val = np.mean(durations)
median_val = np.median(durations)
std_val = np.std(durations, ddof=1)
min_val = np.min(durations)
max_val = np.max(durations)
print(f"Array length       : {durations.size}")
print(f"Mean               : {mean_val:.2f} mins")
print(f"Median             : {median_val:.2f} mins")
print(f"Standard deviation : {std_val:.2f} mins (sample, ddof=1)")
print(f"Minimum            : {min_val:.2f} mins")
print(f"Maximum            : {max_val:.2f} mins")
ratings = df["User_Rating"].to_numpy()
corr = np.corrcoef(durations, ratings)[0, 1]
print(f"\nExtra: correlation between duration and rating = {corr:.3f}")
print("Interpretation: a value near 0 means almost no linear association.")
print("Correlation never proves causation.")
header("TASK 14: Unusually long sessions (IQR) and engagement labels")
q1 = np.percentile(durations, 25)
q3 = np.percentile(durations, 75)
iqr = q3 - q1
low_fence = q1 - 1.5 * iqr
high_fence = q3 + 1.5 * iqr
print(f"Q1 (25th percentile) : {q1:.2f}")
print(f"Q3 (75th percentile) : {q3:.2f}")
print(f"IQR                  : {iqr:.2f}")
print(f"Upper fence (Q3 + 1.5*IQR): {high_fence:.2f} mins")
print(f"Lower fence (Q1 - 1.5*IQR): {low_fence:.2f} mins")
df["Is_Long_Outlier"] = df["Session_Duration_Mins"] > high_fence
long_ones = df[df["Is_Long_Outlier"]]
print(f"\nUnusually long sessions found: {len(long_ones)}")
if long_ones.empty:
    print(f"No session exceeds {high_fence:.2f} mins (the longest is {max_val:.1f} mins), so there are no IQR outliers in this dataset.")
    print("Durations are spread fairly evenly between the minimum and maximum.")
else:
    print(long_ones[["Session_ID", "User_ID", "Feature_Used", "Session_Duration_Mins"]].to_string(index=False))
print("\nEngagement label rules (quartile-based):")
print(f"  Low    : duration < {q1:.2f} mins")
print(f"  Medium : {q1:.2f} <= duration <= {q3:.2f} mins")
print(f"  High   : duration > {q3:.2f} mins")
df["Engagement_Level"] = np.select([df["Session_Duration_Mins"] < q1, df["Session_Duration_Mins"] > q3], ["Low", "High"], default="Medium")
eng_counts = df["Engagement_Level"].value_counts().reindex(["Low", "Medium", "High"])
eng_table = pd.DataFrame({"Sessions": eng_counts, "Share_%": (eng_counts / total * 100).round(2)})
eng_table.index.name = "Engagement_Level"
print("\nEngagement label distribution:")
print(eng_table)
print("\nSample of labelled sessions:")
print(df[["Session_ID", "Feature_Used", "Session_Duration_Mins", "Engagement_Level"]].head(8).to_string(index=False))
eng_by_device = pd.crosstab(df["Device_Type"], df["Engagement_Level"], normalize="index")[["Low", "Medium", "High"]].mul(100).round(1)
print("\nEngagement mix by device type (% of that device's sessions):")
print(eng_by_device)
header("TASK 15: AppUser class and recommendations")
user_device = df.groupby("User_ID")["Device_Type"].agg(lambda d: d.value_counts().index[0])
app_users = [AppUser(uid, dev) for uid, dev in user_device.items()]
print(f"AppUser objects created: {len(app_users)}")
print("Examples (device = the user's most-used device):")
for u in app_users[:5]:
    print("  ", u.describe())
users_per_device = pd.Series([u.device_type for u in app_users]).value_counts()
print("\nUsers per primary device (from the AppUser objects):")
print(users_per_device)
worst_feature = rating_table.index[-1]
worst_rating = rating_table["Avg_Rating"].iloc[-1]
best_feature = rating_table.index[0]
best_rating = rating_table["Avg_Rating"].iloc[0]
feat_duration = df.groupby("Feature_Used")["Session_Duration_Mins"].mean().round(2)
short_feature = feat_duration.idxmin()
short_value = feat_duration.min()
weak_device = dev_table.index[-1]
weak_value = dev_table["Avg_Duration_Mins"].iloc[-1]
small_age = age_table.index[-1]
small_age_sessions = age_table["Sessions"].iloc[-1]
big_age = age_table.index[0]
header("FIVE RECOMMENDATIONS TO IMPROVE APP ENGAGEMENT")
recs = [
    f"Improve the {worst_feature} feature: it has the lowest average rating ({worst_rating}/5) versus {best_feature} at {best_rating}/5. Review its user experience and collect feedback.",
    f"Make {short_feature} sessions more engaging: it has the shortest average session ({short_value} mins). Consider better results, suggestions or follow-up content.",
    f"Time notifications and updates around peak usage: {peak_hour}:00 is the busiest hour and {busy_day} the busiest day, while {quiet_hour}:00 and {quiet_day} are quietest. Use quiet slots for re-engagement campaigns.",
    f"Optimise the {weak_device} experience: it has the shortest average session ({weak_value} mins). Test performance and layout on that platform.",
    f"Target the {small_age} age group with onboarding and offers: it has the fewest sessions ({small_age_sessions}), while {big_age} has the most.",
]
for i, text in enumerate(recs, start=1):
    print(f"{i}. {text}")
print("\nLimitations: this is a synthetic teaching dataset, and differences between")
print("groups are small. These recommendations are suggestions based on observed")
print("associations; they do not prove cause and effect.")
header("SAVING OUTPUT FILES")
cleaned_path = out_dir / "cleaned_mobile_app_usage.csv"
df.to_csv(cleaned_path, index=False)
print(f"Cleaned dataset saved : {cleaned_path}")
lines = ["MOBILE APP USAGE & USER ENGAGEMENT ANALYSIS - SUMMARY", "Student: Mudit Yadav | Admission Number: 2026B01010349", ""]
lines.append(f"Rows (raw)               : {len(raw)}")
lines.append(f"Duplicate records removed: {dup_count}")
lines.append(f"Total sessions           : {total}")
lines.append(f"Unique users             : {users}")
lines.append(f"Average duration (mins)  : {avg_dur:.2f}")
lines.append(f"Median duration (mins)   : {median_val:.2f}")
lines.append(f"Std deviation (mins)     : {std_val:.2f}")
lines.append(f"Most used feature        : {top_features[0]}")
lines.append(f"Peak hour                : {peak_hour}:00")
lines.append(f"Busiest weekday          : {busy_day}")
lines.append(f"IQR long-session outliers: {len(long_ones)}")
lines.append("")
lines.append("Recommendations:")
for i, text in enumerate(recs, start=1):
    lines.append(f"{i}. {text}")
summary_path = out_dir / "analysis_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Summary text saved    : {summary_path}")
print("\nAll 15 tasks completed successfully.")

