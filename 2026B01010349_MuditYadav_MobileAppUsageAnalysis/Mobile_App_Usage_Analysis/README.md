# Mobile App Usage & User Engagement Analysis

 | Python, NumPy & Pandas Individual Industry Project
**Student:** Mudit Yadav | **Admission Number:** 2026B01010349

## Overview
This project analyses 1,015 mobile app session records (`24_mobile_app_usage.csv`) to understand
which features people use, how long they stay, how they rate the app, and when they are most active.
It completes all 15 tasks of the project brief and ends with five data-backed recommendations to
improve engagement.

## Repository structure
```
/SRS/project_SRS.docx                         Software Requirements Specification
/src/analysis.py                              Complete analysis program (Tasks 1-15)
/data/24_mobile_app_usage.csv                 Input dataset
/outputs/cleaned_mobile_app_usage.csv         Cleaned dataset (created by the program)
/outputs/analysis_summary.txt                 Short text summary (created by the program)
/outputs/console_output.txt                   Full program output from one run
/presentation/Mobile_App_Usage_Presentation.pptx   Project presentation
README.md  requirements.txt
```

## How to run
1. Install Python 3.15 or newer.
2. Install the libraries: `pip install -r requirements.txt` (NumPy and Pandas only).
3. From the project folder run: `python src/analysis.py`
   (it also works from inside `src`, and from the VS Code "Run" button).

The program prints every result to the terminal, then saves the cleaned dataset and a short
summary in `/outputs`.

## What the program does
| Task | Summary |
|------|---------|
| 1 | Load the CSV; show shape, columns, data types, sample rows, statistics |
| 2 | Count missing values; detect duplicate `Session_ID` records |
| 3 | Remove duplicates; fill missing duration and rating with the median of the same feature |
| 4 | Convert `Session_Date` to datetime; extract year, month, weekday, hour |
| 5 | Total sessions, unique users, average session duration |
| 6 | Feature popularity table (the data has 8 features, so all 8 are ranked) |
| 7 | Average session duration by device type |
| 8 | Age groups ranked by number of sessions |
| 9 | Average rating by feature |
| 10 | Peak hours and busiest weekdays (plus a monthly summary) |
| 11 | List of most-used features; tuple of device categories |
| 12 | Set of unique users; function for average duration of a selected feature |
| 13 | NumPy mean, median, standard deviation of session duration |
| 14 | IQR outlier check; Low / Medium / High engagement labels (quartile rules) |
| 15 | `AppUser` class and five recommendations |

## Key decisions
- **Duplicates:** 15 repeated `Session_ID` rows were exact copies; the first copy is kept (1,015 to 1,000 rows).
- **Missing values:** 25 durations and 25 ratings are filled with the median of the same `Feature_Used`
  (median ignores extremes; no rows lost). Flag columns record which values were filled.
- **Standard deviation:** sample standard deviation (`ddof=1`).
- **Engagement labels:** Low below Q1, Medium from Q1 to Q3, High above Q3.
- **IQR result:** no session exceeds the upper fence, so there are 0 unusually long sessions. This is reported honestly.
- **No hard-coded answers:** every number is calculated from the CSV.

## Limitations
The dataset is synthetic and differences between groups are small. Results describe associations
in this data and do not prove cause and effect.
