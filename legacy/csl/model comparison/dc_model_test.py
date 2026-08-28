# ============================================================================
# STALE RESULTS WARNING — this script's published conclusions predate the fix
# for penaltyblog's silent truncation of non-integer goal targets. It fits a
# penaltyblog family on xG-blend targets, and penaltyblog's BaseGoalsModel
# coerces those to int before the likelihood, so every lambda here is ~27% too
# low (score equation sum(w*lambda)/sum(w*y) = 0.733 instead of 1.0). Model
# families cannot be meaningfully ranked against each other on a truncated fit —
# they were all equally broken. See src/csl/models/continuous_poisson.py and
# backtest/verify_truncation_fix.py. Re-run against ContinuousPoissonGoalModel
# before citing any number from this file.
# ============================================================================
import os
import sys

import numpy as np
import pandas as pd
import penaltyblog as pb
import matplotlib.pyplot as plt
from tqdm import tqdm

# Resolve the league CSV relative to the repo root so the script is portable
# (the old hardcoded /Users/jordan/Projects/... path no longer exists).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO_ROOT, "data", "raw_data", "CHN_Super League.csv")

# The CSV dates are DD/MM/YYYY (or ISO); a bare to_datetime() day/month-swaps
# DMY and drops every day>12 row. Use the production parser instead.
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from csl.date_utils import parse_date_only_series  # noqa: E402

df = pd.read_csv(CSV)
df['Date'] = parse_date_only_series(df['Date'])
df = df.sort_values('Date').set_index('Date', drop=False)

# Map result to numeric
res_map = {'H': 0, 'D': 1, 'A': 2}
df['res_numeric'] = df['Res'].map(res_map)

start_date = df.query('Season == 2025')['Date'].min()
run_dates = df['Date'][df['Date'] >= start_date].unique()
print(f'Start date: {start_date.date()}, Run dates: {len(run_dates)}')

predictions = []
observed = []

for date in tqdm(run_dates, desc='Processing dates'):
    lookback = pd.Timestamp(date) - pd.DateOffset(years=1)
    train = df[(df['Date'] < date) & (df['Date'] >= lookback)]
    test = df[df['Date'] == date]

    if len(train) == 0:
        continue

    clf = pb.models.DixonColesGoalModel(
        train['HExpG+'],
        train['AExpG+'],
        train['Home'],
        train['Away'],
    )

    try:
        clf.fit()
        if len(test) > 0:
            homes = test['Home'].values
            aways = test['Away'].values
            outcomes = test['res_numeric'].values
            for i in range(len(test)):
                try:
                    prediction = clf.predict(homes[i], aways[i])
                    predictions.append(prediction.home_draw_away)
                    observed.append(outcomes[i])
                except Exception as e:
                    continue
    except Exception as e:
        continue

rps = pb.metrics.rps_average(predictions, observed)
print(f'\nTotal predictions made: {len(predictions)}')
print(f'DixonColes RPS (HExpG+ / AExpG+): {rps}')