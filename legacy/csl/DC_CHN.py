import json
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from csl.models.dc import run_dixon_coles_model
from csl.paths import data_output_dir, data_raw_dir, model_meta_json

input_csv_path = os.path.join(data_raw_dir(), "CHN_Super League.csv")
output_csv_path = os.path.join(data_output_dir(), "CHN_team_stats.csv")

run_dixon_coles_model(input_csv_path, output_csv_path, xi=0.001)

# Record when the model was (re)fit. The dashboard meta export reads this so the UI
# can show the model-update time distinctly from the far more frequent odds fetch.
# Only model/full runs execute this file, so the timestamp survives odds-only refreshes.
model_updated_at = (
    datetime.now(timezone.utc).astimezone(ZoneInfo("Europe/London")).isoformat(timespec="seconds")
)
with open(model_meta_json(), "w", encoding="utf-8") as fh:
    json.dump({"model_updated_at": model_updated_at}, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
