import os
import glob
import pandas as pd

DATA_FOLDER = "live_training_data"

SENSOR_COLS = [
    "ax", "ay", "az",
    "wx", "wy", "wz",
    "angle_x", "angle_y", "angle_z",
]

def get_label_from_filename(path):
    name = os.path.basename(path).upper()

    if name.startswith("ABD"):
        return "ABD"
    if name.startswith("FEL"):
        return "FEL"
    if name.startswith("IR"):
        return "IR"
    if name.startswith("PEN"):
        return "PEN"
    if name.startswith("REST"):
        return "REST"
    if name.startswith("WRONG"):
        return "WRONG"

    return None

def main():
    files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

    if not files:
        print("No CSV files found in live_training_data")
        return

    rows = []

    for file in files:
        df = pd.read_csv(file)

        label = None

        if "label" in df.columns:
            label = str(df["label"].iloc[0]).upper().strip()
        else:
            label = get_label_from_filename(file)

        if label not in ["ABD", "FEL"]:
            continue

        for col in SENSOR_COLS:
            if col not in df.columns:
                print(f"Skipping {file}, missing {col}")
                continue

        summary = {
            "file": os.path.basename(file),
            "label": label,
        }

        for col in ["angle_x", "angle_y", "angle_z", "ax", "ay", "az", "wx", "wy", "wz"]:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            summary[f"{col}_range"] = values.max() - values.min()
            summary[f"{col}_mean"] = values.mean()

        rows.append(summary)

    result = pd.DataFrame(rows)

    if result.empty:
        print("No ABD/FEL files found.")
        return

    print("\n========== ABD vs FEL AXIS RANGE SUMMARY ==========\n")

    cols_to_show = [
        "label",
        "angle_x_range",
        "angle_y_range",
        "angle_z_range",
        "ax_range",
        "ay_range",
        "az_range",
        "wx_range",
        "wy_range",
        "wz_range",
    ]

    print(result[cols_to_show].groupby("label").mean().round(2))

    print("\nSaved detailed result to abd_fel_axis_check.csv")
    result.to_csv("abd_fel_axis_check.csv", index=False)

if __name__ == "__main__":
    main()
