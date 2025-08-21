import os
import glob
import pandas as pd
import numpy as np

# Heuristics for outlier detection and cleaning
# - invalid bbox (x2<=x1 or y2<=y1)
# - bbox too small (area < min_area)
# - extreme jumps in bbox center (> max_center_jump pixels between consecutive frames of same track)
# - duplicated timestamps within same (video,id)
# - cap velocities to reasonable percentile per entire train set
# - optional label whitelist to remove odd classes (e.g., 'kite')

MIN_AREA = 12 * 12   # pixels^2
MAX_CENTER_JUMP = 120  # pixels per frame step (tuned per dataset)
VEL_CLIP_Q = 0.995     # clip extreme velocity outliers
# Labels to keep; set to None to keep all
ALLOWED_LABELS = {"car", "truck", "bus", "motorcycle", "person", "bicycle"}


def add_video_column(df: pd.DataFrame, video_name: str) -> pd.DataFrame:
    df = df.copy()
    df["video"] = video_name
    return df


def read_all_train_csv(root: str = "data/train") -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(root, "*.csv")))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Skip {f}: {e}")
            continue
        base = os.path.splitext(os.path.basename(f))[0]
        df = add_video_column(df, base)
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No CSV found under data/train")
    return pd.concat(dfs, ignore_index=True)


def filter_labels(df: pd.DataFrame) -> pd.DataFrame:
    if not ALLOWED_LABELS:
        return df
    df = df.copy()
    before = len(df)
    df = df[df["label"].isin(ALLOWED_LABELS)]
    print(f"Filtered labels: {before - len(df)} rows removed (kept {sorted(ALLOWED_LABELS)})")
    return df


def basic_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # valid bbox
    df = df[(df.x2 > df.x1) & (df.y2 > df.y1)]
    # area filter
    area = (df.x2 - df.x1) * (df.y2 - df.y1)
    df = df[area >= MIN_AREA]
    # finite values only
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def remove_center_jump_outliers(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (video, oid), g in df.groupby(["video", "id"]):
        g = g.sort_values(["second", "frame"])  # second then frame within video
        # compute center
        cx = (g.x1 + g.x2) / 2.0
        cy = (g.y1 + g.y2) / 2.0
        dcx = cx.diff().abs()
        dcy = cy.diff().abs()
        ok = (dcx <= MAX_CENTER_JUMP) & (dcy <= MAX_CENTER_JUMP)
        # always keep first row
        if len(ok) > 0:
            ok.iloc[0] = True
        rows.append(g.loc[ok])
    return pd.concat(rows, ignore_index=True)


def deduplicate_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop duplicated frames for same (video,id,label)
    df = df.drop_duplicates(subset=["video", "id", "label", "frame"], keep="first")
    # Ensure sorted
    df = df.sort_values(["video", "id", "frame", "second"]).reset_index(drop=True)
    return df


def clip_velocity_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["vx1", "vy1", "vx2", "vy2", "speed_kmh"]:
        if col not in df:
            continue
        vals = df[col].to_numpy()
        # robust percentile clip ignoring NaNs
        lo = np.nanpercentile(vals, (1 - VEL_CLIP_Q) * 100)
        hi = np.nanpercentile(vals, VEL_CLIP_Q * 100)
        df[col] = np.clip(vals, lo, hi)
    return df


def relabel_ids_per_video(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Map id per video so that tracks don't mix across videos
    new_ids = []
    for video, g in df.groupby("video"):
        # stable mapping: original id -> 0..N-1 in order of first appearance
        order = (
            g.sort_values(["frame", "second"])  # appearance order
             .groupby("id").head(1)
             .sort_values(["frame", "second"]).id.unique()
        )
        mapping = {oid: i for i, oid in enumerate(order)}
        new_ids.append(g.assign(id=g.id.map(mapping)))
    return pd.concat(new_ids, ignore_index=True)


def recompute_velocities(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute vx1,vy1,vx2,vy2 from bbox deltas per frame to avoid zero-initial Kalman velocities.
    Drops the first row of each (video,id) track where velocity is undefined.
    """
    parts = []
    for (video, oid), g in df.groupby(["video", "id"], sort=False):
        g = g.sort_values(["frame", "second"]).copy()
        # delta per frame count (handle frame gaps)
        dt = g["frame"].diff()
        dt = dt.replace(0, np.nan)
        for col, dcol in [("x1", "vx1"), ("y1", "vy1"), ("x2", "vx2"), ("y2", "vy2")]:
            v = g[col].diff() / dt
            g[dcol] = v.replace([np.inf, -np.inf], np.nan)
        # drop first row (where dt is NaN)
        g = g.iloc[1:]
        # fill any remaining NaNs in velocities by forward/backward fill within the track, then 0
        for dcol in ["vx1", "vy1", "vx2", "vy2"]:
            g[dcol] = g[dcol].ffill().bfill().fillna(0.0)
        parts.append(g)
    if not parts:
        return df.iloc[0:0]
    out = pd.concat(parts, ignore_index=True)
    return out


def compute_outlier_mask(df: pd.DataFrame) -> pd.Series:
    # Hard bounds (very off-screen)
    cond_bounds = (df.x1 < -50) | (df.y1 < -50) | (df.x2 > 2000) | (df.y2 > 2000)
    # Extreme area jump within same (video,id)
    area = (df.x2 - df.x1) * (df.y2 - df.y1)
    area_jump = area.groupby([df.video, df.id]).diff().abs()
    area_med = area.median() if len(area) else 0
    cond_area_jump = area_jump > (area_med * 5)
    mask = cond_bounds | cond_area_jump.fillna(False)
    return mask


def main():
    df = read_all_train_csv()
    print(f"Loaded {len(df)} rows from train CSVs")

    df = filter_labels(df)

    df = basic_filters(df)
    print(f"After basic filters: {len(df)} rows")

    df = remove_center_jump_outliers(df)
    print(f"After center jump filtering: {len(df)} rows")

    df = deduplicate_and_sort(df)
    print(f"After dedup/sort: {len(df)} rows")

    # Relabel per video then recompute velocities from bbox deltas
    df = relabel_ids_per_video(df)
    print(
        f"After relabel per video: {len(df)} rows; unique videos={df.video.nunique()}, total tracks={df.groupby(['video','id']).ngroups}"
    )

    df = recompute_velocities(df)
    print(f"After recomputing velocities and dropping first row per track: {len(df)} rows")

    df = clip_velocity_outliers(df)
    print(f"After velocity clipping: {len(df)} rows")

    # Compute and drop flagged outliers (do not save a separate file)
    outlier_mask = compute_outlier_mask(df)
    num_flagged = int(outlier_mask.sum())
    print(f"Flagged hard outliers (dropped): {num_flagged}")

    # Only keep clean rows
    df_clean = df.loc[~outlier_mask].copy()

    # Save only merged cleaned file
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    merged_path = os.path.join(out_dir, "train_merged_clean.csv")
    df_clean.to_csv(merged_path, index=False)
    print(f"Saved: {merged_path} ({len(df_clean)} rows)")

    # No per-video CSV outputs


if __name__ == "__main__":
    main()
