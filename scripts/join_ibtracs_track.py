import argparse
import re
import unicodedata

import pandas as pd


def normalize_name(value):
    """
    Normalize storm names so INCYDE and IBTrACS names
    can be matched reliably.
    """

    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    # Remove Unicode accents / special characters
    value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    # Remove everything except letters and numbers
    value = re.sub(r"[^A-Z0-9]", "", value)

    if not value or value in {"NAN", "NONE", "NULL"}:
        return None

    # Handle known IBTrACS / INCYDE naming differences
    aliases = {
        "HIKAAHIKKA": "HIKKA",
        "KYAARKYARR": "KYAAR",
        "BULBULMATMO": "BULBUL",
        "GULABSHAHEENGU": "GULAB",

        # INCYDE calls this storm SHAHEEN,
        # while IBTrACS uses GULAB:SHAHEEN-GU
        "SHAHEEN": "GULAB",
    }

    return aliases.get(value, value)


def main():
    parser = argparse.ArgumentParser(
        description="Join INCYDE cyclone images with IBTrACS track data."
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to INCYDE manifest CSV"
    )

    parser.add_argument(
        "--ibtracs",
        required=True,
        help="Path to cleaned IBTrACS CSV"
    )

    parser.add_argument(
        "--output",
        default="incyde_manifest_with_track.csv",
        help="Output CSV path"
    )

    parser.add_argument(
        "--tolerance_hours",
        type=float,
        default=6.0,
        help="Maximum time difference allowed for matching"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # LOAD FILES
    # ---------------------------------------------------------

    print("Loading manifest...")

    manifest = pd.read_csv(
        args.manifest,
        low_memory=False
    )

    print("Loading IBTrACS...")

    ibtracs = pd.read_csv(
        args.ibtracs,
        low_memory=False
    )

    print(f"INCYDE rows loaded: {len(manifest)}")
    print(f"IBTrACS rows loaded: {len(ibtracs)}")

    # ---------------------------------------------------------
    # CREATE STABLE ROW ID
    # ---------------------------------------------------------

    manifest["_ROW_ID"] = range(len(manifest))

    # ---------------------------------------------------------
    # CONVERT TIMESTAMPS
    # ---------------------------------------------------------

    manifest["timestamp"] = pd.to_datetime(
        manifest["timestamp"],
        errors="coerce",
        format="mixed"
    )

    ibtracs["ISO_TIME"] = pd.to_datetime(
        ibtracs["ISO_TIME"],
        errors="coerce",
        format="mixed"
    )

    # ---------------------------------------------------------
    # NORMALIZE STORM NAMES
    # ---------------------------------------------------------

    manifest["_STORM_KEY"] = manifest["name"].apply(
        normalize_name
    )

    ibtracs["_STORM_KEY"] = ibtracs["NAME"].apply(
        normalize_name
    )

    print(
        "Unique normalized INCYDE storm names:",
        manifest["_STORM_KEY"].nunique()
    )

    # ---------------------------------------------------------
    # PREPARE IBTRACS TRACK DATA
    # ---------------------------------------------------------

    track_columns = [
        "_STORM_KEY",
        "ISO_TIME",
        "LAT",
        "LON",
        "INTENSITY_WIND_KT",
        "INTENSITY_PRES_MB"
    ]

    track = ibtracs[
        [c for c in track_columns if c in ibtracs.columns]
    ].copy()

    track = track.dropna(
        subset=["_STORM_KEY", "ISO_TIME"]
    )

    track = track.sort_values(
        ["_STORM_KEY", "ISO_TIME"]
    )

    # ---------------------------------------------------------
    # PREPARE MANIFEST FOR MATCHING
    # ---------------------------------------------------------

    result = manifest.copy()

    # Create empty columns for track information
    result["LAT"] = pd.NA
    result["LON"] = pd.NA
    result["INTENSITY_WIND_KT"] = pd.NA
    result["INTENSITY_PRES_MB"] = pd.NA

    tolerance = pd.Timedelta(
        hours=args.tolerance_hours
    )

    print()
    print("Starting track matching...")
    print()

    matched_rows = 0
    timestamp_misses = 0
    unmatched_groups = {}

    # ---------------------------------------------------------
    # MATCH STORM BY STORM
    # ---------------------------------------------------------

    for storm_key, group in manifest.groupby(
        "_STORM_KEY",
        dropna=False
    ):

        # -----------------------------------------------------
        # Invalid / missing storm name
        # -----------------------------------------------------

        if pd.isna(storm_key):

            unmatched_groups["NAN"] = len(group)
            continue

        # -----------------------------------------------------
        # Find corresponding IBTrACS storm
        # -----------------------------------------------------

        candidates = track[
            track["_STORM_KEY"] == storm_key
        ].copy()

        if candidates.empty:

            unmatched_groups[storm_key] = len(group)
            continue

        # -----------------------------------------------------
        # Only rows with valid timestamps can be matched
        # -----------------------------------------------------

        valid_manifest = group[
            group["timestamp"].notna()
        ].copy()

        if valid_manifest.empty:
            timestamp_misses += len(group)
            continue

        valid_manifest = valid_manifest.sort_values(
            "timestamp"
        )

        candidates = candidates.sort_values(
            "ISO_TIME"
        )

        # -----------------------------------------------------
        # Nearest timestamp match
        # -----------------------------------------------------

        merged = pd.merge_asof(
            valid_manifest[
                ["_ROW_ID", "timestamp"]
            ],
            candidates,
            left_on="timestamp",
            right_on="ISO_TIME",
            direction="nearest",
            tolerance=tolerance
        )

        # -----------------------------------------------------
        # Copy matched track information back
        # -----------------------------------------------------

        for _, row in merged.iterrows():

            row_id = row["_ROW_ID"]

            if pd.notna(row["ISO_TIME"]):

                result.loc[
                    result["_ROW_ID"] == row_id,
                    "LAT"
                ] = row.get("LAT", pd.NA)

                result.loc[
                    result["_ROW_ID"] == row_id,
                    "LON"
                ] = row.get("LON", pd.NA)

                result.loc[
                    result["_ROW_ID"] == row_id,
                    "INTENSITY_WIND_KT"
                ] = row.get(
                    "INTENSITY_WIND_KT",
                    pd.NA
                )

                result.loc[
                    result["_ROW_ID"] == row_id,
                    "INTENSITY_PRES_MB"
                ] = row.get(
                    "INTENSITY_PRES_MB",
                    pd.NA
                )

                matched_rows += 1

            else:
                timestamp_misses += 1

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    track_matched_mask = (
        result["LAT"].notna()
        & result["LON"].notna()
    )

    actual_matched = track_matched_mask.sum()

    unmatched_rows = len(result) - actual_matched

    print("=" * 60)
    print("TRACK JOIN SUMMARY")
    print("=" * 60)

    print(
        f"Original INCYDE rows:       {len(manifest)}"
    )

    print(
        f"Track-matched rows:         {actual_matched}"
    )

    print(
        f"Rows without track match:   {unmatched_rows}"
    )

    percentage = (
        actual_matched / len(manifest) * 100
        if len(manifest) > 0
        else 0
    )

    print(
        f"Match percentage:           {percentage:.2f}%"
    )

    print(
        f"Timestamp misses:           {timestamp_misses}"
    )

    print(
        f"Unmatched storm names:      {len(unmatched_groups)}"
    )

    if unmatched_groups:

        print()
        print("Unmatched storm names:")

        for name, count in sorted(
            unmatched_groups.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            print(
                f"  {name} - {count} rows"
            )

    # ---------------------------------------------------------
    # REMOVE INTERNAL COLUMNS
    # ---------------------------------------------------------

    result = result.drop(
        columns=[
            "_ROW_ID",
            "_STORM_KEY"
        ],
        errors="ignore"
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    result.to_csv(
        args.output,
        index=False
    )

    print()
    print(
        f"Saved joined manifest to: {args.output}"
    )

    print(
        f"Final rows in output: {len(result)}"
    )


if __name__ == "__main__":
    main()