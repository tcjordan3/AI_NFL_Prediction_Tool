import sys
from pathlib import Path
import pandas as pd
import logging

sys.path.append(str(Path(__file__).resolve().parent.parent))

import configurations as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ─── General Helper Functions ───────────────────────────────────────────

def normalize_name(name: str) -> str:
    """
    Normalize QB names for cross-source merging.
    Converts 'Joe Flacco' and 'J.Flacco' both to 'j.flacco' format.
    Handles suffixes like Jr., Sr., II, III to avoid incorrect initials.
    Handles nflfastR multi-char abbreviations like 'Ty.Taylor', 'Aa.Rodgers'.
    """

    if pd.isna(name):
        return name

    name = name.strip().lower()

    # dot-separated format like "Ty.Taylor"
    if "." in name and " " not in name:
        first, last = name.split(".", 1)
        return f"{first[0]}.{last}"

    # normal space-separated names
    parts = name.split()

    # Strip suffixes
    while parts and parts[-1].strip(".") in cfg.SUFFIXES:
        parts.pop()

    if len(parts) >= 2:
        first = parts[0].strip(".")
        last = parts[-1].strip(".")
        return f"{first[0]}.{last}"

    return name


def load_pfr_df(path_template: str, years: range = range(2018, 2026)) -> pd.DataFrame:
    """
    Load and concatenate raw PFR CSVs across all seasons.

    Args:
        path_template: Path string with {year} placeholder
                       e.g. "data/raw/qb/passing/passing_{year}.csv"

    Returns:
        Raw, uncleaned DataFrame.
    """
    frames = []

    for year in years:
        path = path_template.format(year=year)
        logger.info(f"Loading {path}...")
        df_year = pd.read_csv(path)
        df_year = df_year[df_year["Player"] != "Player"]
        df_year["season"] = year
        frames.append(df_year)

    return pd.concat(frames, ignore_index=True)

class QBProcessor:
    """
    A class to encapsulate the full QB data processing pipeline.
    Contains methods for loading, cleaning, aggregating, merging,
    and validating QB datasets from PFR and nflfastR.
    """

    def __init__(self, years: range = range(2018, 2026)):
        self.years = years

    # ─── Loaders ────────────────────────────────────────────────────────────


    def load_nflfastr(self) -> pd.DataFrame:
        """
        Load and concatenate raw nflfastR parquet files across all seasons.
        Filters to QB dropbacks only.

        Args:
            years: Range of seasons to load (default 2018-2025)

        Returns:
            df_nflfastr: a raw, play-by-play DataFrame.
        """

        df_nflfastr = pd.DataFrame()
        frames = []

        for year in self.years:
            path = cfg.PFR_QB_NFLFASTR.format(year=year)
            logger.info(f"Loading {path}...")
            df_year = pd.read_parquet(path)

            df_year = df_year[df_year["qb_dropback"] == 1]

            frames.append(df_year)

        df_nflfastr = pd.concat(frames, ignore_index=True)

        return df_nflfastr


    # ─── Cleaners ───────────────────────────────────────────────────────────────

    def clean_passing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean PFR passing data.

        Args:
            df: Raw DataFrame loaded from PFR passing CSVs.

        Returns:
            df_clean_passing: a cleaned DataFrame at season-player granularity.
        """

        df_clean_passing = df.copy()

        # Update header names
        df_clean_passing.columns = (
            df_clean_passing.columns
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
            .str.replace("%", "_pct")
            .str.replace("/", "_per_")
            .str.lower()
        )

        # Drop rows in which player is not a QB
        df_clean_passing = df_clean_passing[df_clean_passing["pos"] == "QB"]

        # Rename player_additional
        df_clean_passing = df_clean_passing.rename(columns={"player_additional": "pfr_id"})

        # NOTE: We will not be able to correlate traded players with their teams.
        # We may want to handle this differently in the future
        # For players with a 2TM row, keep only 2TM and drop individual team rows
        players_with_2tm = df_clean_passing[
            df_clean_passing["team"] == "2TM"
        ][["pfr_id", "season"]].apply(tuple, axis=1)

        df_clean_passing = df_clean_passing[
            ~(
                df_clean_passing[["pfr_id", "season"]].apply(tuple, axis=1).isin(players_with_2tm) &
                (df_clean_passing["team"] != "2TM")
            )
        ]

        # Split record into wins, losses, and ties
        qbrec_split = df_clean_passing["qbrec"].str.split("-", expand=True)
        df_clean_passing["wins"]   = pd.to_numeric(qbrec_split[0], errors="coerce")
        df_clean_passing["losses"] = pd.to_numeric(qbrec_split[1], errors="coerce")
        df_clean_passing["ties"]   = pd.to_numeric(qbrec_split[2], errors="coerce")
        df_clean_passing = df_clean_passing.drop(columns=["qbrec"])

        # Type casting
        exclude = ["player", "pfr_id", "team", "pos"]
        numeric_cols = [c for c in df_clean_passing.columns if c not in exclude]
        df_clean_passing[numeric_cols] = df_clean_passing[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # Fill NaN record values with 0
        df_clean_passing[["wins", "losses", "ties"]] = df_clean_passing[["wins", "losses", "ties"]].fillna(0)

        # Drop rows in which QB has fewer than MIN_DROPBACKS passing attempts in the season
        df_clean_passing = df_clean_passing[df_clean_passing["att"] >= cfg.MIN_DROPBACKS]

        # Drop unused columns
        cols_to_drop = ["rk", "awards"]
        df_clean_passing = df_clean_passing.drop(columns=cols_to_drop, errors="ignore")

        # After column normalization, rename the ambiguous duplicate
        df_clean_passing = df_clean_passing.rename(columns={"yds.1": "sack_yds_lost"})

        return df_clean_passing.reset_index(drop=True)


    def clean_advanced_passing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean PFR advanced passing data.

        Args:
            df: Raw DataFrame loaded from PFR advanced passing CSVs.

        Returns:
            df_clean_advanced: a cleaned DataFrame at season-player granularity.
        """

        df_clean_advanced = df.copy()

        # Update malformed column name for pfr_id
        df_clean_advanced = df_clean_advanced.rename(columns={"-9999": "pfr_id"})

        # Update header names
        df_clean_advanced.columns = (
            df_clean_advanced.columns
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
            .str.replace("%", "_pct")
            .str.replace("/", "_per_")
            .str.lower()
        )

        # Drop rows in which player is not a QB
        df_clean_advanced = df_clean_advanced[df_clean_advanced["pos"] == "QB"]

        # For players with a 2TM row, keep only 2TM and drop individual team rows
        players_with_2tm = df_clean_advanced[
            df_clean_advanced["team"] == "2TM"
        ][["pfr_id", "season"]].apply(tuple, axis=1)

        df_clean_advanced = df_clean_advanced[
            ~(
                df_clean_advanced[["pfr_id", "season"]].apply(tuple, axis=1).isin(players_with_2tm) &
                (df_clean_advanced["team"] != "2TM")
            )
        ]

        # Type casting
        exclude = ["player", "pfr_id", "team", "pos"]
        numeric_cols = [c for c in df_clean_advanced.columns if c not in exclude]
        df_clean_advanced[numeric_cols] = df_clean_advanced[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # Fill yds/scr NaN values with 0
        df_clean_advanced["yds_per_scr"] = df_clean_advanced["yds_per_scr"].fillna(0)

        # Drop rows in which QB has fewer than MIN_DROPBACKS passing attempts in the season
        df_clean_advanced = df_clean_advanced[df_clean_advanced["att"] >= cfg.MIN_DROPBACKS]

        # Drop unused columns
        cols_to_drop = ["rk", "awards"]
        df_clean_advanced = df_clean_advanced.drop(columns=cols_to_drop, errors="ignore")

        return df_clean_advanced.reset_index(drop=True)
    

    def clean_rushing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean PFR rushing data.

        Args:
            df: Raw DataFrame loaded from PFR rushing CSVs.

        Returns:
            df_clean_rushing: a cleaned DataFrame at season-player granularity.
        """

        df_clean_rushing = df.copy()

        # Update malformed column name for pfr_id
        df_clean_rushing = df_clean_rushing.rename(columns={"-9999": "pfr_id"})

        # Update header names
        df_clean_rushing.columns = (
            df_clean_rushing.columns
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("-", "_")
            .str.replace("%", "_pct")
            .str.replace("/", "_per_")
            .str.lower()
        )

        # Drop rows in which player is not a QB
        df_clean_rushing = df_clean_rushing[df_clean_rushing["pos"] == "QB"]

        # For players with a 2TM row, keep only 2TM and drop individual team rows
        players_with_2tm = df_clean_rushing[
            df_clean_rushing["team"] == "2TM"
        ][["pfr_id", "season"]].apply(tuple, axis=1)

        df_clean_rushing = df_clean_rushing[
            ~(
                df_clean_rushing[["pfr_id", "season"]].apply(tuple, axis=1).isin(players_with_2tm) &
                (df_clean_rushing["team"] != "2TM")
            )
        ]

        # Type casting
        exclude = ["player", "pfr_id", "team", "pos"]
        numeric_cols = [c for c in df_clean_rushing.columns if c not in exclude]
        df_clean_rushing[numeric_cols] = df_clean_rushing[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # Drop rows in which QB has fewer than MIN_RUSH_ATT_QB rushing attempts in the season
        df_clean_rushing = df_clean_rushing[df_clean_rushing["att"] >= cfg.MIN_RUSH_ATT_QB]

        # Drop unused columns
        cols_to_drop = ["rk", "awards"]
        df_clean_rushing = df_clean_rushing.drop(columns=cols_to_drop, errors="ignore")

        # Add rushing_ prefix to columns to avoid duplicates
        cols_to_prefix = ["att", "yds", "td", "1d", "succ_pct", "lng", "y_per_a", "y_per_g", "a_per_g"]
        df_clean_rushing = df_clean_rushing.rename(columns={c: f"rushing_{c}" for c in cols_to_prefix if c in df_clean_rushing.columns})

        return df_clean_rushing.reset_index(drop=True)


    def clean_nflfastr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean nflfastR play-by-play data.

        Args:
            df: Raw play-by-play DataFrame loaded from nflfastR parquet files.

        Returns:
            df_clean_nflfastr: a cleaned play-by-play DataFrame.
        """

        df_clean_nflfastr = df.copy()

        cols_to_keep = [
            "season", "passer_player_name", "passer_player_id",
            "qb_epa", "cpoe", "air_yards", "was_pressure", "time_to_throw"
        ]

        # Drop unused columns
        df_clean_nflfastr = df_clean_nflfastr[cols_to_keep]

        # Handle Null/None values
        df_clean_nflfastr = df_clean_nflfastr[df_clean_nflfastr["passer_player_id"].notna()]
        df_clean_nflfastr = df_clean_nflfastr[df_clean_nflfastr["passer_player_name"].notna()]
        df_clean_nflfastr["was_pressure"] = df_clean_nflfastr["was_pressure"].astype(float)

        # Type casting
        exclude = ["passer_player_name", "passer_player_id", "was_pressure"]
        numeric_cols = [c for c in df_clean_nflfastr.columns if c not in exclude]
        df_clean_nflfastr[numeric_cols] = df_clean_nflfastr[numeric_cols].apply(pd.to_numeric, errors="coerce")

        return df_clean_nflfastr.reset_index(drop=True)


    # ─── Aggregators ────────────────────────────────────────────────────────────

    def aggregate_nflfastr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse play-by-play nflfastR data to season-player granularity.
        Computes: mean qb_epa, mean cpoe, mean air_yards, pressure_rate,
        mean time_to_throw, and n_dropbacks as a sample size column.

        Args:
            df: Cleaned play-by-play DataFrame from nflfastR.

        Returns:
            df_aggregated: A DataFrame with one row per player per season.
        """

        df_aggregated = df.groupby(["season", "passer_player_id"]).agg(
            passer_player_name = ("passer_player_name", "first"),
            qb_epa_mean = ("qb_epa", "mean"),
            cpoe_mean = ("cpoe", "mean"),
            air_yards_mean = ("air_yards", "mean"),
            pressure_rate = ("was_pressure", "mean"),
            time_to_throw_mean = ("time_to_throw", "mean"),
            n_dropbacks = ("qb_epa", "size")
        ).reset_index()

        # Apply minimum dropback threshold
        df_aggregated = df_aggregated[df_aggregated["n_dropbacks"] >= cfg.MIN_DROPBACKS]

        return df_aggregated.reset_index(drop=True)


    # ─── Merge ──────────────────────────────────────────────────────────────────

    def merge_qb_data(
        self,
        passing: pd.DataFrame,
        advanced: pd.DataFrame,
        rushing: pd.DataFrame,
        nflfastr: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge cleaned PFR passing, PFR advanced passing, and aggregated
        nflfastR data on (player_name, season).
        Left join on PFR passing as the base — nflfastR and advanced stats
        are supplementary and may not cover every player.

        Args:
            passing: Cleaned PFR passing DataFrame.
            advanced: Cleaned PFR advanced passing DataFrame.
            rushing: Cleaned PFR rushing DataFrame.
            nflfastr: Aggregated nflfastR DataFrame.

        Returns:
            df_merged: A merged DataFrame at season-player granularity.
        """

        # Merge PFR passing with PFR advanced passing first
        df_merged = passing.merge(
            advanced.drop(columns=["player", "team", "pos"], errors="ignore"),
            on=["season", "pfr_id"],
            how="left",
            suffixes=("", "_advanced")
        )

        # Merge PFR rushing data next
        df_merged = df_merged.merge(
            rushing.drop(columns=["player", "team", "pos"], errors="ignore"),
            on=["season", "pfr_id"],
            how="left",
            suffixes=("", "_rushing")
        )

        # Add normalized name key for nflfastR merge
        df_merged["name_key"] = df_merged["player"].apply(normalize_name)
        nflfastr["name_key"] = nflfastr["passer_player_name"].apply(normalize_name)

        # Merge the result with aggregated nflfastR data
        df_merged = df_merged.merge(
            nflfastr.drop(columns=["passer_player_name"], errors="ignore"),
            on=["season", "name_key"],
            how="left",
            suffixes=("", "_nflfastr")
        )

        # Drop the name_key column used for merging
        df_merged = df_merged.drop(columns=["name_key", "player_passer_id"], errors="ignore")

        # Drop duplicated per-table columns
        cols_to_drop_after_merge = [
            "age_advanced", "g_advanced", "gs_advanced",
            "cmp_advanced", "att_advanced",
            "ontgt", "ontgt_pct", "plays",
            "yds_advanced", "passatt", "passyds", "rushatt",
            "rushyds", "passatt.1", "passyds.1",
            "age_rushing", "g_rushing", "gs_rushing"
        ]

        # For QBs with insufficient rushing attempts, set rushing stats to 0
        rushing_cols = [c for c in df_merged.columns if c.startswith("rushing_") or c == "fmb"]

        for col in rushing_cols:
            df_merged[col] = df_merged.apply(
                lambda row: 0 if pd.isna(row[col]) else row[col],
                axis=1
            )

        df_merged = df_merged.drop(columns=cols_to_drop_after_merge, errors="ignore")

        return df_merged.reset_index(drop=True)


    # ─── Validation ─────────────────────────────────────────────────────────────

    def validate_qb_dataset(self, df: pd.DataFrame) -> None:
        """
        Run sanity checks on the final merged dataset. Checks:
        - Expected row count (approx 30-40 QBs per season)
        - No duplicate player-season rows
        - Null rates per column
        - Stat ranges (e.g. completion % between 0-100, epa within reason)
        Prints a summary report. Raises ValueError on critical failures.
        """

        # Row count check
        rows = len(df)    
        print(f"Total rows in merged dataset: {rows}")

        # Per-season row count check
        season_counts = df["season"].value_counts().sort_index()
        print("\nRow counts per season:")
        for season, count in season_counts.items():
            print(f"  {season}: {count} rows")
            if count < 30 or count > 70:
                logger.warning(f"  ⚠️ Unusual row count for season {season}: {count} rows")

        # Check for duplicate player-season rows
        duplicates = df.duplicated(subset=["season", "player"], keep=False)
        if duplicates.any():
            dup_rows = df[duplicates][["season", "player"]]
            logger.error(f"Duplicate player-season rows found:\n{dup_rows}")
            raise ValueError("Duplicate player-season rows detected in merged dataset.")
        
        # Null rate check
        print("\nNull rate check:")
        null_values = False

        for col in df.columns:
            null_pct = df[col].isna().mean() * 100
            if null_pct > 0:
                null_values = True
                print(f"  {col}: {null_pct:.1f}% null")

        if not null_values:
            print("  No null values found.")

        # Stat range checks
        # All percentage columns should be between 0 and 100
        pct_cols = [c for c in df.columns if "pct" in c]
        for col in pct_cols:
            if col in df.columns:
                out_of_range = df[(df[col] < 0) | (df[col] > 100)]
                if not out_of_range.empty:
                    logger.warning(f"Values out of range in {col}:\n{out_of_range[['season', 'player', col]]}")

        # EPA should be within a reasonable range (e.g. -3 to +3)     
        if "qb_epa_mean" in df.columns:
            out_of_range = df[(df["qb_epa_mean"] < -3) | (df["qb_epa_mean"] > 3)]
            if not out_of_range.empty:
                logger.warning(f"Suspicious EPA values:\n{out_of_range[['season', 'player', 'qb_epa_mean']]}")

        if "qb_epa_mean" in df.columns:
            unmatched = df["qb_epa_mean"].isna().sum()
            unmatched_pct = unmatched / len(df) * 100
            print(f"\nnflfastR merge: {len(df) - unmatched}/{len(df)} rows matched ({100 - unmatched_pct:.1f}%)")
            if unmatched_pct > 10:
                logger.warning(f"⚠️ {unmatched_pct:.1f}% of rows failed to match nflfastR data")


    # ─── Entry Point ────────────────────────────────────────────────────────────

    def build_qb_dataset(self) -> pd.DataFrame:
        """
        Orchestrates the full pipeline:
        load → clean → aggregate → merge → validate → save.
        Saves final dataset to data/processed/qb_stats_2018_2025.csv.

        Args:
            years: Range of seasons to include in the dataset (default 2018-2025)

        Returns:
            df_merged: the final DataFrame.
        """

        # Load raw data
        logger.info("Loading raw data...")
        df_passing = load_pfr_df(years=self.years, path_template=cfg.PFR_QB_PASSING)
        df_advanced = load_pfr_df(years=self.years, path_template=cfg.PFR_QB_ADVANCED_PASSING)
        df_rushing = load_pfr_df(years=self.years, path_template=cfg.PFR_QB_RUSHING)
        df_nflfastr = self.load_nflfastr()

        # Clean data
        logger.info("Cleaning data...")
        df_clean_passing = self.clean_passing(df_passing)
        df_clean_advanced = self.clean_advanced_passing(df_advanced)
        df_clean_rushing = self.clean_rushing(df_rushing)
        df_clean_nflfastr = self.clean_nflfastr(df_nflfastr)

        # Aggregate nflfastR data
        logger.info("Aggregating nflfastR data...")
        df_aggregated_nflfastr = self.aggregate_nflfastr(df_clean_nflfastr)

        # Merge datasets
        logger.info("Merging datasets...")
        df_merged = self.merge_qb_data(passing=df_clean_passing, advanced=df_clean_advanced, nflfastr=df_aggregated_nflfastr, rushing=df_clean_rushing)

        # Validate final dataset
        logger.info("Validating merged dataset...")
        try:
            self.validate_qb_dataset(df_merged)
        except ValueError as e:
            logger.error(f"Validation failed: {e}")
            raise

        # Save final dataset
        logger.info("Saving final dataset...")
        output_path = cfg.QB_OUTPUT_DATA_FILE.format(start=self.years.start, end=self.years.stop - 1)
        df_merged.to_csv(output_path, index=False)
        logger.info(f"Final dataset saved to {output_path}")

        return df_merged


if __name__ == "__main__":
    processor_qb = QBProcessor(years=cfg.YEARS)
    df_qb = processor_qb.build_qb_dataset()
    logger.info(df_qb.head())