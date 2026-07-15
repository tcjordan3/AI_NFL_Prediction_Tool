"""
Application for viewing historical and predicted NFL player stats
"""

import streamlit as st
import src.configurations as cfg
import pandas as pd
from pathlib import Path

# Page config
st.set_page_config(
    page_title="NFL Player Stats & Predictions",
    page_icon="🏈",
    layout="wide"
)

COLUMN_CONFIG_PASSING = {
        "Yds": st.column_config.NumberColumn("Yds", help="Passing yards"),
        "Att": st.column_config.NumberColumn("Att", help="Passing attempts"),
        "Cmp": st.column_config.NumberColumn("Cmp", help="Completions"),
        "Cmp%": st.column_config.NumberColumn("Cmp%", help="Completion percentage"),
        "Td": st.column_config.NumberColumn("Td", help="Passing touchdowns"),
        "Td%": st.column_config.NumberColumn("Td%", help="Touchdown percentage"),
        "Int": st.column_config.NumberColumn("Int", help="Interceptions"),
        "Int%": st.column_config.NumberColumn("Int%", help="Interception percentage"),
        "Y/A": st.column_config.NumberColumn("Y/A", help="Yards per attempt"),
        "Ay/A": st.column_config.NumberColumn("Ay/A", help="Adjusted yards per attempt (accounts for TDs and INTs)"),
        "Y/C": st.column_config.NumberColumn("Y/C", help="Yards per completion"),
        "Y/G": st.column_config.NumberColumn("Y/G", help="Yards per game"),
        "G": st.column_config.NumberColumn("G", help="Games played"),
    }

# TODO: apply rushing/receiving configs


# ─── Data Loading ───────────────────────────────────────────

@st.cache_resource
def load_data():
    """
    Load player data from CSV file and clean DataFrames

    Returns:
        Tuple of Dataframes containing player stats
    """

    # Load passing data
    df_passing = pd.read_csv(Path(cfg.PREDICTIONS_DIR) / "qb_predictions_2026.csv")

    # TODO: Load rushing data
    df_rushing = pd.DataFrame()

    # TODO: Load receiving data
    df_receiving = pd.DataFrame()

    # Clean DataFrames
    for df in [df_passing, df_rushing, df_receiving]:
        if df.empty:
            continue

        # Drop pfr_id column and id columns
        df.drop(columns=[c for c in df.columns if "pfr_id" in c], inplace=True, errors="ignore")

        # Rename columns
        df.rename(lambda x: x.replace("_pct", "%").replace("_per_", "/").replace("_", " ").title(), axis=1, inplace=True)
        
    return df_passing, df_rushing, df_receiving


def apply_filters(df_passing, df_rushing, df_receiving, pos: str, season: int, players: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply filters to the player data

    Args:
        df_passing: DataFrame containing passing stats
        df_rushing: DataFrame containing rushing stats
        df_receiving: DataFrame containing receiving stats
        pos: Selected position
        season: Selected season
        players: List of selected players

    Returns:
        Tuple of filtered DataFrames
    """

    # Apply position filter
    if pos != "All":
        if pos == "QB" and not df_passing.empty:
            # TODO: For QB, we will want to include rushing once available
            df_rushing = pd.DataFrame()  # Clear rushing data for QB position
            df_receiving = pd.DataFrame()  # Clear receiving data for QB position
        elif pos == "RB" or pos == "WR":
            df_passing = pd.DataFrame()  # Clear passing data for RB and WR positions
            if not df_rushing.empty:
                df_rushing = df_rushing[df_rushing["pos"] == pos]
            if not df_receiving.empty:
                df_receiving = df_receiving[df_receiving["pos"] == pos]
        else: # TE
            df_passing = pd.DataFrame()  # Clear passing data for TE position
            df_rushing = pd.DataFrame()  # Clear rushing data for TE position
            if not df_receiving.empty:
                df_receiving = df_receiving[df_receiving["pos"] == pos]

    # Apply season filter
    if season != "All":
        if not df_passing.empty:
            df_passing = df_passing[df_passing["Season"] == season]
        if not df_rushing.empty:
            df_rushing = df_rushing[df_rushing["Season"] == season]
        if not df_receiving.empty:
            df_receiving = df_receiving[df_receiving["Season"] == season]

    # Apply player filter
    if players:
        if not df_passing.empty:
            df_passing = df_passing[df_passing["Player"].isin(players)]
        if not df_rushing.empty:
            df_rushing = df_rushing[df_rushing["Player"].isin(players)]
        if not df_receiving.empty:
            df_receiving = df_receiving[df_receiving["Player"].isin(players)]

    return df_passing, df_rushing, df_receiving


# ─── Page Entrypoint ───────────────────────────────────────────

def main():
    # Header
    st.title("🏈 NFL Player Stats & Predictions")
    st.markdown("""
    Explore historical NFL data and predictions for the upcoming season!
    """)

    # Load data
    df_passing, df_rushing, df_receiving = load_data()

    # Filters
    st.subheader("Filters:")
    c_pos, c_season, c_player = st.columns(3)

    with c_pos:
        pos = st.selectbox(
            "Position",
            options=["All", "QB", "RB", "WR", "TE"],
            index=0,
            help="Select the position to filter by"
        )

    with c_season:
        season = st.selectbox(
            "Season",
            options=["All"] + sorted(df_passing["Season"].unique(), reverse=True),
            index=0,
            help="Select the season to filter by"
        )

    with c_player:
        if pos == "All":
            player_sets = [df["Player"].tolist() for df in [df_passing, df_rushing, df_receiving] if not df.empty]
        elif pos == "QB":
            player_sets = [df_passing["Player"].tolist()] if not df_passing.empty else []
        elif pos in ("RB", "WR"):
            player_sets = [df["Player"].tolist() for df in [df_rushing, df_receiving] if not df.empty]
        else:  # TE
            player_sets = [df_receiving["Player"].tolist()] if not df_receiving.empty else []

        all_players = sorted(set(p for players in player_sets for p in players))

        players = st.multiselect(
            "Player Name",
            options=all_players,
            default=[],
            help="Search for a player by name"
        )

    # Apply filters
    df_passing, df_rushing, df_receiving = apply_filters(df_passing, df_rushing, df_receiving, pos, season, players)

    # Display data
    st.subheader("Player Stats:")

    if pos != "All":
        if pos == "QB":
            if not df_passing.empty:
                st.markdown("#### Passing Stats")
                st.dataframe(df_passing, hide_index=True, column_config=COLUMN_CONFIG_PASSING)
            if not df_rushing.empty:
                st.markdown("#### Rushing Stats")
                st.dataframe(df_rushing, hide_index=True)
        elif pos == "RB" or pos == "WR":
            if not df_rushing.empty:
                st.markdown("#### Rushing Stats")
                st.dataframe(df_rushing, hide_index=True)
            if not df_receiving.empty:
                st.markdown("#### Receiving Stats")
                st.dataframe(df_receiving, hide_index=True)
        elif not df_receiving.empty:
            st.markdown("#### Receiving Stats")
            st.dataframe(df_receiving, hide_index=True)
    else:
        if not df_passing.empty:
            st.markdown("#### Passing Stats")
            st.dataframe(df_passing, hide_index=True, column_config=COLUMN_CONFIG_PASSING)
        if not df_rushing.empty:
            st.markdown("#### Rushing Stats")
            st.dataframe(df_rushing, hide_index=True)
        if not df_receiving.empty:
            st.markdown("#### Receiving Stats")
            st.dataframe(df_receiving, hide_index=True)


if __name__ == "__main__":
    main()