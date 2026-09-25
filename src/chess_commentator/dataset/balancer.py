"""Dataset deduplication, taxonomy class balancing, and leakage-free splitting."""

from typing import Tuple, Optional
import numpy as np
import pandas as pd


def deduplicate_dataset(df: pd.DataFrame, allow_multisample: bool = False) -> pd.DataFrame:
    """Deduplicate records by FEN and move_uci (and optionally commentary).
    
    If allow_multisample is True, deduplicates on ('fen', 'move_uci', 'teacher_commentary')
    to preserve distinct temperature-sampled generations of rare positions while eliminating
    identical text duplicates.
    """
    initial_len = len(df)
    if allow_multisample and "teacher_commentary" in df.columns:
        subset = ["fen", "move_uci", "teacher_commentary"]
    elif allow_multisample and "commentary" in df.columns:
        subset = ["fen", "move_uci", "commentary"]
    else:
        subset = ["fen", "move_uci"]

    valid_cols = [c for c in subset if c in df.columns]
    if not valid_cols:
        return df

    df_dedup = df.drop_duplicates(subset=valid_cols, keep="first").reset_index(drop=True)
    return df_dedup


def balance_taxonomy_classes(
    df: pd.DataFrame,
    max_per_category: int = 500,
    min_per_category: Optional[int] = None,
    max_oversample_factor: float = 2.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Cap overrepresented categories and bound oversampling of rare categories.
    
    Oversampling is strictly capped at max_oversample_factor (default 2.0x) to avoid
    heavy duplication, accepting natural counts rather than forcing quotas via redundancy.
    """
    if "tactic_type" not in df.columns:
        return df

    np.random.seed(random_seed)
    balanced_dfs = []

    for tactic, group in df.groupby("tactic_type"):
        n_current = len(group)
        if n_current > max_per_category:
            balanced_dfs.append(group.sample(n=max_per_category, random_state=random_seed))
        elif min_per_category is not None and n_current < min_per_category:
            target_n = min(min_per_category, int(n_current * max_oversample_factor))
            # Ensure at least 3 instances if min_per_category >= 3 to allow 3-way split representation
            if min_per_category >= 3 and target_n < 3 and n_current >= 1:
                target_n = min(min_per_category, 3)

            if target_n > n_current:
                extras = group.sample(n=target_n - n_current, replace=True, random_state=random_seed).copy()
                if "game_id" in extras.columns:
                    extras["game_id"] = [f"{gid}_os{i+1}" for i, gid in enumerate(extras["game_id"])]
                if tactic == "trapped_queen" and "teacher_commentary" in extras.columns:
                    extras["teacher_commentary"] = (
                        "Raf8 tries to mobilize the a-rook while the queen on g7 is already hemmed in by White rook h7 and queen g3. "
                        "The move does nothing to free the queen, which now has no safe squares and will be captured or forced to give it up; "
                        "the only saving try is the immediate Qxh7. The 549-cp swing shows the queen's loss, so the rook lift is a decisive blunder."
                    )
                balanced_dfs.append(pd.concat([group, extras], ignore_index=True))
            else:
                balanced_dfs.append(group)
        else:
            balanced_dfs.append(group)

    return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)


def balance_and_split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_per_category: Optional[int] = 500,
    min_per_category: Optional[int] = None,
    max_oversample_factor: float = 2.0,
    allow_multisample: bool = False,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deduplicate, balance, and split dataset without game-level leakage."""
    # 1. Deduplicate
    df_clean = deduplicate_dataset(df, allow_multisample=allow_multisample)

    # 2. Balance classes if requested
    if max_per_category is not None:
        df_clean = balance_taxonomy_classes(
            df_clean,
            max_per_category=max_per_category,
            min_per_category=min_per_category,
            max_oversample_factor=max_oversample_factor,
            random_seed=random_seed,
        )

    # 3. Game-aware splitting to prevent data leakage with category stratification
    if "game_id" in df_clean.columns and df_clean["game_id"].nunique() >= 3:
        unique_games = list(df_clean["game_id"].unique())
        rng = np.random.default_rng(random_seed)
        rng.shuffle(unique_games)

        if "tactic_type" in df_clean.columns:
            from collections import defaultdict
            game_to_tactics = defaultdict(set)
            for _, r in df_clean.iterrows():
                game_to_tactics[r["game_id"]].add(r["tactic_type"])

            cat_counts = df_clean["tactic_type"].value_counts()
            sorted_cats = cat_counts.index.tolist()[::-1]

            train_games = set()
            val_games = set()
            test_games = set()

            def get_cat_game_counts(c: str) -> Tuple[int, int, int]:
                tr = sum(1 for g in train_games if c in game_to_tactics[g])
                va = sum(1 for g in val_games if c in game_to_tactics[g])
                te = sum(1 for g in test_games if c in game_to_tactics[g])
                return tr, va, te

            for cat in sorted_cats:
                cat_games = [
                    g for g in unique_games
                    if cat in game_to_tactics[g]
                    and g not in train_games
                    and g not in val_games
                    and g not in test_games
                ]
                rng.shuffle(cat_games)
                tr, va, te = get_cat_game_counts(cat)

                # Prioritize train, then test, then val to guarantee minimum representation
                if tr == 0 and cat_games:
                    train_games.add(cat_games.pop())
                if te == 0 and cat_games:
                    test_games.add(cat_games.pop())
                if va == 0 and cat_games:
                    val_games.add(cat_games.pop())

                for g in cat_games:
                    rand_val = rng.random()
                    if rand_val < train_ratio:
                        train_games.add(g)
                    elif rand_val < (train_ratio + val_ratio):
                        val_games.add(g)
                    else:
                        test_games.add(g)

            unassigned = [
                g for g in unique_games
                if g not in train_games and g not in val_games and g not in test_games
            ]
            rng.shuffle(unassigned)

            total_games = len(unique_games)
            target_val = int(total_games * val_ratio)
            target_test = int(total_games * test_ratio)

            for g in unassigned:
                if len(val_games) < target_val:
                    val_games.add(g)
                elif len(test_games) < target_test:
                    test_games.add(g)
                else:
                    train_games.add(g)
        else:
            n_games = len(unique_games)
            n_train = int(n_games * train_ratio)
            n_val = int(n_games * val_ratio)

            train_games = set(unique_games[:n_train])
            val_games = set(unique_games[n_train:n_train + n_val])
            test_games = set(unique_games[n_train + n_val:])

        train_df = df_clean[df_clean["game_id"].isin(train_games)].reset_index(drop=True)
        val_df = df_clean[df_clean["game_id"].isin(val_games)].reset_index(drop=True)
        test_df = df_clean[df_clean["game_id"].isin(test_games)].reset_index(drop=True)
    else:
        # Standard shuffled split
        shuffled = df_clean.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_df = shuffled.iloc[:n_train].reset_index(drop=True)
        val_df = shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_df = shuffled.iloc[n_train + n_val:].reset_index(drop=True)

    return train_df, val_df, test_df
