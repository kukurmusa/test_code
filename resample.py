def resample_min_per_group(df, group_cols, min_samples=200, random_state=42):
    grouped = df.groupby(group_cols)
    resampled_groups = []

    for group_key, group_df in grouped:
        if len(group_df) < min_samples:
            continue  # skip small groups
        sample_df = group_df.sample(n=min_samples, random_state=random_state)
        resampled_groups.append(sample_df)

    # Combine all large groups into one DataFrame
    resampled_df = pd.concat(resampled_groups).reset_index(drop=True)
    return resampled_df
