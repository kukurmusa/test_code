def stratified_resample(
    df,
    group_cols,
    min_total_samples=200,
    random_state=42
):
    # Calculate original group sizes
    group_counts = df.groupby(group_cols).size()
    total_count = group_counts.sum()
    
    # Proportional target size per group
    proportions = group_counts / total_count
    target_sizes = (proportions * min_total_samples).round().astype(int)

    # Make sure no group requests more rows than it has
    target_sizes = target_sizes.clip(upper=group_counts)

    # Skip resampling if a group has fewer rows than its target (use full group instead)
    def sample_or_all(group):
        group_key = tuple(group.name) if isinstance(group.name, tuple) else (group.name,)
        n = target_sizes.get(group_key, 0)
        if len(group) <= n:
            return group
        return group.sample(n=n, random_state=random_state)

    resampled = df.groupby(group_cols, group_keys=False).apply(sample_or_all).reset_index(drop=True)
    
    return resampled
