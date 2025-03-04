df_filtered = df[~((df['time'] != df['time'].shift()) & (df['size'] == df['size'].shift()))]
