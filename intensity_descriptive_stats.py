def intensity_descriptive_stats(df_asu):
    """
    Show how to compute descriptive stats of intensities, grouped
    by event and/or by (h_asu, k_asu, l_asu).
    """
    # 1) Per-event summary:
    per_event_stats = df_asu.groupby("event")["I"].describe()
    print("\n[Per-event intensity stats]\n", per_event_stats)

    # 2) Per-event + per-ASU summary (be careful if large!):
    per_asu_stats = df_asu.groupby(["event", "h_asu", "k_asu", "l_asu"])["I"].describe()
    print("\n[Per-event + per-ASU intensity stats]\n", per_asu_stats.head(20),
          "\n... (showing first 20 groups)")
