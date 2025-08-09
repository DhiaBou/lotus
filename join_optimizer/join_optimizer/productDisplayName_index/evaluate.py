def evaluate_filter(merged_df, filtered_df, article_type, base_colour):
    # Ground‑truth positives
    true_df = merged_df[
        (merged_df['_articleType_x'] == article_type) &
        (merged_df['_baseColour_x']   == base_colour)
        ]

    # Counts
    P  = filtered_df['_id'].nunique()                             # predicted positives
    A  = true_df    ['_id'].nunique()                             # actual positives
    TP = filtered_df[filtered_df['_id'].isin(true_df['_id'])] \
        ['_id'].nunique()                                    # true positives
    FP = P - TP                                                   # false positives
    FN = A - TP                                                   # false negatives

    # Metrics (guarding against zero division)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score  = (2 * precision * recall) / (precision + recall) \
        if (precision + recall) > 0 else 0.0

    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'precision': precision,
        'recall': recall,
        'f1': f1_score
    }
