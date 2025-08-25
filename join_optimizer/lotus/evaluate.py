def evaluate_filter(dataset_df, filtered_df, article_type=None, base_colour=None):
    # Ground-truth positives
    if article_type or base_colour:
        true_df = dataset_df[
            (article_type is None or dataset_df['articleType_x'] == article_type) &
            (base_colour is None or dataset_df['baseColour_x'] == base_colour)
            ]
    else:
        true_df = dataset_df

    # Counts
    P  = filtered_df['id'].nunique()
    A  = true_df['id'].nunique()
    TP_ids = filtered_df[filtered_df['id'].isin(true_df['id'])]['id'].unique()
    TP = len(TP_ids)

    FP_ids = set(filtered_df['id']) - set(TP_ids)
    FN_ids = set(true_df['id']) - set(TP_ids)

    FP_df = filtered_df[filtered_df['id'].isin(FP_ids)]
    FN_df = true_df[true_df['id'].isin(FN_ids)]

    precision = TP / (TP + len(FP_ids)) if (TP + len(FP_ids)) > 0 else 0.0
    recall    = TP / (TP + len(FN_ids)) if (TP + len(FN_ids)) > 0 else 0.0
    f1_score  = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'TP': TP,
        'FP': len(FP_ids),
        'FN': len(FN_ids),
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
    }, FP_df, FN_df
