def evaluate_filter(dataset_df, filtered_df, id_column='id'):
    true_df = dataset_df

    # Counts
    P  = filtered_df[id_column].nunique()
    A  = true_df[id_column].nunique()
    TP_ids = filtered_df[filtered_df[id_column].isin(true_df[id_column])][id_column].unique()
    TP = len(TP_ids)

    FP_ids = set(filtered_df[id_column]) - set(TP_ids)
    FN_ids = set(true_df[id_column]) - set(TP_ids)

    FP_df = filtered_df[filtered_df[id_column].isin(FP_ids)]
    FN_df = true_df[true_df[id_column].isin(FN_ids)]

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
