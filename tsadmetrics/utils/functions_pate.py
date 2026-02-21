import numpy as np

from .functions_auc import auc, binary_clf_curve


def convert_vector_to_events_pate(vector_array):
    positives = np.asarray(vector_array) > 0
    if positives.size == 0:
        return []

    transitions = np.diff(positives.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def generate_buffer_points(max_buffer_size, num_splits, include_zero=True):
    if include_zero:
        start_point = 0
        num_points = num_splits + 1
    else:
        start_point = max_buffer_size / num_splits
        num_points = num_splits

    return np.linspace(start_point, max_buffer_size, num=num_points, dtype=int)


def extract_id(id_string):
    numeric_part = "".join(filter(str.isdigit, id_string))
    return int(numeric_part) if numeric_part else None


def categorize_predicted_ranges_with_ids(
    prediction_ranges, label_anomaly_ranges, e_buffer, d_buffer, time_series_length
):
    categorized_ranges_with_ids = {
        "pre_buffer": [],
        "true_detection": [],
        "post_buffer": [],
        "outside": [],
        "partial_missed": [],
    }

    label_anomaly_ids = {i: (start, end) for i, (start, end) in enumerate(label_anomaly_ranges)}
    covered_points_in_label = {anomaly_id: set() for anomaly_id in label_anomaly_ids}
    post_buffer_end_points = {}

    for pred_start, pred_end in prediction_ranges:
        segment_start = pred_start

        for anomaly_id, (label_start, label_end) in label_anomaly_ids.items():
            next_label_start = (
                label_anomaly_ranges[anomaly_id + 1][0]
                if anomaly_id + 1 < len(label_anomaly_ranges)
                else time_series_length
            )

            post_buffer_end = min(label_end + d_buffer, next_label_start - 1)
            post_buffer_end_points[anomaly_id] = post_buffer_end

            previous_post_buffer_end = post_buffer_end_points.get(anomaly_id - 1, -1)
            pre_buffer_start = max(0, label_start - e_buffer, previous_post_buffer_end + 1)
            pre_buffer_end = label_start - 1

            if segment_start < pre_buffer_start:
                outside_end = min(pred_end, pre_buffer_start - 1)
                if segment_start <= outside_end:
                    categorized_ranges_with_ids["outside"].append(
                        {"range": [segment_start, outside_end], "id": "outside"}
                    )

            if segment_start <= pre_buffer_end:
                pre_buffer_segment_start = max(segment_start, pre_buffer_start)
                pre_buffer_segment_end = min(pred_end, pre_buffer_end)
                if pre_buffer_segment_start <= pre_buffer_segment_end:
                    categorized_ranges_with_ids["pre_buffer"].append(
                        {
                            "range": [pre_buffer_segment_start, pre_buffer_segment_end],
                            "pre_buffer_start": pre_buffer_start,
                            "id": f"{anomaly_id}-pre",
                        }
                    )
                segment_start = pre_buffer_segment_end + 1

            if segment_start <= label_end:
                actual_segment_end = min(pred_end, label_end)
                if segment_start <= actual_segment_end:
                    categorized_ranges_with_ids["true_detection"].append(
                        {
                            "range": [segment_start, actual_segment_end],
                            "id": f"{anomaly_id}-true_detection",
                        }
                    )
                    covered_points_in_label[anomaly_id].update(
                        range(segment_start, actual_segment_end + 1)
                    )
                segment_start = actual_segment_end + 1

            if segment_start <= post_buffer_end:
                buffer_segment_end = min(pred_end, post_buffer_end)
                if segment_start <= buffer_segment_end:
                    categorized_ranges_with_ids["post_buffer"].append(
                        {
                            "range": [segment_start, buffer_segment_end],
                            "post_buffer_end": post_buffer_end,
                            "id": f"{anomaly_id}-buf",
                        }
                    )
                segment_start = buffer_segment_end + 1

        if segment_start <= pred_end:
            categorized_ranges_with_ids["outside"].append(
                {"range": [segment_start, pred_end], "id": "outside"}
            )

    for anomaly_id, (start, end) in label_anomaly_ids.items():
        labeled_points = set(range(start, end + 1))
        if labeled_points != covered_points_in_label[anomaly_id] and covered_points_in_label[anomaly_id]:
            missed_points = labeled_points - covered_points_in_label[anomaly_id]
            for point in missed_points:
                categorized_ranges_with_ids["partial_missed"].append(
                    {"range": [point, point], "id": f"{anomaly_id}-partial_missed"}
                )

    return categorized_ranges_with_ids


def cal_wtp_postbuffer(predicted_range, label_anomaly_range, buffer_end_point):
    label_start, label_end = label_anomaly_range
    post_buffer_start = label_end + 1
    post_buffer_end = buffer_end_point

    max_possible_distance = sum(abs(post_buffer_end - y) for y in range(label_start, label_end + 1))
    if max_possible_distance == 0:
        return [1.0] * (predicted_range[1] - predicted_range[0] + 1)

    weights = []
    for x in range(predicted_range[0], predicted_range[1] + 1):
        if post_buffer_start <= x <= post_buffer_end:
            distance = sum(abs(x - y) for y in range(label_start, label_end + 1))
            normalized_distance = distance / max_possible_distance
            weights.append(1 - normalized_distance)
    return weights


def cal_wtp_prebuffer(predicted_range, label_anomaly_range, prebuffer_start_point):
    label_start, label_end = label_anomaly_range
    pre_buffer_start = prebuffer_start_point
    pre_buffer_end = label_start - 1

    max_possible_distance = sum(abs(y - pre_buffer_start) for y in range(label_start, label_end + 1))
    if max_possible_distance == 0:
        return [1.0] * (predicted_range[1] - predicted_range[0] + 1)

    weights = []
    for x in range(predicted_range[0], predicted_range[1] + 1):
        if pre_buffer_start <= x <= pre_buffer_end:
            distance = sum(abs(y - x) for y in range(label_start, label_end + 1))
            normalized_distance = distance / max_possible_distance
            weights.append(1 - normalized_distance)
    return weights


def cal_wfn_partial_missed(predicted_range, label_anomaly_range, delta_buffer):
    label_start, label_end = label_anomaly_range
    buffer_end = label_start + delta_buffer

    max_distance = sum(abs(label_end - y) for y in range(label_start, label_end + 1))
    if max_distance == 0:
        return [1.0] * (predicted_range[1] - predicted_range[0] + 1)

    weights_fn = []
    for point in range(predicted_range[0], predicted_range[1] + 1):
        if point <= buffer_end:
            weight = 1.0
        else:
            distance = sum(abs(point - y) for y in range(label_start, buffer_end + 1))
            normalized_distance = distance / max_distance
            weight = 1 - normalized_distance
        weights_fn.append(weight)
    return weights_fn


def apply_weights(categorized_ranges_with_ids, label_anomaly_ranges):
    weights = {"TP": [], "FP": [], "FN": []}

    for category, ranges_with_ids in categorized_ranges_with_ids.items():
        for item in ranges_with_ids:
            pred_range, pred_id = item["range"], item["id"]
            label_id = extract_id(pred_id)

            if category == "true_detection":
                weights["TP"].extend([1] * (pred_range[1] - pred_range[0] + 1))

            if category == "partial_missed":
                actual_ranges = [
                    range_info["range"]
                    for range_info in categorized_ranges_with_ids["true_detection"]
                    if extract_id(range_info["id"]) == label_id
                ]
                if actual_ranges:
                    actual_range = actual_ranges[0]
                    actual_range_length = actual_range[1] - actual_range[0] + 1
                    weights["FN"].extend(
                        cal_wfn_partial_missed(
                            pred_range, label_anomaly_ranges[label_id], actual_range_length
                        )
                    )

            elif category in ["post_buffer", "pre_buffer"]:
                if category == "post_buffer":
                    tp_weights = cal_wtp_postbuffer(
                        pred_range, label_anomaly_ranges[label_id], item["post_buffer_end"]
                    )
                    weights["TP"].extend(tp_weights)
                    weights["FP"].extend(1 - w for w in tp_weights)
                else:
                    if any(extract_id(x["id"]) == label_id for x in categorized_ranges_with_ids["true_detection"]):
                        tp_weights = cal_wtp_prebuffer(
                            pred_range, label_anomaly_ranges[label_id], item["pre_buffer_start"]
                        )
                        weights["TP"].extend(tp_weights)
                        weights["FP"].extend(1 - w for w in tp_weights)
                    else:
                        weights["FP"].extend([1] * (pred_range[1] - pred_range[0] + 1))

            elif category == "outside":
                weights["FP"].extend([1] * (pred_range[1] - pred_range[0] + 1))

    for i, label_range in enumerate(label_anomaly_ranges):
        if not any(extract_id(item["id"]) == i for item in categorized_ranges_with_ids["true_detection"]):
            weights["FN"].extend([1] * (label_range[1] - label_range[0] + 1))

    summed_weights = {key: float(sum(value)) for key, value in weights.items()}

    precision_denom = summed_weights["TP"] + summed_weights["FP"]
    recall_denom = summed_weights["TP"] + summed_weights["FN"]
    precision = summed_weights["TP"] / precision_denom if precision_denom > 0 else 0.0
    recall = summed_weights["TP"] / recall_denom if recall_denom > 0 else 0.0

    return precision, recall


def clean_and_compute_auc_pr(recall, precision):
    clean_precision = []
    clean_recall = []
    prev_recall = -1.0

    for p, r in zip(precision, recall):
        if r >= prev_recall:
            clean_precision.append(p)
            clean_recall.append(r)
            prev_recall = r

    return auc(np.array(clean_recall), np.array(clean_precision))


def compute_adjusted_scores(
    threshold, y_score, actual_anomaly_ranges, e_buffer, d_buffer, time_series_length
):
    binary_predicted = (y_score >= threshold).astype(int)
    predicted_ranges = convert_vector_to_events_pate(binary_predicted)
    categorized_predicted_ranges = categorize_predicted_ranges_with_ids(
        predicted_ranges,
        actual_anomaly_ranges,
        e_buffer,
        d_buffer,
        time_series_length,
    )
    return apply_weights(categorized_predicted_ranges, actual_anomaly_ranges)


def compute_f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def handle_binary_scores(
    y_true, y_score, e_buffer, d_buffer, num_splits_max_buffer, include_zero
):
    actual_anomaly_ranges = convert_vector_to_events_pate(y_true)
    time_series_length = len(y_true)
    f1_score_list = []

    for selected_early_buffer in generate_buffer_points(
        e_buffer, num_splits_max_buffer, include_zero
    ):
        for selected_delayed_buffer in generate_buffer_points(
            d_buffer, num_splits_max_buffer, include_zero
        ):
            predicted_ranges = convert_vector_to_events_pate(y_score)
            categorized_predicted_ranges = categorize_predicted_ranges_with_ids(
                predicted_ranges,
                actual_anomaly_ranges,
                selected_early_buffer,
                selected_delayed_buffer,
                time_series_length,
            )
            precision, recall = apply_weights(categorized_predicted_ranges, actual_anomaly_ranges)
            f1_score_list.append(compute_f1_score(precision, recall))

    return float(np.mean(f1_score_list)) if f1_score_list else 0.0


def handle_continuous_scores(
    y_true,
    y_score,
    e_buffer,
    d_buffer,
    drop_intermediate,
    big_data,
    num_desired_thresholds,
    num_splits_max_buffer,
    include_zero,
):
    auc_pr_list = []
    actual_anomaly_ranges = convert_vector_to_events_pate(y_true)
    time_series_length = len(y_true)

    fps_orig, tps_orig, thresholds = binary_clf_curve(y_true, y_score)

    if drop_intermediate and len(tps_orig) > 2:
        optimal_idxs = np.where(
            np.concatenate(
                [[True], np.logical_or(np.diff(tps_orig[:-1]), np.diff(tps_orig[1:])), [True]]
            )
        )[0]
        fps_orig = fps_orig[optimal_idxs]
        tps_orig = tps_orig[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if big_data and len(thresholds) > 0:
        percentiles = np.linspace(100, 0, num_desired_thresholds)
        thresholds = np.percentile(thresholds, percentiles)

    for selected_early_buffer in generate_buffer_points(
        e_buffer, num_splits_max_buffer, include_zero
    ):
        for selected_delayed_buffer in generate_buffer_points(
            d_buffer, num_splits_max_buffer, include_zero
        ):
            results = [
                compute_adjusted_scores(
                    threshold,
                    y_score,
                    actual_anomaly_ranges,
                    selected_early_buffer,
                    selected_delayed_buffer,
                    time_series_length,
                )
                for threshold in thresholds
            ]

            if results:
                precision, recall = zip(*results)
                precision = np.array(precision, dtype=float)
                recall = np.array(recall, dtype=float)
            else:
                precision = np.array([], dtype=float)
                recall = np.array([], dtype=float)

            precision = np.hstack(([1.0], precision))
            recall = np.hstack(([0.0], recall))
            auc_pr = clean_and_compute_auc_pr(recall, precision)
            auc_pr_list.append(auc_pr)

    return float(np.mean(auc_pr_list)) if auc_pr_list else 0.0


def PATE(
    y_true,
    y_score,
    e_buffer=100,
    d_buffer=100,
    pos_label=1,
    sample_weight=None,
    n_jobs=1,
    drop_intermediate=True,
    Big_Data=True,
    num_desired_thresholds=250,
    num_splits_MaxBuffer=1,
    include_zero=True,
    binary_scores=False,
):
    del pos_label, sample_weight, n_jobs

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if binary_scores:
        return handle_binary_scores(
            y_true, y_score, e_buffer, d_buffer, num_splits_MaxBuffer, include_zero
        )

    return handle_continuous_scores(
        y_true,
        y_score,
        e_buffer,
        d_buffer,
        drop_intermediate,
        Big_Data,
        num_desired_thresholds,
        num_splits_MaxBuffer,
        include_zero,
    )
