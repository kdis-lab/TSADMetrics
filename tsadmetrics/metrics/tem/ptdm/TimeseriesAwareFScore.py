from ....base.Metric import Metric
from ....utils.functions_conversion import full_series_to_segmentwise
import math
class TimeseriesAwareFScore(Metric):
    """
    Calculate time series aware F-score for anomaly detection in time series.

    This metric is based on the range_based_f_score, but introduces two key modifications.  
    First, a predicted anomalous segment is only counted as a true positive if it covers at least a fraction 
    :math:`{\\theta}` of the ground‑truth anomaly range. Second, each labeled anomaly is extended by a tolerance window of 
    length :math:`{\\delta}` at its end, within which any overlap contribution decays linearly from full weight down to zero.  
    Unlike the original range-based formulation, this variant omits cardinality and positional bias terms, 
    focusing solely on overlap fraction and end‑tolerance decay.
    

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8
        For more information, see the original paper:
            https://doi.org/10.1145/3357384.3358118

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"taf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        y_true (np.array):
            The ground truth binary labels for the time series data.
        y_pred (np.array):
            The predicted binary labels for the time series data.
        beta (float):
            The beta value, which determines the weight of precision in the combined score.
            Default is 1, which gives equal weight to precision and recall.
        alpha (float):
            Relative importance of the existence reward versus overlap reward (:math:`{0 \\leq \\alpha \\leq 1}`).
        delta (float):
            Tolerance window length at the end of each true anomaly segment.
                - If past_range is True, :math:`{\\delta}` must be a float in (0, 1], representing the fraction of the segment’s 
                    length to extend. E.g., :math:`{\\delta}` = 0.5 extends a segment of length 10 by 5 time steps.
                - If past_range is False, :math:`{\\delta}` must be a non-negative integer, representing an absolute number of 
                    time steps to extend each segment.
        theta (float):
            Minimum fraction (:math:`{ 0 \\leq \\theta \\leq 1}`) of the true anomaly range that must be overlapped by 
            predictions for the segment to count as detected.
        past_range (bool):
            Determines how :math:`{\\delta}` is interpreted.
                - True: :math:`{\\delta}` is treated as a fractional extension of each segment’s length.
                - False: :math:`{\\delta}` is treated as an absolute number of time steps.
    """
    name = "taf"
    binary_prediction = True
    param_schema = {
        "beta": {
            "default": 1.0,
            "type": float
        },
        "alpha": {
            "default": 0.5,
            "type": float
        },
        "delta": {
            "default": 5,
            "type": float  
        },
        "theta": {
            "default": 0.5,
            "type": float
        },
        "past_range": {
            "default": False,
            "type": bool
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="taf", **kwargs)

    def _gen_ambiguous(self, y_true_sw, y_pred_sw):
        ambiguous_inst = []
        delta = self.params["delta"]
        past_range = self.params["past_range"]
        for i in range(len(y_true_sw)):
            start_id = y_true_sw[i][1] + 1
            end_id = end_id = start_id + delta

            if past_range:
                end_id = start_id + int(delta * (y_true_sw[i][1] - y_true_sw[0]))
            
            #if the next anomaly occurs during the theta, update the end_id
            if i+1 < len(y_true_sw) and end_id > y_true_sw[i+1][0]:
                end_id = y_true_sw[i+1][0] - 1

            if start_id > end_id:
                start_id = -2
                end_id = -1

            ambiguous_inst.append([start_id, end_id])
        return ambiguous_inst


    def _min_max_norm(self, value, org_min, org_max, new_min, new_max) -> float:
        if org_min == org_max:
            return new_min
        else:
            return (float)(new_min) + (float)(value - org_min) * (new_max - new_min) / (org_max - org_min)
    
    def _decaying_func(self, val: float) -> float:
        assert (-6 <= val <= 6)
        return 1 / (1 + math.exp(val))

    def _ascending_func(self, val: float) -> float:
        assert (-6 <= val <= 6)
        return 1 / (1 + math.exp(val * -1))

    def _uniform_func(self, val: float) -> float:
        return 1.0
    
    def _sum_of_func(self, start_time, end_time, org_start, org_end,
                     func) -> float:
        val = 0.0
        for timestamp in range(start_time, end_time + 1):
            val += func(self._min_max_norm(timestamp, org_start, org_end, -6, 6))
        return val
    
    def _overlap_and_subsequent_score(self, anomaly, ambiguous, prediction) -> float:
        score = 0.0

        detected_start = max(anomaly[0], prediction[0])
        detected_end = min(anomaly[1], prediction[1])

        score += self._sum_of_func(detected_start, detected_end,
                                   anomaly[0], anomaly[1], self._uniform_func)

        if ambiguous[0] < ambiguous[1]:
            detected_start = max(ambiguous[0], prediction[0])
            detected_end = min(ambiguous[1], prediction[1])

            score += self._sum_of_func(detected_start, detected_end,
                                       ambiguous[0], ambiguous[1], self._decaying_func)
        return score
    def _TaP_dp_value(self, y_true_sw, y_pred_sw):

        ambiguous_inst = self._gen_ambiguous(y_true_sw, y_pred_sw)
        threshold = self.params["theta"]
        correct_predictions = []
        total_score = 0.0
        total_score_p = 0.0
        for prediction_id in range(len(y_pred_sw)):
            max_score = y_pred_sw[prediction_id][1] - y_pred_sw[prediction_id][0] + 1

            score = 0.0
            for anomaly_id in range(len(y_true_sw)):
                anomaly = y_true_sw[anomaly_id]
                ambiguous = ambiguous_inst[anomaly_id]

                score += self._overlap_and_subsequent_score(anomaly, ambiguous, y_pred_sw[prediction_id])
            total_score_p += score / max_score
            if (score/max_score) >= threshold:
                total_score += 1.0
                correct_predictions.append(prediction_id)

        if len(y_pred_sw) == 0:
            return 0.0, 0.0
            
        else:
            TaP_p_value = total_score_p / len(y_pred_sw)
            return TaP_p_value, total_score / len(y_pred_sw)
        

    def _TaR_dp_value(self, y_true_sw, y_pred_sw):
        ambiguous_inst = self._gen_ambiguous(y_true_sw, y_pred_sw)
        threshold = self.params["theta"]
        total_score = 0.0
        detected_anomalies = []
        total_score_p = 0.0
        for anomaly_id in range(len(y_true_sw)):
            anomaly = y_true_sw[anomaly_id]
            ambiguous = ambiguous_inst[anomaly_id]

            max_score = self._sum_of_func(anomaly[0], anomaly[1],
                                          anomaly[0], anomaly[1], self._uniform_func)

            score = 0.0
            for prediction in y_pred_sw:
                score += self._overlap_and_subsequent_score(anomaly, ambiguous, prediction)
            
            total_score_p += min(1.0, score/max_score)
            if min(1.0, score / max_score) >= threshold:
                total_score += 1.0
                detected_anomalies.append(anomaly_id)

        if len(y_true_sw) == 0:
            return 0.0, 0.0
        else:
            TaR_p_value = total_score_p / len(y_true_sw)
            return TaR_p_value, total_score / len(y_true_sw)
        
    
    def _compute_precision(self, y_true_sw, y_pred_sw):
        """
        Calculate precision for time series aware F-score.

        Parameters:
            y_true_sw (np.array): Ground truth binary labels in segment-wise format.
            y_pred_sw (np.array): Predicted binary labels in segment-wise format.
        Returns:
            float: The precision value.
        """
        tapp_value, tapd_value = self._TaP_dp_value(y_true_sw, y_pred_sw)

        alpha = self.params["alpha"]
        return alpha * tapd_value + (1 - alpha) * tapp_value
    
    def _compute_recall(self, y_true_sw, y_pred_sw):
        """
        Calculate recall for time series aware F-score.

        Parameters:
            y_true_sw (np.array): Ground truth binary labels in segment-wise format.
            y_pred_sw (np.array): Predicted binary labels in segment-wise format.
        Returns:
            float: The recall value.
        """
        tarp_value, tard_value = self._TaR_dp_value(y_true_sw, y_pred_sw)

        alpha = self.params["alpha"]
        return alpha * tard_value + (1 - alpha) * tarp_value
    
    def _compute(self, y_true, y_pred):
        """
        Calculate the time series aware F-score.

        Parameters:
            y_true (np.array): Ground truth binary labels.
            y_pred (np.array): Predicted binary labels.

        Returns:
            float: The time series aware F-score.
        """
        y_true_sw = full_series_to_segmentwise(y_true)
        y_pred_sw = full_series_to_segmentwise(y_pred)



        precision = self._compute_precision(y_true_sw, y_pred_sw)
        recall = self._compute_recall(y_true_sw, y_pred_sw)
        if precision == 0 or recall == 0:
            return 0

        beta = self.params["beta"]
        return ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
