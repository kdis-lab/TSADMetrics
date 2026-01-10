from ....base.Metric import Metric
import numpy as np
import math
from ....utils.functions_conversion import full_series_to_segmentwise, full_series_to_pointwise
class EnhancedTimeseriesAwareFScore(Metric):
    """
    Calculate enhanced time series aware F-score for anomaly detection in time series.
    
    This metric is similar to the range-based F-score in that it accounts for both detection existence 
    and overlap proportion. Additionally, it requires that a significant fraction :math:`{\\theta_r}` of each true anomaly 
    segment be detected, and that a significant fraction :math:`{\\theta_p}` of each predicted segment overlaps with the 
    ground truth. Finally, F-score contributions from each event are weighted by the square root of the 
    true segment’s length, providing a compromise between point-wise and segment-wise approaches.

    Reference:
        Implementation based on:
            https://link.springer.com/article/10.1007/s10618-023-00988-8
        For more information, see the original paper:
            https://doi.org/10.1145/3477314.3507024

    Attributes:
        name (str):
            Fixed name identifier for this metric: `"etaf"`.
        binary_prediction (bool):
            Indicates whether this metric expects binary predictions. Always `True`
            since it requires binary anomaly scores.

    Parameters:
        theta_p (float):
            Minimum fraction (:math:`{0 \\leq \\theta_p \\leq 1}`) of a predicted segment that must be overlapped 
            by ground truth to count as detected.
        theta_r (float):
            Minimum fraction (:math:`{0 \\leq \\theta_r \\leq 1}`) of a true segment that must be overlapped 
            by predictions to count as detected.
    """
    name = "etaf"
    binary_prediction = True
    param_schema = {
        "theta_p": {
            "default": 0.5,
            "type": float
        },
        "theta_r": {
            "default": 0.5,
            "type": float
        }
    }

    def __init__(self, **kwargs):
        super().__init__(name="etaf", **kwargs)

    def _min_max_norm(self, value, org_min, org_max, new_min, new_max) -> float:
        if org_min == org_max:
            return new_min
        else:
            return (float)(new_min) + (float)(value - org_min) * (new_max - new_min) / (org_max - org_min)
    
    def _decaying_func(self, val: float) -> float:
        assert (-6 <= val <= 6)
        return 1 / (1 + math.exp(val))
    
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
    
    def _gen_ambiguous(self,y_true_sw, y_pred_sw):
        ambiguous_inst = []
        for i in range(len(y_true_sw)):
            start_id = y_true_sw[i][1] + 1
            end_id = start_id 

            if i + 1 < len(y_true_sw) and end_id > y_true_sw[i + 1][0]:
                end_id = y_true_sw[i + 1][0] - 1

            if start_id > end_id:
                start_id = -2
                end_id = -1

            ambiguous_inst.append([start_id, end_id])
        return ambiguous_inst
    
    def _compute_overlap_scores_and_weights(self, y_true_sw, y_pred_sw):

        predictions_weight = []
        predictions_total_weight = 0.0
        #computing weights
        for a_prediction in y_pred_sw:
            first, last = a_prediction
            temp_weight = math.sqrt(last-first+1)
            predictions_weight.append(temp_weight)
            predictions_total_weight += temp_weight

        #computing the score matrix
        ambiguous_inst = self._gen_ambiguous(y_true_sw, y_pred_sw)
        overlap_score_mat_org = np.zeros((len(y_true_sw), len(y_pred_sw)))
        for anomaly_id in range(len(y_true_sw)):
            for prediction_id in range(len(y_pred_sw)):
                overlap_score_mat_org[anomaly_id, prediction_id] = \
                    float(self._overlap_and_subsequent_score(y_true_sw[anomaly_id], ambiguous_inst[anomaly_id], y_pred_sw[prediction_id]))

        #computing the maximum scores for each anomaly or prediction
        max_anomaly_score = []
        max_prediction_score = []
        for an_anomaly in y_true_sw:
            start, end = an_anomaly
            max_anomaly_score.append(float(self._sum_of_func(start, end, start, end, self._uniform_func)))
        for a_prediction in y_pred_sw:
            max_prediction_score.append(a_prediction[1]-a_prediction[0] + 1)


        return predictions_weight, predictions_total_weight, overlap_score_mat_org, max_anomaly_score, max_prediction_score

    def _pruning(self, y_true_sw, y_pred_sw, overlap_score_mat_elm, max_anomaly_score, max_prediction_score):


        while True:
            tars = overlap_score_mat_elm.sum(axis=1)/max_anomaly_score
            elem_anomaly_ids = set(np.where(tars<self.params['theta_r'])[0]) - set(np.where(tars==0.0)[0])
            for id in elem_anomaly_ids:
                overlap_score_mat_elm[id] = np.zeros(len(y_pred_sw))
            taps = overlap_score_mat_elm.sum(axis=0)/max_prediction_score
            elem_prediction_ids = set(np.where(taps<self.params['theta_p'])[0]) - set(np.where(taps==0.0)[0])
            for id in elem_prediction_ids:
                overlap_score_mat_elm[:, id] = np.zeros(len(y_true_sw))

            if len(elem_anomaly_ids) == 0 and len(elem_prediction_ids) == 0:
                break
        return overlap_score_mat_elm

    def _etar_p(self, y_true_sw, y_pred_sw, overlap_score_mat_elm, predictions_weight, predictions_total_weight, max_prediction_score):
        """
        Calculate precision for the enhanced time series aware F-score.

        Parameters:
            y_true_sw (np.array):
                The ground truth binary labels for the time series data, in segment-wise format.
            y_pred_sw (np.array):
                The predicted binary labels for the time series data, in segment-wise format.
            overlap_score_mat_org (np.array):
                The original overlap score matrix.
            max_anomaly_score (list):
                The maximum scores for each anomaly segment.
            max_prediction_score (list):
                The maximum scores for each prediction segment.
        Returns:
            float: The precision value.
        """
        etap_d = 0
        etap_p = 0
        if len(y_true_sw) == 0.0 or len(y_pred_sw) == 0.0:
            etap_d,etap_p = 0.0, 0.0
        
        etap_d = overlap_score_mat_elm.sum(axis=0) / max_prediction_score
        etap_p = etap_d

        etap_d = np.where(etap_d >= self.params['theta_p'], 1.0, etap_d)
        etap_d = np.where(etap_d <  self.params['theta_p'], 0.0, etap_d)
        corrected_id_list = np.where(etap_d >= self.params['theta_p'])[0]

        detection_scores = etap_d
        portion_scores = etap_p


        scores = (detection_scores + detection_scores * portion_scores)/2
        final_score = 0.0
        for i in range(max(len(scores),len(etap_d),len(corrected_id_list))):
            if i < len(scores):
                final_score += float(predictions_weight[i]) * scores[i]
            
        
        final_score /= float(predictions_total_weight)
        return final_score

    
    def _etar_d(self, y_true_sw, y_pred_sw, overlap_score_mat_elm, max_anomaly_score, max_prediction_score):
        """
        Calculate recall for the enhanced time series aware F-score.

        Parameters:
            y_true_sw (np.array):
                The ground truth binary labels for the time series data, in segment-wise format.
            y_pred_sw (np.array):
                The predicted binary labels for the time series data, in segment-wise format.
            overlap_score_mat_org (np.array):
                The original overlap score matrix.
            max_anomaly_score (list):
                The maximum scores for each anomaly segment.
            max_prediction_score (list):
                The maximum scores for each prediction segment.
        Returns:
            float: The recall value.
        """
        if len(y_true_sw) == 0.0 or len(y_pred_sw) == 0.0:
            return np.zeros(len(y_true_sw)), []
        theta = self.params['theta_r']
        scores = overlap_score_mat_elm.sum(axis=1) / max_anomaly_score
        scores = np.where(scores >= theta, 1.0, scores)
        scores = np.where(scores <  theta, 0.0, scores)
        detected_id_list = np.where(scores >= theta)[0]

        return scores, detected_id_list
    def _compute(self, y_true, y_pred):
        """
        Calculate the enhanced time series aware F-score.

        Parameters:
            y_true (np.array):
                The ground truth binary labels for the time series data.
            y_pred (np.array):
                The predicted binary labels for the time series data.

        Returns:
            float: The time series aware F-score, which is the harmonic mean of precision and recall, adjusted by the beta value.
        """

        if np.sum(y_pred) == 0:
            return 0
        y_true_sw = np.array(full_series_to_segmentwise(y_true))
        y_pred_sw = np.array(full_series_to_segmentwise(y_pred))

        predictions_weight, predictions_total_weight, overlap_score_mat_org, max_anomaly_score, max_prediction_score = self._compute_overlap_scores_and_weights(y_true_sw,y_pred_sw)
        overlap_score_mat_elm = self._pruning(y_true_sw, y_pred_sw, overlap_score_mat_org, max_anomaly_score, max_prediction_score)
        detection_scores, detected_id_list = self._etar_d(y_true_sw, y_pred_sw, overlap_score_mat_elm, max_anomaly_score, max_prediction_score)
        precision = self._etar_p(y_true_sw, y_pred_sw, overlap_score_mat_elm, predictions_weight, predictions_total_weight, max_prediction_score)

        if len(y_true_sw) == 0 or len(y_pred_sw) == 0:
            portion_scores = 0.0
        else:
            portion_scores = overlap_score_mat_elm.sum(axis=1) / max_anomaly_score
            portion_scores = np.where(portion_scores > 1.0, 1.0, portion_scores)
        recall = ((detection_scores + detection_scores * portion_scores)/2).mean()

        if precision + recall == 0:
            return 0.0
        else:
            return (2 * recall * precision) / (recall + precision)
        
