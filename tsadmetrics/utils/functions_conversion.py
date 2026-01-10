import numpy as np

def pointwise_to_segmentwise(pointwise):
    """Reformat anomaly time series from pointwise to segmentwise"""
    segmentwise = []

    prev = -10
    for point in pointwise:
        if point > prev + 1:
            segmentwise.append([point, point])
        else:
            segmentwise[-1][-1] += 1
        prev = point
    return np.array(segmentwise)


def segmentwise_to_pointwise(segmentwise):
    """Reformat anomaly time series from segmentwise to pointwise"""
    pointwise = []

    for start, end in segmentwise:
        for point in range(start, end + 1):
            pointwise.append(point)

    return np.array(pointwise)


def segmentwise_to_full_series(segmentwise, length):
    """Reformat anomaly time series from segmentwise to full_series"""
    pw = segmentwise_to_pointwise(segmentwise)

    return  pointwise_to_full_series(pw, length)

def pointwise_to_full_series(pointwise, length):
    """Reformat anomaly time series from pointwise to full_series"""
    anomalies_full_series = np.zeros(length)
    if len(pointwise) > 0:
        assert pointwise[-1] < length
        anomalies_full_series[pointwise] = 1
    return np.array(anomalies_full_series)

def full_series_to_pointwise(full_series):
    """Reformat anomaly time series from full_series to pointwise"""
    anomalies_pointwise = []
    for i in range(len(full_series)):
        if full_series[i]==1:
            anomalies_pointwise.append(i)
    return np.array(anomalies_pointwise)

def full_series_to_segmentwise(full_series):
    """Reformat anomaly time series from full_series to segmentwise"""
    anomalies_segmentwise = []
    i=0
    while i < len(full_series):
        if full_series[i] == 1:
            start = i
            while i < len(full_series) and full_series[i] == 1:
                i += 1
            end = i - 1
            anomalies_segmentwise.append([start, end])
        else:
            i += 1
    return np.array(anomalies_segmentwise)