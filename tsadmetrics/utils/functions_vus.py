import numpy as np

# This file includes code derived from the VUS project:
# https://github.com/TheDatumOrg/VUS
#
# No license was specified in the original repository at the time of inclusion (April 2025),
# which may imply that all rights are reserved by the original author(s).
#
# This code has been copied and modified for internal use only within the tsadmetrics project.
# If you are the author or copyright holder and would like us to remove or relicense
# this code, please contact us.
#
# This file is NOT intended for redistribution as a standalone component or as a derivative
# of the original VUS project unless proper licensing is clarified.


def range_convers_new( label):
		'''
		input: arrays of binary values 
		output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
		'''
		L = []
		i = 0
		j = 0 
		while j < len(label):
			# print(i)
			while label[i] == 0:
				i+=1
				if i >= len(label):  #?
					break			 #?
			j = i+1
			# print('j'+str(j))
			if j >= len(label):
				if j==len(label):
					L.append((i,j-1))
	
				break
			while label[j] != 0:
				j+=1
				if j >= len(label):
					L.append((i,j-1))
					break
			if j >= len(label):
				break
			L.append((i, j-1))
			i = j
		return L

def new_sequence(label, sequence_original, window):
    a = max(sequence_original[0][0] - window // 2, 0)
    sequence_new = []
    for i in range(len(sequence_original) - 1):
        if sequence_original[i][1] + window // 2 < sequence_original[i + 1][0] - window // 2:
            sequence_new.append((a, sequence_original[i][1] + window // 2))
            a = sequence_original[i + 1][0] - window // 2
    sequence_new.append((a, min(sequence_original[len(sequence_original) - 1][1] + window // 2, len(label) - 1)))
    return sequence_new


def sequencing(x, L, window=5):
		label = x.copy().astype(float)
		length = len(label)

		for k in range(len(L)):
			s = L[k][0]
			e = L[k][1]

			x1 = np.arange(e + 1, min(e + window // 2 + 1, length))
			label[x1] += np.sqrt(1 - (x1 - e) / (window))

			x2 = np.arange(max(s - window // 2, 0), s)
			label[x2] += np.sqrt(1 - (s - x2) / (window))

		label = np.minimum(np.ones(length), label)
		return label


def RangeAUC_volume_opt_mem(labels_original, score, windowSize, thre=250):
		window_3d = np.arange(0, windowSize + 1, 1)
		P = np.sum(labels_original)
		seq = range_convers_new(labels_original)
		l = new_sequence(labels_original, seq, windowSize)

		score_sorted = -np.sort(-score)

		tpr_3d = np.zeros((windowSize + 1, thre + 2))
		fpr_3d = np.zeros((windowSize + 1, thre + 2))
		prec_3d = np.zeros((windowSize + 1, thre + 1))

		auc_3d = np.zeros(windowSize + 1)
		ap_3d = np.zeros(windowSize + 1)

		tp = np.zeros(thre)
		N_pred = np.zeros(thre)
		p = np.zeros((thre, len(score)))

		for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
			threshold = score_sorted[i]
			pred = score >= threshold
			p[k] = pred
			N_pred[k] = np.sum(pred)

		for window in window_3d:
			labels_extended = sequencing(labels_original, seq, window)
			L = new_sequence(labels_extended, seq, window)

			TF_list = np.zeros((thre + 2, 2))
			Precision_list = np.ones(thre + 1)
			j = 0

			for i in np.linspace(0, len(score) - 1, thre).astype(int):
				labels = labels_extended.copy()
				existence = 0

				for seg in L:
					labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * p[j][seg[0]:seg[1] + 1]
					if (p[j][seg[0]:(seg[1] + 1)] > 0).any():
						existence += 1
				for seg in seq:
					labels[seg[0]:seg[1] + 1] = 1

				N_labels = 0
				TP = 0
				for seg in l:
					TP += np.dot(labels[seg[0]:seg[1] + 1], p[j][seg[0]:seg[1] + 1])
					N_labels += np.sum(labels[seg[0]:seg[1] + 1])

				TP += tp[j]
				FP = N_pred[j] - TP

				existence_ratio = existence / len(L)

				P_new = (P + N_labels) / 2
				recall = min(TP / P_new, 1)

				TPR = recall * existence_ratio

				N_new = len(labels) - P_new
				FPR = FP / N_new
				Precision = TP / N_pred[j]
				j += 1

				TF_list[j] = [TPR, FPR]
				Precision_list[j] = Precision

			TF_list[j + 1] = [1, 1]
			tpr_3d[window] = TF_list[:, 0]
			fpr_3d[window] = TF_list[:, 1]
			prec_3d[window] = Precision_list

			width = TF_list[1:, 1] - TF_list[:-1, 1]
			height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
			AUC_range = np.dot(width, height)
			auc_3d[window] = (AUC_range)

			width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
			height_PR = Precision_list[1:]
			AP_range = np.dot(width_PR, height_PR)
			ap_3d[window] = (AP_range)
		return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)    

def RangeAUC_volume_opt( labels_original, score, windowSize, thre=250):
		window_3d = np.arange(0, windowSize + 1, 1)
		P = np.sum(labels_original)
		seq = range_convers_new(labels_original)
		l = new_sequence(labels_original, seq, windowSize)

		score_sorted = -np.sort(-score)

		tpr_3d = np.zeros((windowSize + 1, thre + 2))
		fpr_3d = np.zeros((windowSize + 1, thre + 2))
		prec_3d = np.zeros((windowSize + 1, thre + 1))

		auc_3d = np.zeros(windowSize + 1)
		ap_3d = np.zeros(windowSize + 1)

		tp = np.zeros(thre)
		N_pred = np.zeros(thre)

		for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
			threshold = score_sorted[i]
			pred = score >= threshold
			N_pred[k] = np.sum(pred)

		for window in window_3d:

			labels_extended = sequencing(labels_original, seq, window)
			L = new_sequence(labels_extended, seq, window)

			TF_list = np.zeros((thre + 2, 2))
			Precision_list = np.ones(thre + 1)
			j = 0

			for i in np.linspace(0, len(score) - 1, thre).astype(int):
				threshold = score_sorted[i]
				pred = score >= threshold
				labels = labels_extended.copy()
				existence = 0

				for seg in L:
					labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * pred[seg[0]:seg[1] + 1]
					if (pred[seg[0]:(seg[1] + 1)] > 0).any():
						existence += 1
				for seg in seq:
					labels[seg[0]:seg[1] + 1] = 1

				TP = 0
				N_labels = 0
				for seg in l:
					TP += np.dot(labels[seg[0]:seg[1] + 1], pred[seg[0]:seg[1] + 1])
					N_labels += np.sum(labels[seg[0]:seg[1] + 1])

				TP += tp[j]
				FP = N_pred[j] - TP

				existence_ratio = existence / len(L)

				P_new = (P + N_labels) / 2
				recall = min(TP / P_new, 1)

				TPR = recall * existence_ratio
				N_new = len(labels) - P_new
				FPR = FP / N_new

				Precision = TP / N_pred[j]

				j += 1
				TF_list[j] = [TPR, FPR]
				Precision_list[j] = Precision

			TF_list[j + 1] = [1, 1]  # otherwise, range-AUC will stop earlier than (1,1)

			tpr_3d[window] = TF_list[:, 0]
			fpr_3d[window] = TF_list[:, 1]
			prec_3d[window] = Precision_list

			width = TF_list[1:, 1] - TF_list[:-1, 1]
			height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
			AUC_range = np.dot(width, height)
			auc_3d[window] = (AUC_range)

			width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
			height_PR = Precision_list[1:]

			AP_range = np.dot(width_PR, height_PR)
			ap_3d[window] = AP_range

		return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)

def generate_curve(label,score,slidingWindow, version='opt', thre=250):
    if version =='opt_mem':
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = RangeAUC_volume_opt_mem(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)
    else:
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = RangeAUC_volume_opt(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)
        
    X = np.array(tpr_3d).reshape(1,-1).ravel()
    X_ap = np.array(tpr_3d)[:,:-1].reshape(1,-1).ravel()
    Y = np.array(fpr_3d).reshape(1,-1).ravel()
    W = np.array(prec_3d).reshape(1,-1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0])-1)
    
    return Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d
