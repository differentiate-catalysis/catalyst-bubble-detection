import numpy as np
import cv2
from scipy.ndimage.morphology import distance_transform_edt
import torch

def bde(prediction, label):

    tmp = cv2.findContours(label, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    label_contours = tmp[0] if len(tmp) == 2 else tmp[1]
    labels_contour = np.concatenate(label_contours).squeeze(axis=1) if label_contours else np.array([])

    tmp = cv2.findContours(prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    prediction_contours = tmp[0] if len(tmp) == 2 else tmp[1]
    prediction_contour = np.concatenate(prediction_contours).squeeze(axis=1) if prediction_contours else np.array([])

    if labels_contour.size == 0 or prediction_contour.size == 0:
        return np.nan

    label_map = np.ones(label.shape)
    label_indices = np.ravel_multi_index(labels_contour.T, label_map.shape)
    np.put(label_map, label_indices, 0)
    label_dist = distance_transform_edt(label_map)

    bde_value = np.mean(label_dist[prediction_contour[:, 0], prediction_contour[:, 1]])

    return bde_value

# Compute precistion and recall given contours
def calc_precision_recall(contours_a, contours_b, threshold):

    count = 0
    for b in contours_b:
        if contours_a.shape[0] == 0: break
        #np.linalg.norm quickly finds Euclidian distance
        distances = np.linalg.norm(b - contours_a, axis=1)
        if np.min(distances) < threshold:
            count += 1

    if count != 0:
        precision_recall = count/len(contours_b)
    else:
        precision_recall = 0

    return precision_recall, count, len(contours_b)

# Compute BF1 score using predicted image and ground truth
def bf1score(prediction, label, n_class=None, threshold=2,
            mask=False, mask_contour=False):

    if n_class == None:
        n_class = max(np.unique(label)) + 1

    classes = np.arange(n_class)
    bfscores = np.zeros(n_class, dtype=float)
    precisions = np.zeros(n_class, dtype=float)
    recalls = np.zeros(n_class, dtype=float)

    prediction, label = np.asarray(prediction).astype(np.uint8), np.asarray(label).astype(np.uint8)
    if isinstance(mask, np.ndarray):
        prediction = np.multiply(prediction, mask).astype(np.uint8)
        label = np.multiply(label, mask).astype(np.uint8)

    for target_class in classes:

        gt = label.copy()
        # 0 class for background
        if target_class == 0:
            gt[label != 0] = 0
            gt[label == 0] = 1
            if isinstance(mask, np.ndarray):
                gt[mask == 0] = 0
        else:
            # Make other classes 0 value
            gt[label != target_class] = 0

        # Find contours using OpenCV-python package
        tmp = cv2.findContours(gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        label_contours = tmp[0] if len(tmp) == 2 else tmp[1]
        labels_contour = np.concatenate(label_contours).squeeze(axis=1) if label_contours else np.array([])
        if len(labels_contour.shape) == 2:
            labels_contour = labels_contour[(labels_contour[:, 0] != 0) & (labels_contour[:, 0] != label.shape[0] - 1) & (labels_contour[:, 1] != 0) & (labels_contour[:, 1] != label.shape[1] - 1)]

        # Exclude mask contour points
        if isinstance(mask, np.ndarray):
            labels_contour = np.array([x for x in labels_contour if np.intersect1d(np.where(mask_contour[:, 0] == x[0]), np.where(mask_contour[:, 1] == x[1])).size == 0])

        pr = prediction.copy()
        # 0 class for background
        if target_class == 0:
            pr[prediction != 0] = 0
            pr[prediction == 0] = 1
            if isinstance(mask, np.ndarray):
                pr[mask == 0] = 0
        else:
            # Make other classes 0 value
            pr[pr != target_class] = 0

        tmp = cv2.findContours(pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        prediction_contours = tmp[0] if len(tmp) == 2 else tmp[1]
        prediction_contour = np.concatenate(prediction_contours).squeeze(axis=1) if prediction_contours else np.array([])
        if len(prediction_contour.shape) == 2:
            prediction_contour = prediction_contour[(prediction_contour[:, 0] != 0) & (prediction_contour[:, 0] != prediction.shape[0] - 1) & (prediction_contour[:, 1] != 0) & (prediction_contour[:, 1] != prediction.shape[1] - 1)]
        # Exclude mask contour points
        if isinstance(mask, np.ndarray):
            prediction_contour = np.array([x for x in prediction_contour if np.intersect1d(np.where(mask_contour[:, 0] == x[0]), np.where(mask_contour[:, 1] == x[1])).size == 0])

        # Calculate precision and recall
        precision, numerator, denominator = calc_precision_recall(
            labels_contour, prediction_contour, threshold)    # Precision
        #print("\tprecision:", denominator, numerator)

        recall, numerator, denominator = calc_precision_recall(
            prediction_contour, labels_contour, threshold)    # Recall
        #print("\trecall:", denominator, numerator)
        if precision == 0 or recall == 0:
            f1 = np.nan
        else:
            f1 = 2*recall*precision/(recall+precision)

        # Save values as list form
        bfscores[target_class] = f1
        precisions[target_class] = precision
        recalls[target_class] = recall

    return bfscores, precisions, recalls


def metrics(conf_mat, verbose=True):
    c = conf_mat.shape[0]

    # Ignore dividing by zero error
    np.seterr(divide='ignore', invalid='ignore')

    # Divide diagonal entries of confusion matrix by sum of its columns and
    # rows to respectively obtain precision and recall.
    precision = np.nan_to_num(conf_mat.diagonal()/conf_mat.sum(0))
    recall = conf_mat.diagonal()/conf_mat.sum(1)
    f1_score = (2 * precision * recall) / (precision + recall)

    # Initialize empty array for IoU computation
    IoUs = np.zeros(c)
    union_sum = 0

    # Loop through rows of confusion matrix; divide each diagonal entry by the
    # sum of its row and column (while avoiding double-counting that entry).
    for i in range(c):
        union = conf_mat[i, :].sum()+conf_mat[:, i].sum()-conf_mat[i, i]
        union_sum += union
        IoUs[i] = conf_mat[i, i]/union

    # Accuracy computed by dividing sum of confusion matrix diagonal with
    # the sum of the confusion matrix
    acc = conf_mat.diagonal().sum()/conf_mat.sum()
    IoU = IoUs.mean()

    # IoU of second class corresponds to that of Somas, which we record.
    class_iou = IoUs[1]
    if verbose:
        print('precision:', np.round(precision, 5), precision.mean())
        print('recall:', np.round(recall, 5), recall.mean())
        print('IoUs:', np.round(IoUs, 5), IoUs.mean())
        print('Full F1 Score:', np.round(f1_score, 5), f1_score.mean())
    return acc, IoU, precision, recall, class_iou

def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    #correct = predictions.eq(labels.cpu()).sum().item()
    correct = predictions.eq(labels).sum().item()
    acc = correct/np.prod(labels.shape)
    return acc

def tomography_mask(dimensions):
    X, Y = 852, 852
    hard_mask = np.tile(int(0), (X, Y))
    for coord in [[j,i] for i in range(Y) for j in range(X) if np.sqrt((i-Y/2+20)**2+(j-X/2+45)**2) <= 340]:
        hard_mask[coord[0], coord[1]] = int(1)

    tmp = cv2.findContours(hard_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    mask_contours = tmp[0] if len(tmp) == 2 else tmp[1]
    mask_contour = np.concatenate(mask_contours).squeeze(axis=1) if mask_contours else np.array([])
    hard_mask = hard_mask[:dimensions[0], :dimensions[1]]
    return hard_mask, mask_contour

def none_mask(dimensions):
    return None, None

masks_mapping = {
    'tomography': tomography_mask,
    'none': none_mask
}
