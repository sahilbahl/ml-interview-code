from typing import Optional

import numpy as np


def accuracy_score(y_gt: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_gt) == 0:
        return np.nan

    if len(y_gt) != len(y_pred):
        raise ValueError("Dimensions are not matching")

    return np.count_nonzero(y_gt == y_pred) / len(y_gt)


def precision_score(
    y_gt: np.ndarray,
    y_pred: np.ndarray,
    average: Optional[str] = None,
    zero_division: float = 0.0,
) -> np.ndarray | float:
    cf_mat = confusion_matrix(y_gt, y_pred)

    # Handle zero division
    column_sums = np.sum(cf_mat, axis=0)
    # Replace zeros with ones before division to avoid warnings
    # We'll handle the zeros after division
    safe_column_sums = np.copy(column_sums)
    safe_column_sums[safe_column_sums == 0] = 1

    precision_scores = cf_mat.diagonal() / safe_column_sums

    # Set scores for classes with no predictions to the zero_division value
    precision_scores[column_sums == 0] = float(zero_division)

    if average is None:
        return precision_scores
    elif average == "macro":
        return np.mean(precision_scores)
    elif average == "weighted":
        return np.average(precision_scores, weights=np.sum(cf_mat, axis=1))
    else:
        raise ValueError("average must be one of 'macro', 'weighted', or None")


def recall_score(
    y_gt: np.ndarray,
    y_pred: np.ndarray,
    average: Optional[str] = None,
    zero_division: float = 0.0,
) -> np.ndarray | float:
    """Calculate recall score with zero division handling."""
    cf_mat = confusion_matrix(y_gt, y_pred)

    # Handle zero division
    row_sums = np.sum(cf_mat, axis=1)
    # Create a safe version to avoid division by zero warnings
    safe_row_sums = np.copy(row_sums)
    safe_row_sums[safe_row_sums == 0] = 1

    # Calculate recall scores
    recall_scores = cf_mat.diagonal() / safe_row_sums

    # Set scores for classes with no actual samples to the zero_division value
    recall_scores[row_sums == 0] = float(zero_division)

    if average is None:
        return recall_scores
    elif average == "macro":
        return np.mean(recall_scores)
    elif average == "weighted":
        # For weighted average, use non-zero weights to avoid division by zero
        if np.sum(row_sums) == 0:
            return float(zero_division)
        return np.average(recall_scores, weights=row_sums)
    else:
        raise ValueError("average must be one of 'macro', 'weighted', or None")


def f1_score(
    y_gt: np.ndarray, y_pred: np.ndarray, average: Optional[str] = None
) -> np.ndarray | float:
    precision_scores = precision_score(y_gt, y_pred)
    recall_scores = recall_score(y_gt, y_pred)

    f1_scores = (
        2 * (precision_scores * recall_scores) / (precision_scores + recall_scores)
    )

    if average is None:
        return f1_scores
    elif average == "macro":
        return np.mean(f1_scores)
    elif average == "weighted":
        cf_mat = confusion_matrix(y_gt, y_pred)
        class_counts = np.sum(cf_mat, axis=1)
        return np.average(f1_scores, weights=class_counts)
    else:
        raise ValueError("average must be one of 'macro', 'weighted', or None")


def confusion_matrix(y_gt: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # num_pts = len(y_gt)
    # classes, indexes = np.unique(np.concatenate((y_gt, y_pred)), return_inverse=True)

    # gt_class_index = defaultdict(list)
    # pred_class_index = defaultdict(list)
    # for index, cl in enumerate(indexes):
    #     if index < num_pts:
    #         gt_class_index[cl].append(index)
    #     else:
    #         pred_class_index[cl].append(index - num_pts)

    # num_classes = len(classes)
    # confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # for gt_class in classes:
    #     for pred_class in classes:
    #         confusion_matrix[gt_class, pred_class] =
    #           len(np.intersect1d(gt_class_index[gt_class], pred_class_index[pred_class]))

    # return confusion_matrix
    if len(y_gt) == 0 or len(y_pred) == 0:
        return np.empty(0)

    num_data_pts = len(y_gt)
    classes, indexes = np.unique(np.concatenate((y_gt, y_pred)), return_inverse=True)
    y_gt_idx = indexes[0:num_data_pts]
    y_pred_idx = indexes[num_data_pts:]
    # class_to_index = {label: idx for idx, label in enumerate(classes)}

    # y_gt_idx = np.array([class_to_index[label] for label in y_gt])
    # y_pred_idx = np.array([class_to_index[label] for label in y_pred])

    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    np.add.at(cm, (y_gt_idx, y_pred_idx), 1)
    return cm
