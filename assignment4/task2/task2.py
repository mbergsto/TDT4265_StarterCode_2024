import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    x_max_p = prediction_box[2]
    x_min_p = prediction_box[0]
    y_max_p = prediction_box[3]
    y_min_p = prediction_box[1]

    area_pred = (x_max_p - x_min_p) * (y_max_p - y_min_p)

    x_max_gt = gt_box[2]
    x_min_gt = gt_box[0]
    y_max_gt = gt_box[3]
    y_min_gt = gt_box[1]

    area_gt = (x_max_gt - x_min_gt) * (y_max_gt - y_min_gt)

    # Compute intersection

    x_max = min(x_max_p, x_max_gt)
    x_min = max(x_min_p, x_min_gt)
    y_max = min(y_max_p, y_max_gt)
    y_min = max(y_min_p, y_min_gt)

    if x_max < x_min or y_max < y_min:
        intersection = 0                    # If no overlap = no intersection = no iou
    
    else:
        intersection = (x_max - x_min) * (y_max - y_min)

    # Compute union
    union = area_pred + area_gt - intersection

    # Compute iou
    iou = intersection / union
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1
    
    precision = num_tp / (num_tp + num_fp)
    
    return precision


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fn == 0:
        return 0
    
    recall = num_tp / (num_tp + num_fn)
    
    return recall


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    
    matches = []
    for i, pred_box in enumerate(prediction_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                matches.append((i, j, iou))    # Store the indices and the iou value in a list as a tuple


    # Sort all matches on IoU in descending order

    matches = sorted(matches, key=lambda x: x[2], reverse=True)      # Sort the matches based on the iou value (index 2) in descending order
    
    # Find all matches with the highest IoU threshold

    best_matches = []
    return_prediction_boxes = []
    return_gt_boxes = []
    for i, j, _ in matches:
        if i not in [x[0] for x in best_matches] and j not in [x[1] for x in best_matches]:
            best_matches.append((i, j))                             # Store the indices in a list as a tuple
            return_prediction_boxes.append(prediction_boxes[i])     # Store the prediction boxes in a list
            return_gt_boxes.append(gt_boxes[j])                     # Store the ground truth boxes in a list

    return np.array(return_prediction_boxes), np.array(return_gt_boxes)


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    # Find all matches
    matched_prediction_boxes, _ = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    # Calculate true positives, false positives, false negatives
    true_pos = len(matched_prediction_boxes)                        # True positives = number of matched prediction boxes
    false_pos = len(prediction_boxes) - true_pos                    # False positives = number of prediction boxes - matched prediction boxes = unmatched prediction boxes
    false_neg = len(gt_boxes) - true_pos                            # False negatives = number of ground truth boxes - matched prediction boxes = unmatched ground truth boxes

    return {"true_pos": true_pos, "false_pos": false_pos, "false_neg": false_neg}


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    # Calculate true positives, false positives, false negatives for all images
    true_pos = 0
    false_pos = 0
    false_neg = 0
    
    for pred_box, gt_box in zip(all_prediction_boxes, all_gt_boxes):
        individual_image_result = calculate_individual_image_result(pred_box, gt_box, iou_threshold)
        true_pos += individual_image_result["true_pos"]
        false_pos += individual_image_result["false_pos"]
        false_neg += individual_image_result["false_neg"]

    # Calculate precision and recall
    precision = calculate_precision(true_pos, false_pos, false_neg)
    recall = calculate_recall(true_pos, false_pos, false_neg)

    return precision, recall

def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE

    precisions = [] 
    recalls = []

    # Calculate precision and recall for each confidence threshold
    for confidence_threshold in confidence_thresholds:
        true_pos = 0
        false_pos = 0
        false_neg = 0

        # Calculate true positives, false positives, false negatives for all images
        for pred_box, gt_box, confidence_score in zip(all_prediction_boxes, all_gt_boxes, confidence_scores):
            # Filter out the prediction boxes with confidence score below the current confidence threshold
            accepted_pred_boxes = pred_box[confidence_score > confidence_threshold]
            individual_image_result = calculate_individual_image_result(accepted_pred_boxes, gt_box, iou_threshold)
            true_pos += individual_image_result["true_pos"]
            false_pos += individual_image_result["false_pos"]
            false_neg += individual_image_result["false_neg"]
        
        precision = calculate_precision(true_pos, false_pos, false_neg)
        recall = calculate_recall(true_pos, false_pos, false_neg)
        precisions.append(precision)
        recalls.append(recall)
    
    
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    average_precision = 0
    calculated_precisions = []

    for recall_level in recall_levels:
        # Find max precision for recall >= recall_level
        valid_precisions = precisions[recalls >= recall_level]
        current_precision = np.max(valid_precisions) if len(valid_precisions) > 0 else 0
        calculated_precisions.append(current_precision)

    average_precision = np.mean(calculated_precisions)
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
