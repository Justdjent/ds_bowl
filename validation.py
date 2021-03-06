import numpy as np
import utils
from torch import nn
from torch.nn import functional as F

def validation_binary(model: nn.Module, criterion, valid_loader, num_classes=None):
    model.eval()
    losses = []

    jaccard = []

    for inputs, targets in valid_loader:
        inputs = utils.variable(inputs, volatile=True)
        targets = utils.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        jaccard += [get_jaccard(targets, (outputs > 0).float()).data[0]]

    valid_loss = np.mean(losses)  # type: float

    valid_jaccard = np.mean(jaccard)

    print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard))
    metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard}
    return metrics


def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return (intersection / (union - intersection + epsilon)).mean()


def validation_multi(model: nn.Module, criterion, valid_loader, num_classes):
    model.eval()
    losses = []
    jaccard = []
    # confusion_matrix = np.zeros(
    #     (num_classes, num_classes), dtype=np.uint32)
    for inputs, targets in valid_loader:
        inputs = utils.variable(inputs, volatile=True)
        targets = utils.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        for cls in range(num_classes):
            if cls == 0:
                jaccard_target = (targets[:, 0] == cls).float()
            else:
                jaccard_target = (targets[:, cls - 1] == 1).float()
            # jaccard_output = outputs[:, cls].exp()

            jaccard_output = F.sigmoid(outputs[:, cls])
            jaccard += [get_jaccard(jaccard_target, jaccard_output)]
        #     intersection = (jaccard_output * jaccard_target).sum()
        #
        #     union = jaccard_output.sum() + jaccard_target.sum() + eps
        # output_classes = outputs[:, 0].data.cpu().numpy().argmax(axis=1)
        # target_classes = targets[:, 0].data.cpu().numpy()
        # confusion_matrix += calculate_confusion_matrix_from_arrays(
        #     output_classes, target_classes, num_classes)

    # confusion_matrix = confusion_matrix[1:, 1:]  # exclude background
    valid_loss = np.mean(losses)  # type: float
    valid_jaccard = np.mean(jaccard)
    # ious = {'iou_{}'.format(cls + 1): iou
    #         for cls, iou in enumerate(calculate_iou(confusion_matrix))}
    #
    # dices = {'dice_{}'.format(cls + 1): dice
    #          for cls, dice in enumerate(calculate_dice(confusion_matrix))}
    #
    # average_iou = np.mean(list(ious.values()))
    # average_dices = np.mean(list(dices.values()))

    # print(
    #     'Valid loss: {:.4f}, average IoU: {:.4f}, average Dice: {:.4f}'.format(valid_loss, average_iou, average_dices))
    # print(
    #     'Valid loss: {:.4f}'.format(valid_loss))
    print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard.data[0]))
    # metrics = {'valid_loss': valid_loss, 'iou': average_iou}
    # metrics = {'valid_loss': valid_loss}
    metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard.data[0]}
    # metrics.update(ious)
    # metrics.update(dices)
    return metrics


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious


def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices

def calc_metric(labels, y_pred):
    # Compute number of objects
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))
    print("Number of true objects:", true_objects)
    print("Number of predicted objects:", pred_objects)
    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn)
        print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

