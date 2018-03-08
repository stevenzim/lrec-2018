from sklearn.metrics import classification_report

import numpy as np


def evaluate_results(y_expected, y_predicted, y_probs = None, prob_thresh = .95):
    '''Get evaluation results e.g. Precision, Recall and F1 scores to 4 sig figs
        If probabilities for ys is provided then only return classification report
        with prediction probabilities greater than threshold default 95%'''
    if y_probs == None:
        return classification_report(y_expected, y_predicted, digits=4)
    #below usefule for when you want to only evaluate based on predictions with probability > threshold
    #NOTE: does not have test YET
    thresh_expected = []
    thresh_predicted = []
    for expect, predict, prob in zip(y_expected,y_predicted,y_probs):
        if prob.max() > prob_thresh:
            thresh_expected.append(expect)
            thresh_predicted.append(predict)
        else:
            continue
    return classification_report(np.asarray(thresh_expected), np.asarray(thresh_predicted), digits=4)


def sem_eval_f1_avg(Y_gold, Y_predict):
    """
    Useful to produce evaluation metric required for SemEval (F1-Pos + F1-Neg) / 2
    NOTE: This metric can be passed in as scorer for Gridsearch for optimization of paramaters
    :param Y_gold: Y list of gold labels
    :param Y_predict: Y list of predicted labels
    :return: Returns only the avg score of F1 for positive and negative labels
    """
    eval_report = evaluate_results(Y_gold, Y_predict)
    eval_report_data = eval_report.split()
    sem_eval_neg_f1 = float(eval_report_data[7])
    sem_eval_pos_f1 = float(eval_report_data[17])
    sem_eval_avg_f1 = (sem_eval_neg_f1 + sem_eval_pos_f1) / 2
    return sem_eval_avg_f1



def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes
    taken from https://gist.github.com/zachguo/10296432"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels:
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)):
            cell = "%{0}.1d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print