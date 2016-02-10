import math

#benchmark.m, converted
def benchmark(pred_labels, true_labels):
    errors = 0
    for i in range(len(pred_labels)):
        if pred_labels[i] != true_labels[i]:
            errors+=1
    return errors / float(len(true_labels))