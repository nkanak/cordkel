import os
from tabulate import tabulate
import utils
import numpy
import json

def print_logit_table(logit_logs):
    unique_combination_name = set([obj['name'] for objs in logit_logs for obj in objs])
    data = []
    results = []
    for name in unique_combination_name:
        scores = [obj['score'] for objs in logit_logs for obj in objs if obj['name'] == name]
        recalls = [obj['recall'] for objs in logit_logs for obj in objs if obj['name'] == name]
        precisions = [obj['precision'] for objs in logit_logs for obj in objs if obj['name'] == name]
        data.append(
            [name, round(numpy.mean(scores), 4), round(numpy.mean(recalls), 4), round(numpy.mean(precisions), 4)])
        results.append({
            'name': name,
            'accuracies': scores,
            'recalls': recalls,
            'precisions': precisions
        })
    with open('results_logit.json', 'w', newline='') as f:
        json.dump(results, f, indent=2)
    print(tabulate(data, headers=['Name', 'Accuracy', 'Recall', 'Precision'], tablefmt='grid'))

def print_nn_table(nn_logs):
    unique_combination_name = set([obj['name'] for objs in nn_logs for obj in objs])
    data = []
    results = []
    for name in unique_combination_name:
        scores = [obj['history']['val_accuracy'][-1] for objs in nn_logs for obj in objs if obj['name'] == name]
        recalls = [obj['history']['val_recall'][-1] for objs in nn_logs for obj in objs if obj['name'] == name]
        precisions = [obj['history']['val_precision'][-1] for objs in nn_logs for obj in objs if obj['name'] == name]
        train_loss = [obj['history']['loss'][-1] for objs in nn_logs for obj in objs if obj['name'] == name]
        test_loss = [obj['history']['val_loss'][-1] for objs in nn_logs for obj in objs if obj['name'] == name]
        data.append(
            [
                name,
                round(numpy.mean(scores), 4),
                round(numpy.mean(recalls), 4),
                round(numpy.mean(precisions), 4),
                round(numpy.mean(train_loss), 4),
                round(numpy.mean(test_loss), 4),
                abs(round(numpy.mean(train_loss), 4) - round(numpy.mean(test_loss), 4)),
            ])

        results.append({
            'name': name,
            'accuracies': scores,
            'recalls': recalls,
            'precisions': precisions
        })
    with open('results_nn.json', 'w', newline='') as f:
        json.dump(results, f, indent=2)
    print(tabulate(data, headers=['Name', 'Accuracy', 'Recall', 'Precision', 'Train loss', 'Test loss', 'Abs loss diff'], tablefmt='grid'))

def run():
    filenames = [filename for filename in os.listdir('results') if '.json' in filename]
    logs = []
    for filename in filenames:
        with open('results/%s'%(filename)) as f:
            logs.append(json.load(f))
    logit_logs = [log['logit'] for log in logs]
    nn_logs = [log['nn'] for log in logs]

    print('############## Logistic regression ##############')
    print_logit_table(logit_logs)
    print('############## Neural Network ##############')
    print_nn_table(nn_logs)

if __name__ == '__main__':
    run()