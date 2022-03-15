import json
from collections import defaultdict

from sklearn.metrics import confusion_matrix

from metadata.class_labels import ntu120_code_labels, anu_bullying_pair_labels, bly_labels, anubis_ind_actions
# from test_fields.kinetics_analysis import get_kinetics_dict
import numpy as np


def get_result_confusion_jsons(gt, pred, data_type, acc_f_name_prefix=None):
    if 'ntu' in data_type:
        code_labels = ntu120_code_labels
    elif 'bly' in data_type:
        code_labels = bly_labels
        gt = np.array(gt)[:, 0]
    elif 'front' in data_type:
        code_labels = {}
        tmp = {}
        for key in anubis_ind_actions:
            tmp[anubis_ind_actions[key]] = key
        for key in tmp:
            code_labels[tmp[key] + 1] = key
    # elif 'kinetics' in data_type:
    #     code_labels = get_kinetics_dict()
    else:
        raise NotImplementedError

    correct_dict = defaultdict(list)
    for idx in range(len(gt)):
        correct_dict[gt[idx]].append(int(pred[idx] == gt[idx]))
    correct_dict_ = correct_dict.copy()

    for a_key in correct_dict:
        correct_dict[a_key] = '{:.6f}'.format(sum(correct_dict[a_key]) / len(correct_dict[a_key]))

    label_acc = {}
    for a_key in correct_dict:
        label_acc[code_labels[int(a_key) + 1]] = float(correct_dict[a_key])

    label_acc = dict(sorted(label_acc.items(), key=lambda item: item[1]))
    label_acc_keys = list(label_acc.keys())

    conf_mat = confusion_matrix(gt, pred)

    most_confused = {}
    for i in correct_dict.keys():
        confusion_0 = np.argsort(conf_mat[int(i)])[::-1][0]
        confusion_1 = np.argsort(conf_mat[int(i)])[::-1][1]

        most_confused[code_labels[int(i) + 1]] = [
            "{}  {}".format(code_labels[confusion_0 + 1], conf_mat[int(i)][confusion_0]),
            "{}  {}".format(code_labels[confusion_1 + 1], conf_mat[int(i)][confusion_1]),
            "{}".format(len(correct_dict_[i]))
        ]
    most_confused_ = {}
    for i in label_acc_keys:
        most_confused_[i] = most_confused[i]

    if acc_f_name_prefix is not None:
        with open('{}_confusion_matrix.json'.format(acc_f_name_prefix), 'w') as f:
            json.dump(most_confused_, f, indent=4)
        with open('{}_accuracy_per_class.json'.format(acc_f_name_prefix), 'w') as f:
            json.dump(label_acc, f, indent=4)

    return label_acc, most_confused_

