import os
import sys
import numpy
import json

from abc import ABCMeta
from abc import abstractmethod
from collections import defaultdict

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

from six import with_metaclass

import numpy as np
from torch.nn import functional as F
from torch import nn
import torch


def create_coarse_data(model_name, grouping):
    """
    Loads coarse labels for training and testing. Requires labels file named "<model_name>_<grouping[i]>.txt" in labels
    folder for each i. Labels file should list the coarse label for each fine label and be readable by numpy.loadtxt.
    :param model_name: Name of the used dataset.
    :param grouping: List of names of groupings, e.g. ['20_group_similar', 'default']. Requires to be in labels folder.
    Repetitions are allowed, e.g. ['id','id','id'].
    :return: cats: List of category mappings (each a numpy array); e.g. cats[i][label] gives coarse category of label in
    i-th grouping.
    """
    cats = []
    for i in range(0, len(grouping)):
        cats.append(
            numpy.loadtxt(os.path.join('labels', model_name + '_' + grouping[i] + '.txt'), dtype=numpy.int32))
    return cats


def create_grouping_v2(matrix, num_groups, group_similar, save_labels, model_name, size_margin=1.1, class_map=None):
    """
    Algorithm to generate coarse groupings of fine labels.
    :param matrix: Confusion matrix of fine labels after training.
    :param num_groups: Number of coarse groups to generate.
    :param group_similar: Flag, whether similar labels should be grouped together. True for 'Group Similar', False for
    'Split Similar' groupings.
    :param save_labels: Flag whether generated coarse labels should be saved. Save location is labels folder. Save name
    is <model_name>_<num_groups>_group_similar.txt or analogously with split_similar. Warning: Overwrites existing.
    :param model_name: Name of the dataset the labels are generated for.
    :param size_margin: Factor for maximum group size against average group size (rounds up).
    """

    if class_map is not None:
        matrix = matrix[class_map][:, class_map]
        print(matrix.shape)
        print(matrix)
    for i in range(numpy.shape(matrix)[0]):
        matrix[i][i] = 0
    print('sum of conf_matrix', matrix.sum())
    if group_similar:
        matrix = numpy.max(matrix) - matrix

    num_classes = matrix.shape[0]

    for i in range(numpy.shape(matrix)[0]):
        matrix[i][i] = 0
    print('conf_matrix', matrix)

    matrix_symm = 0.5 * numpy.add(matrix, numpy.transpose(matrix))
    print('conf_matrix_symm', matrix_symm)

    current_ranking = [0] * num_classes
    for i in range(0, num_classes):
        current_ranking[i] = numpy.sum(matrix_symm[i])

    next = numpy.argmin(
        current_ranking)  # choose most confused element first for group similar / least confused for split similar

    clusters = []  # stores assigned fine classes for each cluster
    for i in range(0, num_groups):
        clusters.append([])

    print(len(clusters))

    clusters[0].append(next)  # first element in cluster 0
    current_ranking = numpy.zeros(
        (num_classes, num_groups + 1))  # stores for each element number of confused classes from each cluster
    current_ranking[next][num_groups] = -1  # last row for sum of confusions
    #  num_classes-1 more iterations
    for zzyxz in range(1, num_classes):
        initialized = True
        for cluster in clusters:
            if len(cluster) == 0:
                initialized = False
        #  compute confusions with each cluster for each element
        for i in range(0, num_classes):
            # only for elements not assigned yet (assigned marked by -1)
            if current_ranking[i][num_groups] != -1:
                for j in range(0, num_groups):
                    current_ranking[i][j] = 0
                    for k in clusters[j]:
                        current_ranking[i][j] += matrix_symm[k][i]
                    if len(clusters[j]) > 0:
                        current_ranking[i][j] /= len(clusters[j])

                # current_ranking[i][num_groups] = numpy.max(current_ranking[i][:num_groups]) - numpy.min(
                # current_ranking[i][:num_groups])
                closest_dist = current_ranking[i][0]  # just an inital value, distance to cluster 0
                for j in range(0, num_groups):
                    # only check populated clusters
                    if len(clusters[j]) > 0:
                        temp = current_ranking[i][j]
                        if temp < closest_dist:
                            closest_dist = temp

                current_ranking[i][num_groups] = closest_dist

        extreme_arg = []  # extreme is max or min
        same = 0
        if not initialized:
            max = -0.5  # ignores -1
            for i in range(0, len(current_ranking)):
                if current_ranking[i][num_groups] >= max:
                    if current_ranking[i][num_groups] == max:
                        extreme_arg.append(i)
                        same += 1
                    else:
                        same = 0
                        max = current_ranking[i][num_groups]
                        extreme_arg = [i]
        else:
            min = numpy.max(current_ranking, axis=0)[num_groups]
            for i in range(0, len(current_ranking)):
                if current_ranking[i][num_groups] <= min and current_ranking[i][num_groups] > -0.5:
                    if current_ranking[i][num_groups] == min:
                        extreme_arg.append(i)
                        same += 1
                    else:
                        same = 0
                        min = current_ranking[i][num_groups]
                        extreme_arg = [i]
        if same > 0:
            next = numpy.random.choice(numpy.array(extreme_arg))
        else:
            next = extreme_arg[0]

        while (True):
            pref = numpy.argmin(current_ranking[next][:num_groups])
            if not group_similar:
                # if no confusions with multiple clusters, choose random (instead of the first one with 0)
                if current_ranking[next][pref] == 0:
                    rand = numpy.random.randint(low=0, high=num_groups)
                    while (True):
                        if rand == 0:
                            break
                        pref += 1
                        if pref >= num_groups:
                            pref = 0
                        if current_ranking[next][pref] == 0:
                            rand -= 1

            if len(clusters[pref]) >= size_margin * num_classes / num_groups:
                current_ranking[next][pref] = 1000000000
            else:
                clusters[pref].append(next)
                break
        current_ranking[next][num_groups] = -1

    labels = [0] * num_classes
    for i in range(0, num_groups):
        for j in range(0, len(clusters[i])):
            labels[clusters[i][j]] = i
    print(labels)

    if group_similar and save_labels:
        numpy.savetxt(os.path.join('labels', model_name + '_' + str(num_groups) + '_group_similar.txt'),
                      numpy.array(labels, dtype=numpy.int32),
                      delimiter=" ")
        labels = numpy.loadtxt(os.path.join('labels', model_name + '_' + str(num_groups) + '_group_similar.txt'),
                               dtype=numpy.int32)
    elif save_labels:
        numpy.savetxt(os.path.join('labels', model_name + '_' + str(num_groups) + '_split_similar.txt'),
                      numpy.array(labels, dtype=numpy.int32),
                      delimiter=" ")
        labels = numpy.loadtxt(os.path.join('labels', model_name + '_' + str(num_groups) + '_split_similar.txt'),
                               dtype=numpy.int32)
    counts = []
    for i in range(0, num_groups):
        counts.append(0)
    for elem in labels:
        counts[elem] = counts[elem] + 1
    print(counts)

    return labels

class Unpacker(nn.Module):
    def __init__(self, module, input_key='image', output_key='output'):
        super(Unpacker, self).__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.module = module

    def forward(self, sample, *args, **kwargs):
        x = sample[self.input_key]
        x.data = x.data.float()
        x = self.module.forward(x)
        sample[self.output_key] = x
        return sample

class CrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Wrapper for `torch.nn.CrossEntropyLoss` that accepts dictionaries as input.

    :param output_key:
        key in given sample dict which maps to the output tensor
    :param target_key:
        key in given sample dict which maps to the target tensor
    :param weight: a manual rescaling weight given to each
        class. If given, it has to be a Tensor of size `C`. Otherwise, it is
        treated as if having all ones.
    :param reduction:
        Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``.
        ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
        Default: ``'mean'``
    :param ignore_index: Specifies a target value that is ignored
        and does not contribute to the input gradient. When
        :attr:`size_average` is ``True``, the loss is averaged over
        non-ignored targets.
    """
    def __init__(self, output_key='output', target_key='label',
                 weight=None, reduction='mean', ignore_index=-100):
        nn.CrossEntropyLoss.__init__(self,
                                     weight=weight,
                                     reduction=reduction,
                                     ignore_index=ignore_index)
        self.output_key = output_key
        self.target_key = target_key

    def forward(self, sample):
        #print(sample[self.output_key].argmax(dim=1))
        #print(sample[self.output_key].argmin(dim=1))
        #print('tar:',sample[self.target_key].squeeze())
        return nn.CrossEntropyLoss.forward(
             self, sample[self.output_key], sample[self.target_key].squeeze()
        )

class Metric(with_metaclass(ABCMeta)):
    """
    Abstract class which is to be inherited by every metric.
    As usual, this class is designed to handle dictionaries.

    :param output_key: the key with which the output is found in the input dictionary
    :param target_key: the key with which the target is found in the imput dictionary
    :param metric_key: the key with which the metric is to be stored in the output dictionary
    """

    def __init__(self, output_key='output', target_key='target_image', metric_key='metric'):
        self.metric_key = metric_key
        self.output_key = output_key
        self.target_key = target_key
        self.total = 0
        self.steps = 0

    def reset(self):
        self.total = 0
        self.steps = 0

    @abstractmethod
    def value(self):
        """
        implement to return the currently stored metric.
        :return: current metric
        """
        pass

    @abstractmethod
    def __call__(self, sample):
        """
        implement a call that processes given sample dictionary to compute a metric.
        :param sample: dictionary
        :return: metric
        """
        pass


class AuxAccuracyMetric(Metric):
    """
    Computes the top-k accuracy metric for given classification scores,
    i.e. predicted class probabilities.
    The metric is computed as {1 if target_i in top_k_predicted_classes_i else 0 for all i in n} / n.
    Takes list of output_keys for base output and auxiliary classifiers, base should be given first, auxiliary
    classifiers	make use of cats to map outputs.
    :param output_key: the key(s) with which the output(s) is/are found in the input dictionary
    :param target_key: the key with which the target is found in the input dictionary
    :param cats: List with one entry per aux classifier. Entry should be a numpy array mapping a fine class to its
                coarse class. Note: some implementations may expect an array of coarse classes for each fine class.
    :param exp_combination_factor: Weight by which each output is exponentiated to change influence of aux classifiers
                in combination
    """

    def __init__(self, top_k=1,
                 output_keys='output', target_key='label', cats=None, compute_combined=False,
                 exp_combination_factor=0.3):
        Metric.__init__(self, output_keys, target_key, None)
        try:
            top_k[0]
        except (AttributeError, TypeError):
            top_k = (top_k,)
        self.top_k = top_k
        if not isinstance(output_keys, list):
            output_keys = [output_keys]
        self.output_keys = output_keys
        self.target_key = target_key
        self.compute_combined = compute_combined
        self.num_outputs = len(self.output_keys)
        if compute_combined:
            self.num_outputs += 1

        cuda_cats = []
        for cat in cats:
            cuda_cats.append(torch.from_numpy(cat).long().cuda())
        self.cats = cuda_cats
        self.ecf = exp_combination_factor
        self.reset()

    def reset(self):
        self.correct = []
        for i in range(self.num_outputs):
            self.correct.append(defaultdict(lambda: 0))
        self.n = []
        for i in range(self.num_outputs):
            self.n.append(0)

    def value(self):
        dic = {}
        for i in range(self.num_outputs):
            for k, v in self.correct[i].items():
                dic[k] = v.item() / self.n[i]
        # return {k: v.item() / self.n for k, v in self.correct.items()}
        return dic

    def __call__(self, sample):
        for i in range(len(self.output_keys)):
            # compute accuracy of base output
            if i == 0:
                n = sample[self.output_key[i]].size()[0]
                self.n[i] += n
                target = sample[self.target_key].data.view(n, 1)
                output = sample[self.output_key[i]].data
                predictions = output.sort(1, descending=True)[1]
                for k in self.top_k:
                    self.correct[i]['base top%d acc' % k] += \
                        predictions[:, :k].eq(target).sum()
            # compute accuracy of auxiliary outputs
            else:
                n = sample[self.output_key[i]].size()[0]
                self.n[i] += n
                target = self.cats[i - 1][sample[self.target_key].data.view(n, 1)]
                output = sample[self.output_key[i]].data
                predictions = output.sort(1, descending=True)[1]
                for k in self.top_k:
                    self.correct[i]['aux' + str(i - 1) + ' top%d acc' % k] += \
                        predictions[:, :k].eq(target).sum()
        if self.compute_combined:
            # multiply base output with auxiliary outputs mapped to the fine classes and raised to factor self.ecf
            n = sample[self.output_key[0]].size()[0]
            self.n[-1] += n
            target = sample[self.target_key].data.view(n, 1)
            output = F.softmax(sample[self.output_key[0]].data, dim=1)
            for i in range(len(self.output_keys) - 1):
                aux_output = F.softmax(sample[self.output_key[i + 1]].data, dim=1)
                aux_reshaped = torch.zeros((n, output.shape[1]), dtype=torch.float32).cuda()
                for j in range(self.cats[i].size()[0]):
                    aux_reshaped[:, j] = aux_output[:, self.cats[i][j]]
                aux_reshaped = aux_reshaped.pow(self.ecf)
                output = output * aux_reshaped

            predictions = output.sort(1, descending=True)[1]
            for k in self.top_k:
                self.correct[-1][' top%d comb. acc' % k] += \
                    predictions[:, :k].eq(target).sum()

        return self.value()


def count_parameters(model):
    """
    Count parameters of the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_confusion_matrix_step(prev_matrix, batch_output, batch_target):
    """
    Auxiliary function for generate_confusion_matrix (doing one batch).
    """
    batch_output = torch.argmax(batch_output, dim=1).flatten()
    # print("output",batch_output.size())
    batch_target = batch_target.flatten()
    # print("target",batch_target.size())
    for id1, id2 in zip(batch_target, batch_output):
        prev_matrix[id1, id2] += 1
    return prev_matrix


def generate_confusion_matrix(model, val_loader, num_classes, output_key='output', device='cuda:0'):
    """
    Generate a confusion matrix based on a model, a dataset loader val_loader and the observed output key.
    :return: Confusion matrix as torch tensor.
    """
    model.eval()
    matrix = torch.zeros(size=(num_classes, num_classes)).to(device)
    for iteration, mini_batch in val_loader:
        for sample in mini_batch:
            with torch.no_grad():
                batch_output = model(sample)[output_key]
                batch_target = sample['label']

                matrix = generate_confusion_matrix_step(matrix, batch_output, batch_target)
    return matrix


def get_group_label_names(group, label_names):
    s = '['
    for elem in group:
        label = label_names[elem][:min(len(label_names[elem]), 22)]
        s += label.split(',')[0]
        # s += label_names[elem][:min(len(label_names[elem]), 25)]
        s += '|'
    s = s[:-1]
    s += ']'
    target_len = len(group) * 23 + 2
    s = s.ljust(target_len)
    return s


def generate_imagenet_labels():
    import requests
    LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
    classes = {int(key): value for (key, value)
               in requests.get(LABELS_URL).json().items()}
    return classes
