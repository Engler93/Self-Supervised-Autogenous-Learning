import os
import sys
import numpy
import json

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from tensorflow.python.keras import utils


def create_coarse_data(y_train, y_test, model_name, grouping):
	"""
	Loads coarse labels for training and testing. Requires labels file named "<model_name>_<grouping[i]>.txt" in labels
	folder for each i. Labels file should list the coarse label for each fine label and be readable by numpy.loadtxt.
	:param y_train: Fine labels for training set.
	:param y_test: Fine labels for validation set.
	:param model_name: Name of the used dataset.
	:param grouping: List of names of groupings, e.g. ['20_group_similar', 'default']. Requires to be in labels folder.
	Repetitions are allowed, e.g. ['id','id','id'].
	:return: cats: List of category mappings (each a numpy array); y_train_c: list of coarse label set for training for
	each coarse grouping; y_test_c: list of coarse label set for validation/testing for each coarse grouping.
	"""
	cats = []
	for i in range(0, len(grouping)):
		cats.append(numpy.loadtxt(os.path.join('labels', model_name + '_' + grouping[i] + '.txt'), dtype=numpy.int32))
	y_train_c = []
	y_test_c = []
	# create coarse training data
	for cat in cats:
		y_train_c_temp = numpy.zeros(0)
		y_test_c_temp = numpy.zeros(0)
		for y in y_train:
			y_train_c_temp = numpy.append(y_train_c_temp, cat[y])  # category of label y stored in cat dictionary
		for y in y_test:
			y_test_c_temp = numpy.append(y_test_c_temp, cat[y])
		y_train_c.append(y_train_c_temp)
		y_test_c.append(y_test_c_temp)
	return cats, y_train_c, y_test_c


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

	for i in range(numpy.shape(matrix)[0]):
		matrix[i][i] = 0

	if group_similar:
		matrix = numpy.max(matrix) - matrix

	num_classes = matrix.shape[0]

	for i in range(numpy.shape(matrix)[0]):
		matrix[i][i] = 0

	matrix_symm = 0.5 * numpy.add(matrix, numpy.transpose(matrix))

	current_ranking = [0] * num_classes
	for i in range(0, num_classes):
		current_ranking[i] = numpy.sum(matrix_symm[i])

	next = numpy.argmin(
		current_ranking)  # choose most confused element first for group similar / least confused for split similar

	clusters = []  # stores assigned fine classes for each cluster
	for i in range(0, num_groups):
		clusters.append([])

	clusters[0].append(next)  # first element in cluster 0
	current_ranking = numpy.zeros(
		(num_classes, num_groups + 1))  # stores for each element number of confused classes from each cluster
	current_ranking[next][num_groups] = -1  # last row for sum of confusions

	#  99 more iterations
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
			# print('same:',same)
			# print(extreme)
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

	if save_labels:
		suffix = '_group_similar.txt' if group_similar else '_split_similar.txt'
		numpy.savetxt(os.path.join('labels', model_name + '_' + str(num_groups) + suffix),
					  numpy.array(labels, dtype=numpy.int32),
					  delimiter=" ")
		labels = numpy.loadtxt(os.path.join('labels', model_name + '_' + str(num_groups) + suffix),
							   dtype=numpy.int32)

	counts = []
	for i in range(0, num_groups):
		counts.append(0)
	for elem in labels:
		counts[elem] = counts[elem] + 1
	print(counts)

	return labels


def log_combined_acc(model, x_test, y_test, cats, num_classes, combined_accuracy_log, exp_combination_factor=1.0):
	"""
	Computes combined accuracy, prints it and stores in given combined_accuracy_log (list).
	"""
	pred = model.predict(x_test)
	y_actual = y_test.argmax(axis=1)  # correct class
	y_pred = pred[0]
	y_pred_c = pred[1:]

	# combining prediction
	for y in zip(y_pred, *y_pred_c):
		if max(y[0]) < 1.01:
			for i in range(0, num_classes):
				for cat, y_c in zip(cats, y[1:]):
					y[0][i] = y[0][i] * (y_c[cat[i]] ** exp_combination_factor)

	for i in range(0, y_pred.shape[0]):
		if y_pred[i].sum() > 0:
			y_pred[i] = y_pred[i] / y_pred[i].sum()

	y_predict = y_pred.argmax(axis=1)  # predicted class (after combination)
	acc = accuracy_score(y_actual, y_predict)

	# logging combined accuracy
	combined_accuracy_log.append(acc)
	print('combined acc: ' + str(acc))


def classification_analysis(model, x_test, y_test, cats, num_classes, from_generator=False):
	"""
	Collect and print stats about errors of fine and coarse classification, helps analysing potential of aux loss.
	Outputs that count fixable cases, etc. only check the first coarse grouping. Ignoring conflicting predicitons
	evaluates all coarse groupings.
	:param model: Keras model that was trained and should be analyzed.
	:param x_test: Inputs of fine validation/testing set.
	:param y_test: Labels of fine validation/testing set.
	:param cats: List of category mappings.
	:param num_classes: Number of fine classes.
	"""
	if not from_generator:
		pred = model.predict(x_test)
		y_actual = y_test.argmax(axis=1)  # correct class
		total = len(x_test)
	else:
		pred = model.predict_generator(x_test, workers=0, max_queue_size=80,
									   use_multiprocessing=False)  # x_test = ds_val
		y_actual = y_test  # is already onehot
		print('pred0', pred[0].shape)
		print('pred1', pred[1].shape)
		total = x_test.size
	y_pred = pred[0]
	y_pred_c = pred[1:]
	y_predict = y_pred.argmax(axis=1)  # predicted class (before combination)
	y_predict_c = []
	for pred in y_pred_c:
		y_predict_c.append(pred.argmax(axis=1))
	y_predict_c = numpy.array(y_predict_c)
	y_predict_c = numpy.swapaxes(y_predict_c, 0, 1)

	num_auxs = len(cats)
	correct_array = numpy.zeros((num_auxs + 1, num_auxs + 1))  # for [i,j]: if i correct, how many from j are correct.
	wrong_array = numpy.zeros((num_auxs + 1, num_auxs + 1))  # for [i,j]: if i wrong, how many from j are wrong.

	for y, y_c, y_act in zip(y_predict, y_predict_c, y_actual):
		for i in range(len(cats)):
			if y_c[i] == cats[i][y_act]:
				for j in range(len(cats)):
					if y_c[j] == cats[j][y_act]:
						correct_array[i][j] = correct_array[i][j] + 1
				if y == y_act:
					correct_array[i][-1] = correct_array[i][-1] + 1
			else:
				for j in range(len(cats)):
					if y_c[j] != cats[j][y_act]:
						wrong_array[i][j] = wrong_array[i][j] + 1
				if y != y_act:
					wrong_array[i][-1] = wrong_array[i][-1] + 1
		if y == y_act:
			for j in range(len(cats)):
				if y_c[j] == cats[j][y_act]:
					correct_array[-1][j] = correct_array[-1][j] + 1
			if y == y_act:
				correct_array[-1][-1] = correct_array[-1][-1] + 1
		else:
			for j in range(len(cats)):
				if y_c[j] != cats[j][y_act]:
					wrong_array[-1][j] = wrong_array[-1][j] + 1
			if y != y_act:
				wrong_array[-1][-1] = wrong_array[-1][-1] + 1
	for i in range(correct_array.shape[0] - 1):
		print('If AUX' + str(i) + ' correct, then', correct_array[i], 'also correct and',
			  correct_array[i][i] - correct_array[i], 'wrong.')
	print('If Base correct, then', correct_array[-1], 'also correct and', correct_array[-1][-1] - correct_array[-1],
		  'wrong.')

	for i in range(wrong_array.shape[0] - 1):
		print('If AUX' + str(i) + ' wrong, then', wrong_array[i], 'also wrong and', wrong_array[i][i] - wrong_array[i],
			  'correct.')
	print('If Base wrong, then', wrong_array[-1], 'also wrong and', wrong_array[-1][-1] - wrong_array[-1], 'wrong.')

	correct_when_conflicting_ignored = 0
	wrong_when_conflicting_ignored = 0
	both_wrong_in_kept_out = 0
	coarse_right_in_kept_out = []
	coarse_right_when_conflicting_ignored = []
	for cat in cats:
		coarse_right_in_kept_out.append(0)
		coarse_right_when_conflicting_ignored.append(0)
	fine_right_in_kept_out = 0
	ignored = 0
	not_ignored = 0
	for i in range(0, len(y_predict)):
		different = False
		for cat, y_c in zip(cats, y_pred_c):
			if cat[y_predict[i]] != numpy.argmax(y_c[i]):
				different = True
		if not different:
			not_ignored += 1
			if y_predict[i] == y_actual[i]:
				correct_when_conflicting_ignored += 1
			else:
				wrong_when_conflicting_ignored += 1
			for j in range(0, len(cats)):
				if numpy.argmax(y_pred_c[j][i]) == cats[j][y_actual[i]]:
					coarse_right_when_conflicting_ignored[j] += 1

		else:
			if y_predict[i] == y_actual[i]:
				fine_right_in_kept_out += 1
				for j in range(0, len(cats)):
					if numpy.argmax(y_pred_c[j][i]) == cats[j][y_actual[i]]:
						coarse_right_in_kept_out[j] += 1
			else:
				got_one = False
				for j in range(0, len(cats)):
					if numpy.argmax(y_pred_c[j][i]) == cats[j][y_actual[i]]:
						coarse_right_in_kept_out[j] += 1
						got_one = True
				if not got_one:
					both_wrong_in_kept_out += 1
			ignored += 1
	# print(coarse_right_in_kept_out)

	if correct_when_conflicting_ignored + wrong_when_conflicting_ignored > 0:
		error_when_conflicting_ignored = float(wrong_when_conflicting_ignored) / (
				correct_when_conflicting_ignored + wrong_when_conflicting_ignored)
	else:
		error_when_conflicting_ignored = 0
	print('accuracy when ignoring conflicting predictions:', str(1 - error_when_conflicting_ignored))
	fraction_kept = float(not_ignored) / (ignored + not_ignored)
	print('fraction of kept:', fraction_kept)

	if ignored == 0:
		ignored = -1
	for i in range(0, len(coarse_right_in_kept_out)):
		print('Coarse' + str(i) + ' right in kept out set:', float(coarse_right_in_kept_out[i]) / ignored)
	print('Fine right in kept out set:', float(fine_right_in_kept_out) / ignored)
	print('All wrong in kept out set:', float(both_wrong_in_kept_out) / ignored)

	print('fine acc:', (correct_when_conflicting_ignored + fine_right_in_kept_out) / total)
	for i in range(len(cats)):
		print('coarse' + str(i) + ' acc:',
			  (coarse_right_when_conflicting_ignored[i] + coarse_right_in_kept_out[i]) / total)


def lr_scheduler(epoch, epochs, learning_rate, max_epoch):
	"""
	Called during fit, updates the learning rated in certain epochs.
	:param epoch: current epoch
	:param lr: current learning rate
	:return: new learning rate
	"""
	if epoch < max_epoch:
		print('epoch:', epoch, 'lr:', (epoch + 1) / max_epoch, '*', learning_rate, sep=' ')
		return learning_rate * ((epoch + 1) / max_epoch)
	else:
		print('epoch:', epoch, 'lr:', 1 - float((epoch + 1 - max_epoch)) / (epochs + 1 - max_epoch), '*', learning_rate,
			  sep=' ')
		return learning_rate * (1 - float((epoch + 1 - max_epoch)) / (epochs + 1 - max_epoch))


def generator_for_multi_outputs(generator, X, Yi, batch_size):
	"""
	ImageDataGenerator from Keras can only deal with one output. This method applies it to a list of outputs by applying
	it for each output seperately.
	:param generator: An ImageDataGenerator
	:param X: x_train
	:param Yi: list of y_train outputs
	:return:
	"""
	genYi = []
	for Y in Yi:
		genYi.append(generator.flow(X, Y, batch_size=batch_size, seed=1337))
	while True:
		Z0 = genYi[0].next()
		X0 = Z0[0]
		Yii = []
		Yii.append(Z0[1])
		for i in range(1, len(genYi)):
			Yii.append(genYi[i].next()[1])
		yield X0, Yii


def to_categorical_smooth(y, num_classes=None, to_one_hot=True, temperature=0):
	"""
	Generates sparse labels with smoothing (1-hot encoded vectors with smoothing).
	:param y: ndarray of size Nx1. Indices of class labels.
	:param num_classes: int or None. Number of classes to encode. If None, the number of unique indices is used in case
						one hot encoding is requested.
	:param to_one_hot: bool. Do a 1-hot encoding first.
	:param temperature: float between 0 and 1. The amount of smoothing to add. Starting from the 1-hot encoding,
						the temperature indicates how much from 1 is subtracted and split evenly across remaining bins.
	:return: sparse label encodings (1-hot encoded labels with smoothing)
	"""
	if to_one_hot:
		onehot = utils.to_categorical(y, num_classes)
	else:
		onehot = y
	if temperature > 0:
		k = onehot.shape[-1]
		onehot[onehot.nonzero()] -= temperature
		onehot[onehot == 0] += temperature / float(k - 1)
	return onehot


def obtain_input_shape(input_shape,
					   default_size,
					   min_size,
					   data_format,
					   require_flatten,
					   weights=None):
	"""Internal utility to compute/validate a model's input shape.

	# Arguments
		input_shape: Either None (will return the default network input shape),
			or a user-provided shape to be validated.
		default_size: Default input width/height for the model.
		min_size: Minimum input width/height accepted by the model.
		data_format: Image data format to use.
		require_flatten: Whether the model is expected to
			be linked to a classifier via a Flatten layer.
		weights: One of `None` (random initialization)
			or 'imagenet' (pre-training on ImageNet).
			If weights='imagenet' input channels must be equal to 3.

	# Returns
		An integer shape tuple (may include None entries).

	# Raises
		ValueError: In case of invalid argument values.
	"""
	if weights != 'imagenet' and input_shape and len(input_shape) == 3:
		if data_format == 'channels_first':
			if input_shape[0] not in {1, 3}:
				warnings.warn(
					'This model usually expects 1 or 3 input channels. '
					'However, it was passed an input_shape with ' +
					str(input_shape[0]) + ' input channels.')
			default_shape = (input_shape[0], default_size, default_size)
		else:
			if input_shape[-1] not in {1, 3}:
				warnings.warn(
					'This model usually expects 1 or 3 input channels. '
					'However, it was passed an input_shape with ' +
					str(input_shape[-1]) + ' input channels.')
			default_shape = (default_size, default_size, input_shape[-1])
	else:
		if data_format == 'channels_first':
			default_shape = (3, default_size, default_size)
		else:
			default_shape = (default_size, default_size, 3)
	if weights == 'imagenet' and require_flatten:
		if input_shape is not None:
			if input_shape != default_shape:
				raise ValueError('When setting `include_top=True` '
								 'and loading `imagenet` weights, '
								 '`input_shape` should be ' +
								 str(default_shape) + '.')
		return default_shape
	if input_shape:
		if data_format == 'channels_first':
			if input_shape is not None:
				if len(input_shape) != 3:
					raise ValueError(
						'`input_shape` must be a tuple of three integers.')
				if input_shape[0] != 3 and weights == 'imagenet':
					raise ValueError('The input must have 3 channels; got '
									 '`input_shape=' + str(input_shape) + '`')
				if ((input_shape[1] is not None and input_shape[1] < min_size) or
						(input_shape[2] is not None and input_shape[2] < min_size)):
					raise ValueError('Input size must be at least ' +
									 str(min_size) + 'x' + str(min_size) +
									 '; got `input_shape=' +
									 str(input_shape) + '`')
		else:
			if input_shape is not None:
				if len(input_shape) != 3:
					raise ValueError(
						'`input_shape` must be a tuple of three integers.')
				if input_shape[-1] != 3 and weights == 'imagenet':
					raise ValueError('The input must have 3 channels; got '
									 '`input_shape=' + str(input_shape) + '`')
				if ((input_shape[0] is not None and input_shape[0] < min_size) or
						(input_shape[1] is not None and input_shape[1] < min_size)):
					raise ValueError('Input size must be at least ' +
									 str(min_size) + 'x' + str(min_size) +
									 '; got `input_shape=' +
									 str(input_shape) + '`')
	else:
		if require_flatten:
			input_shape = default_shape
		else:
			if data_format == 'channels_first':
				input_shape = (3, None, None)
			else:
				input_shape = (None, None, 3)
	if require_flatten:
		if None in input_shape:
			raise ValueError('If `include_top` is True, '
							 'you should specify a static `input_shape`. '
							 'Got `input_shape=' + str(input_shape) + '`')
	return input_shape


def write_exp_id(exp):
	"""
	Writes the id of the Sacred-Experiment exp into corresponding config.json file into params dict and exp_id key.
	:param exp: Sacred-Experiment. Helps identifying experiment directory.
	:return:
	"""
	with open(os.path.join(exp.observers[0].dir, 'config.json'), 'r') as run_file:
		run_dict = json.load(run_file)
		run_dict['params']['exp_id'] = os.path.split(exp.observers[0].dir)[-1]
	with open(os.path.join(exp.observers[0].dir, 'config.json'), 'w') as run_file:
		run_file.write(json.dumps(run_dict, indent=2))


def returnCAM_keras(feature_conv, weight_softmax, class_idx):
	import cv2
	# generate the class activation maps upsample to 32x32
	size_upsample = (32, 32)
	bz, h, w, nc = feature_conv.shape
	output_cam = []
	for idx in class_idx:
		cam = weight_softmax[:, idx].dot(feature_conv.reshape((h * w, nc)).transpose())
		cam = cam.reshape(h, w)
		# print(cam)
		# print(cam)
		min = numpy.min(cam)
		cam = cam - min
		max = numpy.max(cam)
		cam_img = cam / max
		# print(cam_img)
		cam_img = numpy.clip(cam_img, 0, 1)
		cam_img = numpy.uint8(255 * cam_img)
		output_cam.append(cv2.resize(cam_img, size_upsample) / 255)
	# print(cam_img)
	return output_cam


def generate_cifar_100_labels():
	return [
		'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
		'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
		'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
		'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
		'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
		'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
		'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
		'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
		'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
		'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
		'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
		'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
		'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
		'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
		'worm'
	]
