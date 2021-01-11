from __future__ import print_function, division, unicode_literals

import logging
import os
import os.path as pt
from collections import defaultdict
import time
from datetime import timedelta, datetime

import torch

from logger import make_printer, print


class AverageValue(object):
	def __init__(self):
		self.total = 0
		self.steps = 0

	def value(self):
		return self.total / self.steps

	def __call__(self, value):
		self.steps += 1
		self.total += value
		return self.value()


def nicetime(seconds):
	"""
	formats given time in form of seconds to a nice string representation.

	:return: string representation of time
	"""

	hours = seconds // 3600
	minutes = seconds % 3600 // 60
	seconds = seconds % 60
	if hours:
		return '%d:%02d:%02d hours' % (hours, minutes, seconds)
	elif minutes:
		return '%d:%02d minutes' % (minutes, seconds)
	else:
		return '%d seconds' % seconds


class RemainingTimer(object):
	"""
	Estimates remaining time of some operation.
	Each time it is called, an internal counter is increased by 1.
	The measured operation is assumed to be done once this counter reaches goal.
	Out of the frequency of the calls one can compute an estimated speed and thus remaining time.

	:param goal: the number of calls until the operation is assumed to be done.
	"""

	def __init__(self, goal):
		self.start = time.time()
		self.i = 0
		self.goal = goal

	def _rem(self):
		now = time.time()
		n = self.goal
		self.i = self.i
		self.i += 1
		delta = now - self.start
		speed = delta / self.i
		return speed * (n - self.i)

	def __call__(self):
		return nicetime(self._rem())


class ETATimer(RemainingTimer):
	"""
	Simple modification of the RemainingTimer.
	When called, instead of returning remainin time, return ETA (estimated time of arrival).
	"""

	def __call__(self):
		now = datetime.now()
		delta = timedelta(seconds=self._rem())
		done = now + delta
		return done.replace(microsecond=0)


class Trainer(object):
	"""
	The Trainer can be used to train a given network.
	It alternately trains one epoch and validates
	the resulting net one epoch.
	Given loss is evaluated each batch,
	gradients are computed and optimizer used to updated weights.
	The loss is also passed to the policy,
	which might update the learning rate.
	Useful information about the training
	flow is regularly printed to the console,
	including an estimated time of arrival.

	:param network:
		Some network that is to be trained.
		If multiple gpus are used (i.e. multiple devices passed to the data loader)
		a ParallelApply module has to be wrapped around.
	:param optimizer:
		some torch optimzer, e.g. SGD or ADAM, given the network's parameters.
	:param loss:
		some loss function, e.g. CEL or MSE. Make sure to use crumpets.torch.loss
		or implement your own ones, but do not use torch losses directly, since
		they are not capable of handling crumpets sample style (i.e dictionaries).
	:param metric:
		some metric to further measure network's quality.
		Similar to losses, use crumpets.torch.metrics
	:param train_policy:
		some policy to maintain learning rates and such,
		in torch usually called lr_schedulers.
		After each iteration it, given the current loss,
		updates learning rates and potentially other hyperparameters.
	:param val_policy:
		same as train_policy, but updates after validation epoch.
	:param train_iter:
		iterator for receiving training samples,
		usually this means a :class:`~TorchTurboDataLoader` instance.
	:param val_iter:
		same as train_iter, but for retrieving validation samples.
	:param outdir:
		Output directory for models.
	:param val_loss:
		same as loss, but applied during validation.
		Default is None, which results in using loss again for validation.
	:param val_metric:
		same as metric, but applied during validation.
		Default is None, which results in using metric again for validation.
	:param snapshot_interval:
		Number of epochs between snapshots.
		Set to 0 or `None` to disable snapshots.
		Default is 1, which means taking a snapshot after every epoch.
	:param quiet:
		If True, trainer will not print to console and will not attempt
		to create a logfile.
	"""

	def __init__(
			self,
			network,
			optimizer,
			loss,
			metric,
			train_policy,
			train_iter,
			val_iter,
			outdir,
			val_loss=None,
			val_metric=None,
			snapshot_interval=1,
			quiet=False,
			device='cuda'
	):
		self.state = {
			'epoch': 0,
			'network': network,
			'optimizer': optimizer,
			'train_policy': train_policy,
			'loss': loss,
			'metric': metric,
			'train_iter': train_iter,
			'val_iter': val_iter,
			'train_metric_values': [],
			'val_metric_values': [],
			'val_loss': val_loss,
			'val_metric': val_metric,
			'snapshot_interval': snapshot_interval,
			'quiet': quiet,
			'outdir': outdir
		}
		self.hooks = defaultdict(list)
		self.device = device

	def add_hook(self, name, fun):
		"""
		Add a function hook for the given event.
		Function must accept trainer `state` dictionary as first
		positional argument the current, as well as further keyword
		arguments depending on the type of hook.

		The following events are available during training:

		- `'train_begin'`: run at the beginning of a training epoch
		- `'train_end'`: run after a training epoch has ended
		- `'train_pre_forward'`: run before the forward step;
		  receives kwarg `sample`
		- `'train_forward'`: run after the forward step;
		  receives kwargs `metric`, `loss`, and `output`
		- `'train_backward'`: run after the backward step;
		  receives kwargs `metric`, `loss`, and `output`

		During validation the following hooks are available:

		- `'val_begin'`: run at the beginning of a training epoch
		- `'val_end'`: run after a training epoch has ended
		- `'val_pre_forward'`: run before the forward step;
		  receives kwarg `sample`
		- `'val_forward'`: run after the forward step;
		  receives kwargs `metric`, `loss`, and `output`

		:param name:
			The event name.
			See above for available hook names and when they are executed.
		:param fun:
			A function that is to be invoked when given event occurs.
			See above for method signature.
		"""
		self.hooks[name].append(fun)

	def remove_hook(self, name, fun):
		"""
		Remove the function hook with the given name.

		:param name:
			type of hook to remove
		:param fun:
			hook function object to remove
		:return:
		"""
		self.hooks[name].remove(fun)

	def _run_hooks(self, name, *args, **kwargs):
		"""
		invokes functions hooked to event ``name`` with parameters *args and **kwargs.
		"""
		for fun in self.hooks[name]:
			fun(self.state, *args, **kwargs)

	def train(self, num_epochs, start_epoch=0):
		"""
		starts the training, logs loss and metrics in logging file and prints progress
		in the console, including an ETA. Also stores snapshots of current model each epoch.

		:param num_epochs:
			number of epochs to train
		:param start_epoch:
			the first epoch, default to 0.
			Can be set higher for finetuning, etc.
		"""
		try:
			rem = ETATimer(num_epochs - start_epoch)
			for epoch in range(start_epoch + 1, num_epochs + 1):
				self.state['epoch'] = epoch
				if not self.state['quiet']:
					print('Epoch', epoch)
				self.print_info()
				self.train_epoch(epoch)
				if self.state['val_iter'] is not None:
					self.validate_epoch()
				self.snapshot(epoch)
				if not self.state['quiet']:
					print('ETA:', rem())
			return self.state
		except Exception as e:
			logging.exception(e)
			raise

	def _param_groups(self):
		return self.state['optimizer'].param_groups

	def _lrs(self):
		return [g['lr'] for g in self._param_groups()]

	def print_info(self):
		"""
		prints and logs current learning rates as well as the epoch.

		:param epoch: the current epoch.
		"""
		if not self.state['quiet']:
			s = 'learning rates ' + (', '.join(map(str, self._lrs())))
			print(s)

	def snapshot(self, epoch):
		"""
		stores snapshot of current model (including optimizer state),
		uses epoch for naming convention (but does always store current model).

		:param epoch: epoch for naming output file
		"""
		interval = self.state['snapshot_interval']
		if interval is not None and interval > 0 and epoch % interval == 0:
			path = pt.join(self.state['outdir'], 'epoch_%02d.pth' % epoch)
			torch.save(dict(model_state=self.state['network'].state_dict(),
							optimizer_state=self.state['optimizer'].state_dict(), epoch=epoch), path
					   )

	def train_epoch(self, epoch):
		"""
		trains one epoch, is invoked by train function. Usually not necessary to be called outside.

		:return: train metric result
		"""
		network = self.state['network'].train()
		optimizer = self.state['optimizer']
		loss = self.state['loss']
		loss_metric = AverageValue()
		metric = self.state['metric']
		metric.reset()
		policy = self.state['train_policy']
		n = len(self.state['train_iter'])
		printer = make_printer(desc='TRAIN', total=n,
							   disable=self.state['quiet'])
		train_metric = dict()
		self._run_hooks('train_begin')
		iteration_in_epoch = 0
		for x, y in self.state['train_iter']:
			sample = {'image': x, 'label': y}
			iteration = epoch * n + iteration_in_epoch
			iteration_in_epoch += 1
			optimizer.zero_grad()
			sample['image'] = sample['image'].to(self.device)
			sample['label'] = sample['label'].to(self.device)
			self._run_hooks('train_pre_forward',
							sample=sample)
			output = network.forward(sample)
			l = loss(output)
			# _show([sample[0][0]['image']], ms=0)
			train_metric.update(
				metric(output),
				loss=loss_metric(l).item()
			)
			"""if m > 1:
				l /= m"""
			self._run_hooks('train_forward',
							metric=train_metric, loss=l, output=output)
			l.backward()
			self._run_hooks('train_backward',
							metric=train_metric, loss=l, output=output)
			policy.step(iteration / n, train_metric['loss'])
			optimizer.step()
			printer(**train_metric)
		self.state['train_metric_values'].append(train_metric)
		self._run_hooks('train_end')
		return train_metric

	def validate_epoch(self):
		"""
		Validate once.
		Invoked by train function.
		Usually not necessary to be called outside.

		:return: val metric result
		"""
		network = self.state['network'].eval()
		loss = self.state['val_loss'] or self.state['loss']
		loss_metric = AverageValue()
		metric = self.state['val_metric'] or self.state['metric']
		metric.reset()
		n = len(self.state['val_iter'])
		printer = make_printer(desc='VAL', total=n,
							   disable=self.state['quiet'])
		val_metric = dict()

		self._run_hooks('val_begin')
		for x, y in self.state['val_iter']:
			sample = {'image': x, 'label': y}
			sample['image'] = sample['image'].to(self.device)
			sample['label'] = sample['label'].to(self.device)

			self._run_hooks('val_pre_forward',
							sample=sample)
			with torch.no_grad():
				output = network.forward(sample)
				l = loss(output)
			val_metric.update(
				metric(output),
				loss=loss_metric(l).item(),
			)
			self._run_hooks('val_forward',
							metric=val_metric, loss=l, output=output)
			printer(**val_metric)
		self.state['val_metric_values'].append(val_metric)
		self._run_hooks('val_end')
		return val_metric
