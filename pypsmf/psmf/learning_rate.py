# -*- coding: utf-8 -*-

import abc


class BaseLearningRate(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get(self, t):
        """ Get the learning rate for time t """


class ConstantLearningRate(BaseLearningRate):
    def __init__(self, lr):
        self.lr = lr

    def get(self, t):
        return self.lr


class ExponentialLearningRate(BaseLearningRate):
    def __init__(self, lr_start, lr_end, steps):
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.steps = steps

    def get(self, t):
        return self.lr_start * pow(self.lr_end / self.lr_start, t / self.steps)
