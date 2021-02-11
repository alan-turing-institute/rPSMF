# -*- coding: utf-8 -*-

"""Builtin nonlinearities for PSMF

To define your own nonlinearity, simply inherit from BaseNonLinearity. These 
are mainly added for convenience.

"""

import abc
import autograd.numpy as np


class BaseNonLinearity(metaclass=abc.ABCMeta):
    def __init__(self, rank):
        self.rank = rank

    @property
    def dimensions(self):
        r = self.rank
        return [eval(s, {"r": r}) for s in self.dims]

    @property
    def n_params(self):
        return sum(self.dimensions)

    def unpack_theta(self, theta):
        theta = theta.squeeze()
        start = 0
        blocks = []
        for dim in self.dimensions:
            blocks.append(theta[start : start + dim])
            start += dim
        return blocks

    @abc.abstractmethod
    def __call__(self, theta, x, t):
        """To be implemented by child classes"""
        pass


class RandomWalk(BaseNonLinearity):
    """Random Walk

    This corresponds to the behavior where f(x_t) = x_t. It is technically not
    a nonlinearity, but a linear function. It takes no parameters.

    """

    dims = []

    def __init__(self):
        super().__init__(0)

    def __call__(self, theta, x, t):
        return x


class ScaledWalk(BaseNonLinearity):
    """Scaled Random Walk

    Implements the function f(x_t) = A x_t + b, where A is an r x r matrix and
    b and r x 1 vector. Using a bias vector is optional.
    """

    dims = ["r*r"]

    def __init__(self, rank, bias=True):
        super().__init__(rank)
        self.bias = bias

    def __call__(self, theta, x, t):
        r = self.rank
        A, b = self.unpack_theta(theta)
        A = A.reshape(r, r)
        if self.bias:
            return A @ x + b
        return A @ x


class Sinusoid(BaseNonLinearity):
    """Sinusoid nonlinearity

    Implements the function f(x_t) = A sin(2 * pi * b * t + c * x_t), where A
    is an r x r matrix, and b and c are r x 1 vectors.

    """

    dims = ["r*r", "r", "r"]

    def __init__(self, rank, scaled=True, phased=True):
        super().__init__(rank)
        self.scaled = scaled
        self.phased = phased
        if not scaled:
            self.dims.pop(0)
        if not phased:
            self.dims.pop(-1)

    def __call__(self, theta, x, t):
        r = self.rank
        if self.scaled and self.phased:
            A, b, c = self.unpack_theta(theta)
            A = A.reshape(r, r)
            return A @ np.sin(2 * np.pi * b * t + c * x)
        elif self.scaled:
            A, b = self.unpack_theta(theta)
            A = A.reshape(r, r)
            return A @ np.sin(2 * np.pi * b * t + x)
        elif self.phased:
            b, c = self.unpack_theta(theta)
            return np.sin(2 * np.pi * b * t + c * x)
        b = theta
        return np.sin(2 * np.pi * b * t + x)


class FourierBasis(BaseNonLinearity):
    """Fourier basis nonlinearity

    Implements a weighted Fourier series nonlinearity:

        f(x_t) = sum_{n=1}^N A_n sin(2 * pi * b_n * t + c_n * x_t) +
                             D_n cos(2 * pi * e_n * t + f_n * x_t)

    where the number of terms N can be specified.
    """

    def __init__(self, rank, N=1):
        super().__init__(rank)
        self.N = N
        self.dims = ["r*r"] * 2 * N + ["r"] * 4 * N

    def __call__(self, theta, x, t):
        r = self.rank
        N = self.N
        params = self.unpack_theta(theta)
        out = 0
        for n in range(N):
            A = params[2 * n]
            A = A.reshape(r, r)
            b = params[2 * N + 4 * n]
            c = params[2 * N + 4 * n + 1]
            out += A @ np.sin(2 * np.pi * b * t + c * x)

            D = params[2 * n + 1]
            D = D.reshape(r, r)
            e = params[2 * N + 4 * n + 2]
            f = params[2 * N + 4 * n + 3]
            out += D @ np.cos(2 * np.pi * e * t + f * x)
        return out
