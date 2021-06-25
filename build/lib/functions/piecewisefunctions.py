import numpy as np
from abc import ABC, abstractmethod
import numbers

class Piece(ABC):

    @abstractmethod
    def __init__(self, fromVal, fromY, t):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @property
    @abstractmethod
    def ninf(self):
        pass

    @ninf.setter
    def ninf(self, x):
        raise TypeError

    @property
    @abstractmethod
    def inf(self):
        pass

    @inf.setter
    def inf(self, x):
        raise TypeError

    @abstractmethod
    def __call__(self,x):
        pass

    @abstractmethod
    def __iadd__(self, other):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __imul__(self, scale):
        pass

    @abstractmethod
    def __mul__(self, sclae):
        pass

    @abstractmethod
    def __rmul__(self, scale):
        pass

    @abstractmethod
    def attributes(self):
        pass

    @abstractmethod
    def max(self):
        pass

class PiecewiseFunction(ABC):

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @property
    @abstractmethod
    def ninf(self):
        pass

    @ninf.setter
    def ninf(self, x):
        raise TypeError

    @property
    @abstractmethod
    def inf(self):
        pass

    @inf.setter
    def inf(self, x):
        raise TypeError

    @property
    @abstractmethod
    def piece_type(self):
        pass

    @piece_type.setter
    def piece_type(self, x):
        raise TypeError

    def __call__(self, x):
        if isinstance(x, numbers.Number):
            k = 0
            while x>self.knots[k+1]:
                k += 1
                if k == self.length-1:
                    break
            return self.pieces[k](x)
        else:
            y = np.empty_like(x)
            for k, p in zip(self.knots[:self.length], self.pieces[:self.length]):
                y[x >= k] = p(x[x >= k])
            return y

    def __iadd__(self, other: Piece):
        for i in range(self.length):
            self.pieces[i] += other
        return self
            
    @abstractmethod
    def __add__(self, other: Piece):
        pass

    def __radd__(self, other: Piece):
        return self.__add__(other)

    def __imul__(self, scale):
        for i in range(self.length):
            self.pieces[i] *= scale
        return self
            
    @abstractmethod
    def __mul__(self, scale):
        pass

    def __rmul__(self, scale):
        return self.__mul__(scale)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.length <= index:
            raise IndexError("Index is out of bound.")
        return self.pieces[index], self.knots[index]

    @abstractmethod
    def __setitem__(self, index, value):
        pass

    def max(self):
        maximum = -np.inf
        x = 0
        for i in range(self.length):
            piece_x, piece_max = self.pieces[i].max()
            if maximum < piece_max:
                maximum = piece_max
                x = piece_x
        return x, maximum
    
    @abstractmethod
    def append(self, element: Piece, knot):
        pass

    def range(self, start, stop, num):
        ii = np.logical_and( start <= self.knots, stop >= self.knots)
        x = np.linspace(start, stop, num=num)
        y = self(x)
        for i in self.knots[1:self.length]:
            if np.any(i == x) or np.all(i > x) or np.all(x > i):
                continue
            ii = np.argmin(np.abs(i - x))
            x[ii] = i
            y[ii] = self(x[ii])
        return x,y

    def plot_range(self, start, stop, num=1000, save=False, file = "function_plot.png"):
        from matplotlib import pyplot as plt
        x, y = self.range(start, stop, num)
        plt.plot(x, y)
        if save:
            plt.savefig(file)
            plt.close()
        else:
            plt.show()

