import numpy as np
from abc import ABC, abstractmethod
import numbers
#########DELETE THIS##########
from IPython import embed####
#############################

def flood(func, threshold):
    underwater = True
    left = right = 0
    solution_exits = False

    piece_type = type(func[0][0])
    function_type = type(func)
    
    segments = []
    segment_start = 0
    segment_end = 0
    out = function_type()

    if func[0][0](func[0][1]) < threshold:
        segment_start = out.ninf
        out.append(piece_type(t=threshold), func[0][1])
    else:
        out.append(func[0][0], func[0][1])
        underwater = False

    for i in range(len(func)-1):
        piece, knot = func[i]
        _, next_knot = func[i+1]
        left, right, left_solution_exists, right_solution_exists = piece.solve(threshold)

        if underwater and left_solution_exists and left < next_knot and left > knot:
            out.append(piece, left)
            segment_end = left
            segments.append((segment_start, segment_end))
            underwater = False
        elif i != 0 and not underwater:
            out.append(piece, knot)

        if not underwater and right_solution_exists and right < next_knot and right > knot:
            out.append(piece_type(t=threshold), right)
            segment_start = right
            underwater = True

    if underwater:
        segment_end = func.inf
        segments.append((segment_start, segment_end))

    out.append(piece_type(t=-np.inf), func.inf)
    
    return out, segments

def _check_input(n, lamb, w):

    if np.any(lamb < 0):
        raise Exception("Lambda > 0 isn't satisfied.")

    if np.any(w < 0):
        raise Exception("Weight > 0 isn't satisfied.")

    if isinstance(lamb, np.ndarray) and len(lamb) != n-1:
        raise Exception("The length of the lambda array must be one less than the length of the signal array.")

    if isinstance(lamb, numbers.Number):
        lamb = np.repeat(lamb, n)

    if isinstance(w, np.ndarray) and len(w) != n:
        raise Exception("The length of the weight array must be equal to the length of the signal array.")

    if isinstance(w, numbers.Number):
        w = np.repeat(w, n)

    return lamb, w

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
            if np.any(i == x):
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

class SquaredError(Piece):
    # a * x^2 + b * x + c
    def __init__(self, fromVal=None, fromY=None, t=None):
        if fromY is not None:
            y, w = fromY
            self._a = -w
            self._b = 2*y*w
            self._c = 0
        elif t is not None:
            self._a = 0
            self._b = 0
            self._c = t
        elif fromVal is not None: 
            a, b, c = fromVal
            self._a = a
            self._b = b
            self._c = c
        else:
            self._a = 0
            self._b = 0
            self._c = 0

    def __str__(self):
        return f"{self._a}x^2 + {self._b}x + {self._c}"

    def __repr__(self):
        return f"SquaredError(fromVal = ({self._a}, {self._b}, {self._c}))"

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, val):
        self._a = val

    @property
    def b(self):
        return self._b
    
    @b.setter
    def b(self, val):
        self._b = val

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, val):
        self._c = val

    @property
    def ninf(self):
        return -np.inf

    @property
    def inf(self):
        return np.inf

    def __call__(self,x):
        return (self._a * x + self._b) * x + self._c

    def __iadd__(self, other):
        self._a += other._a
        self._b += other._b
        self._c += other._c
        return self

    def __add__(self, other):
        if isinstance(other, PiecewiseSquaredError):
            return other.__add__(self)
        _a = self._a + other._a
        _b = self._b + other._b
        _c = self._c + other._c
        return SquaredError(fromVal=(_a,_b,_c))

    def __imul__(self, scale):
        self._a *= scale
        self._b *= scale
        self._c *= scale
        return self

    def __mul__(self, scale):
        _a = self._a * scale
        _b = self._b * scale
        _c = self._c * scale
        return SquaredError(fromVal=(_a,_b,_c))

    def __rmul__(self, scale):
        return self.__mul__(scale)

    def attributes(self):
        return (self._a,self._b,self._c)

    def max(self):
        if self._a == 0:
            if self._b > 0:
                return np.inf, np.inf
            if self._b < 0:
                return -np.inf, np.inf
            else:
                return -np.inf, self._c
        return -self._b / 2 / self._a, self(-self._b / 2 / self._a)

    def solve(self, t):
        c = self._c - t
        lsolution_exists = rsolution_exists = True
        delta = np.power(self._b, 2) - 4 * self._a * c
        if delta == 0:
            lsolution = rsolution = -self._b / 2 / self._a
        elif delta < 0:
            lsolution = rsolution = np.nan
            lsolution_exists = rsolution_exists = False
        else:
            lsolution = (-self._b + np.sqrt(delta)) / 2 / self._a
            rsolution = (-self._b - np.sqrt(delta)) / 2 / self._a
            if lsolution > rsolution:
                tmp = lsolution
                lsolution = rsolution
                rsolution = tmp
        return lsolution, rsolution, lsolution_exists, rsolution_exists

class PiecewiseSquaredError(PiecewiseFunction):

    def __init__(self, piece_list=None, knot_list=None):
        if piece_list is None and knot_list is None:
            self.pieces = np.empty(10,dtype=SquaredError)
            self.knots = np.empty(10,dtype=float)
            self.length = 0
        elif piece_list is not None and knot_list is not None:
            ii = np.argsort(knot_list)
            self.pieces = np.array(piece_list)[ii]
            self.knots = np.array(knot_list)[ii]
            self.length = len(self.pieces)
        else:
            raise "Both knot list and function list are needed."

    def __str__(self):
        if self.length == 1:
            string =  str(self.pieces[0])
        elif self.length == 2:
            string = f"{{x < {self.knots[1]}: {self.pieces[0]} , {self.knots[1]} <= x: {self.pieces[1]}}}"
        else:
            string = f"{{x < {self.knots[1]}: {self.pieces[0]}, "
            for i in range(1,self.length-1):
                string += f"{self.knots[i]} <= x < {self.knots[i+1]}: {self.pieces[i]}, "
            string += f"{self.knots[self.length-1]} <= x: {self.pieces[self.length-1]}}}"
        return string

    def __repr__(self):
        if self.length == 0:
            return "PiecewiseSquaredError()"
        string_pieces = "np.array(["
        for i in self.pieces[:self.length-1]:
            string_pieces += f"{repr(i)}, "
        string_pieces += f"{repr(self.pieces[self.length-1])}])"
        string_knots = "np.array([-np.inf, "
        for i in self.knots[1:self.length-1]:
            string_knots += f"{repr(i)}, "
        string_knots += f"{repr(self.knots[self.length-1])}])"
        return f"PiecewiseSquaredError(piece_list = {string_pieces}, knot_list = {string_knots})"

    @property
    def ninf(self):
        return -np.inf

    @property
    def inf(self):
        return np.inf

    @property
    def piece_type(self):
        return SquaredError

    def __add__(self, other: Piece):
        piece_list = [None] * self.length
        for i in range(self.length):
            piece_list[i] = self.pieces[i] + other
        return PiecewiseSquaredError(piece_list, self.knots[:self.length])

    def __mul__(self, scale):
        piece_list = [None] * self.length
        for i in range(self.length):
            piece_list[i] = self.pieces[i] * scale
        return PiecewiseSquaredError(pieces_list, self.knots[:self.length])

    def __setitem__(self, index, value):
        if self.length <= index:
            raise Exception("Index is out of bound.")
        piece, knot = value
        if not isinstance(piece, SquaredError):
            raise TypeError
        self.pieces[index] = piece
        self.knots[index] = knot

    def append(self, piece, knot):
        if not isinstance(piece,SquaredError):
            raise Exception("Piece to be added needs to be a Squared type.")

        if self.length == len(self.pieces):
            tmp = self.pieces
            self.pieces = np.empty(self.length*2, dtype=SquaredError)
            self.pieces[:self.length] = tmp
            tmp = self.knots
            self.knots = np.empty(self.length*2, dtype=SquaredError)
            self.knots[:self.length] = tmp

        self.pieces[self.length] = piece
        self.knots[self.length] = knot
        self.length += 1
        ii = np.argsort(self.knots[:self.length])
        self.pieces[:self.length] = self.pieces[:self.length][ii]
        self.knots[:self.length] = self.knots[:self.length][ii]

def backtrace(segments, xprimes):
    solution = np.empty_like(xprimes)
    n = len(solution)
    solution[n-1] = xprimes[n-1]
    for i in reversed(range(n-1)):
        solution[i] = solution[i+1]
        for s, e in segments[i]:
            if s <= solution[i+1] and solution[i+1] <= e:
                solution[i] = xprimes[i]
                break
    return solution

def approximator(y, lamb, w=1, function_type=PiecewiseSquaredError):

    n = len(y)
    lamb, w = _check_input(n, lamb, w)
    func1 = function_type()
    func2 = function_type()
    piece_type = func1.piece_type

    func1.append(piece_type(fromY = (y[0], w[0])), func1.ninf)
    func1.append(piece_type(t = -np.inf), func1.inf)

    segments = [None] * (n-1)
    xprimes = np.empty(n)

    for i in range(1,len(y)):
        xprimes[i-1], yprime = func1.max()
        func2, segments[i-1] = flood(func1, yprime - lamb[i])
        func1 = func2 + piece_type(fromY = (y[i], w[i]))

    xprimes[n-1], yprime = func1.max()

    return backtrace(segments, xprimes)

def backtraceN(segments, xprimes):
    n = xprimes.shape[0]
    T = xprimes.shape[1]-1
    solution = np.empty(xprimes.shape[0])
    solution[n-1] = xprimes[n-1,T]
    for i in reversed(range(n-1)):
        solution[i] = solution[i+1]
        for s, e in segments[i,T]:
            if s <= solution[i+1] and solution[i+1] <= e:
                solution[i] = xprimes[i+1,T]
                T -= 1
                break
    return solution

def approximateN(y, N=2, w=1, function_type=PiecewiseSquaredError):

    n = len(y)
    _, w = _check_input(n, 1, w)
    piece_type = function_type().piece_type
    func1s = np.empty(N, dtype=function_type)
    func2s = np.empty(N, dtype=function_type)
    xprimes = np.empty((n, N), dtype=float)
    segments = np.empty((n-1, N), dtype=list)

    ninf = piece_type().ninf
    inf = piece_type().inf

    for i in range(N):
        func1s[i] = function_type(piece_list = [piece_type(fromY=(y[0], w[0])), piece_type(t=-np.inf)], knot_list=[ninf, inf])

    for i in range(1, n):
        for j in reversed(range(N)):

            yprime = -np.inf

            if j <= i and j != 0:
                xprimes[i-1,j], yprime = func1s[j-1].max()

            if j == i:
                func1s[j] = function_type(piece_list=[piece_type(t=yprime), piece_type(t=-np.inf)], knot_list=[ninf, inf])
                func2s[j] = function_type(piece_list=[piece_type(t=yprime), piece_type(t=-np.inf)], knot_list=[ninf, inf])
            else:
                func2s[j], segments[i-1,j] = flood(func1s[j], yprime)

            func1s[j] = func2s[j] + piece_type(fromY=(y[i], w[i]))

    xprimes[n-1, N-1], yprime = func1s[N-1].max()

    embed()
    return backtraceN(segments, xprimes)

#def divide_in_two(y, w=1, function_type = PiecewiseSquaredError):
#    
#    n = len(y)
#    _, w = _check_input(n, 1, w)
#    piece_type = func1.piece_type
#    first_level = []
#    for i in range(len(y)):
#        if i != 0:
#            first_level.append(first_level[i-1] + piece_type(fromY = (y[i], w[i])))
#        else:
#            first_level.append(piece_type(fromY = (y[i], w[i])))
#    func1s = [function_type()] * n
#    func2s = [function_type()] * n

pieces = [SquaredError(fromVal=(-1,-3,1)), SquaredError(fromVal=(-1,2,5)), SquaredError(fromVal=(-1,8,-15)), SquaredError(fromVal=(0,0,-np.inf))]
knots = [-np.inf,-0.8,10/3,np.inf]
a = PiecewiseSquaredError(piece_list=pieces,knot_list=knots)
y = np.load("y.npy")
#y = np.r_[np.random.rand(10), np.random.rand(15) + 15, np.random.rand(5) + 7]
embed()
