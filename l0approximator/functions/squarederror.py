from .piecewisefunctions import Piece, PiecewiseFunction
import numpy as np

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


