import numpy as np
import numbers
from .functions.squarederror import PiecewiseSquaredError, SquaredError

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

def approximate(y, lamb, w=1, function_type=PiecewiseSquaredError):

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
            if s <= solution[i] and solution[i] <= e:
                solution[i] = xprimes[i,T]
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
#
    for i in range(1, n):
        for j in reversed(range(N)):

            yprime = -np.inf

            if j <= i and j != 0:
                xprimes[i-1,j], yprime = func1s[j-1].max()

            if j == i:
                func2s[j] = function_type(piece_list=[piece_type(t=yprime), piece_type(t=-np.inf)], knot_list=[ninf, inf])
            else:
                func2s[j], segments[i-1,j] = flood(func1s[j], yprime)

            func1s[j] = func2s[j] + piece_type(fromY=(y[i], w[i]))

    xprimes[n-1, N-1], yprime = func1s[N-1].max()

    return backtraceN(segments, xprimes)
