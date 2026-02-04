import numpy as np


class Function:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x):
        return self.fn(x)

    def __eq__(self, other):
        return type(self) is type(other) and self.fn == other.fn

    def __hash__(self):
        return hash((type(self), self.fn))


class Constant(Function):
    def __init__(self, c):
        super().__init__(None)
        self.c = c

    def __call__(self, x):
        x = np.asarray(x)
        return np.ones(x.shape) * self.c

    def __eq__(self, other):
        return super().__eq__(other) and self.c == other.c

    def __hash__(self):
        return hash((super().__hash__(), self.c))


class Identity(Function):
    def __init__(self):
        super().__init__(None)

    def __call__(self, x):
        return x


class Segments(Function):
    """
    Each segment if defined by a condition and a function.

    Where condition holds, function is evaluated and its result is stored.
    Segments are evaluated is sequence so a later one can rewrite the previous.
    """

    def __init__(self, *segments):
        super().__init__(None)
        self.segments = segments

    def __call__(self, x):
        x = np.asarray(x)
        output = np.full(x.shape, np.nan)
        for cond, fn in self.segments:
            ind = cond(x)
            output[ind] = fn(x[ind])
        return output

    def __eq__(self, other):
        return super().__eq__(other) and self.segments == other.segments

    def __hash__(self):
        return hash((super().__hash__(), self.segments))


class Sum(Function):
    def __init__(self, *elements):
        super().__init__(None)
        self.elements = elements

    def __call__(self, x):
        acc = None
        for el in self.elements:
            current = el(x)
            if acc is None:
                acc = current
            else:
                acc = acc + current
        return acc

    def __eq__(self, other):
        return super().__eq__(other) and self.segments == other.elements

    def __hash__(self):
        return hash((super().__hash__(), self.elements))
