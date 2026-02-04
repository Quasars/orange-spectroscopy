import numbers
from collections import defaultdict
from functools import reduce

import Orange
import numpy as np
from Orange.data import FileFormat, ContinuousVariable, StringVariable, Domain
from scipy.io import matlab


class MatlabReader(FileFormat):
    EXTENSIONS = ('.mat',)
    DESCRIPTION = "Matlab"

    # Matlab 7.3+ files are not handled by scipy reader

    def read(self):
        who = matlab.whosmat(self.filename)
        if not who:
            raise IOError("Couldn't load matlab file " + self.filename)
        else:
            ml = matlab.loadmat(self.filename, chars_as_strings=True)
            ml = {a: b for a, b in ml.items() if isinstance(b, np.ndarray)}

            def num_elements(array):
                return reduce(lambda x, y: x * y, array.shape, 1)

            def find_biggest(arrays):
                sizes = []
                for n, c in arrays.items():
                    sizes.append((num_elements(c), n))
                return max(sizes)[1]

            def is_string_array(array):
                return issubclass(array.dtype.type, np.str_)

            def is_number_array(array):
                return issubclass(array.dtype.type, numbers.Number)

            numeric = {n: a for n, a in ml.items() if is_number_array(a)}

            # X is the biggest numeric array
            X = ml.pop(find_biggest(numeric)) if numeric else None

            # find an array with compatible shapes
            attributes = []
            if X is not None:
                name_array = None
                for name in sorted(ml):
                    con = ml[name]
                    if con.shape in [(X.shape[1],), (1, X.shape[1])]:
                        name_array = name
                        break
                names = ml.pop(name_array).ravel() if name_array else range(X.shape[1])
                names = [str(a).rstrip() for a in names]  # remove matlab char padding
                attributes = [ContinuousVariable.make(a) for a in names]

            meta_names = []
            metas = []

            meta_size = None
            if X is None:
                counts = defaultdict(list)
                for name, con in ml.items():
                    counts[len(con)].append(name)
                if counts:
                    meta_size = max(counts.keys(), key=lambda x: len(counts[x]))
            else:
                meta_size = len(X)
            if meta_size:
                for name, con in ml.items():
                    if len(con) == meta_size:
                        meta_names.append(name)

            meta_data = []
            for m in sorted(meta_names):
                f = ml[m]
                if is_string_array(f) and len(f.shape) == 1:  # 1D string arrays
                    metas.append(StringVariable.make(m))
                    f = np.array([a.rstrip() for a in f])  # remove matlab char padding
                    f.resize(meta_size, 1)
                    meta_data.append(f)
                elif is_number_array(f) and len(f.shape) == 2:
                    if f.shape[1] == 1:
                        names = [m]
                    else:
                        names = [m + "_" + str(i + 1) for i in range(f.shape[1])]
                    for n in names:
                        metas.append(ContinuousVariable.make(n))
                    meta_data.append(f)

            meta_data = np.hstack(tuple(meta_data)) if meta_data else None

            domain = Domain(attributes, metas=metas)
            if X is None:
                X = np.zeros((meta_size, 0))
            return Orange.data.Table.from_numpy(domain, X, Y=None, metas=meta_data)
