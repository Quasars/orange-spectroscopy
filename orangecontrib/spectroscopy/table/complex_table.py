from copy import deepcopy
from numbers import Integral

import numpy as np
from Orange.data.table import _ArrayConversion, _FromTableConversion, _thread_local, _optimize_indices, \
    assure_domain_conversion_sparsity, _check_arrays, _compute_column
from Orange.misc.cache import IDWeakrefCache
from scipy import sparse as sp
from Orange.data import Domain, Table, Variable, DomainConversion, DiscreteVariable


class _ArrayConversionComplex(_ArrayConversion):
    def __init__(self, *args):
        super().__init__(*args)
        if self.dtype is not object:
            self.dtype = ComplexTable.DTYPE

class _FromTableConversionComplex(_FromTableConversion):
    def __init__(self, source, destination):
        conversion = DomainConversion(source, destination)

        self.X = _ArrayConversionComplex("X", conversion.attributes,
                                  destination.attributes, conversion.sparse_X,
                                  source)
        self.Y = _ArrayConversionComplex("Y", conversion.class_vars,
                                  destination.class_vars, conversion.sparse_Y,
                                  source)
        self.metas = _ArrayConversionComplex("metas", conversion.metas,
                                      destination.metas, conversion.sparse_metas,
                                      source)

        self.subarray = []
        self.columnwise = []

        for part in [self.X, self.Y, self.metas]:
            if part.subarray_from is None:
                self.columnwise.append(part)
            else:
                self.subarray.append(part)



class ComplexTable(Table):
    DTYPE = np.complex128

    @classmethod
    def from_table(cls, domain, source, row_indices=...):
        """
        Create a new table from selected columns and/or rows of an existing
        one. The columns are chosen using a domain. The domain may also include
        variables that do not appear in the source table; they are computed
        from source variables if possible.

        The resulting data may be a view or a copy of the existing data.

        :param domain: the domain for the new table
        :type domain: Orange.data.Domain
        :param source: the source table
        :type source: Orange.data.Table
        :param row_indices: indices of the rows to include
        :type row_indices: a slice or a sequence
        :return: a new table
        :rtype: Orange.data.Table
        """
        if domain is source.domain:
            table = cls.from_table_rows(source, row_indices)
            # assure resulting domain is the instance passed on input
            table.domain = domain
            # since sparse flags are not considered when checking for
            # domain equality, fix manually.
            with table.unlocked_reference():
                table = assure_domain_conversion_sparsity(table, source)
            return table

        new_cache = _thread_local.conversion_cache is None
        try:
            if new_cache:
                _thread_local.conversion_cache = IDWeakrefCache({})
                _thread_local.domain_cache = IDWeakrefCache({})
            else:
                try:
                    return _thread_local.conversion_cache[(domain, source)]
                except KeyError:
                    pass

            # avoid boolean indices; also convert to slices if possible
            row_indices = _optimize_indices(row_indices, len(source))

            self = cls()
            self.domain = domain

            try:
                table_conversion = \
                    _thread_local.domain_cache[(domain, source.domain)]
            except KeyError:
                table_conversion = _FromTableConversionComplex(source.domain, domain)
                _thread_local.domain_cache[(domain, source.domain)] = table_conversion

            # if an array can be a subarray of the input table, this needs to be done
            # on the whole table, because this avoids needless copies of contents

            with self.unlocked_reference():
                self.X, self.Y, self.metas = \
                    table_conversion.convert(source, row_indices,
                                             clear_cache_after_part=new_cache)
                self.W = source.W[row_indices]
                self.name = getattr(source, 'name', '')
                self.ids = source.ids[row_indices]
                self.attributes = getattr(source, 'attributes', {})
                if new_cache:  # only deepcopy attributes for the outermost transformation
                    self.attributes = deepcopy(self.attributes)
                _thread_local.conversion_cache[(domain, source)] = self
            return self
        finally:
            if new_cache:
                _thread_local.conversion_cache = None
                _thread_local.domain_cache = None

    @classmethod
    def from_numpy(cls, domain, X, Y=None, metas=None, W=None,
                   attributes=None, ids=None):
        """
        Construct a table from numpy arrays with the given domain. The number
        of variables in the domain must match the number of columns in the
        corresponding arrays. All arrays must have the same number of rows.
        Arrays may be of different numpy types, and may be dense or sparse.

        :param domain: the domain for the new table
        :type domain: Orange.data.Domain
        :param X: array with attribute values
        :type X: np.array
        :param Y: array with class values
        :type Y: np.array
        :param metas: array with meta attributes
        :type metas: np.array
        :param W: array with weights
        :type W: np.array
        :return:
        """
        X, Y, W = _check_arrays(X, Y, W, dtype=cls.DTYPE)
        metas, = _check_arrays(metas, dtype=object, shape_1=X.shape[0])
        ids, = _check_arrays(ids, dtype=int, shape_1=X.shape[0])

        if domain is None:
            domain = Domain.from_numpy(X, Y, metas)

        if Y is None:
            if not domain.class_vars or sp.issparse(X):
                Y = np.empty((X.shape[0], 0), dtype=cls.DTYPE)
            else:
                own_data = X.flags.owndata and X.base is None
                Y = X[:, len(domain.attributes):]
                X = X[:, :len(domain.attributes)]
                if own_data:
                    Y = Y.copy()
                    X = X.copy()
        if metas is None:
            metas = np.empty((X.shape[0], 0), object)
        if W is None or W.size == 0:
            W = np.empty((X.shape[0], 0))
        elif W.shape != (W.size, ):
            W = W.reshape(W.size).copy()

        if X.shape[1] != len(domain.attributes):
            raise ValueError(
                "Invalid number of variable columns ({} != {})".format(
                    X.shape[1], len(domain.attributes))
            )
        if Y.ndim == 1:
            if not domain.class_var:
                raise ValueError(
                    "Invalid number of class columns "
                    f"(1 != {len(domain.class_vars)})")
        elif Y.shape[1] != len(domain.class_vars):
            raise ValueError(
                "Invalid number of class columns ({} != {})".format(
                    Y.shape[1], len(domain.class_vars))
            )
        if metas.shape[1] != len(domain.metas):
            raise ValueError(
                "Invalid number of meta attribute columns ({} != {})".format(
                    metas.shape[1], len(domain.metas))
            )
        if not X.shape[0] == Y.shape[0] == metas.shape[0] == W.shape[0]:
            raise ValueError(
                "Parts of data contain different numbers of rows.")

        self = cls()
        with self.unlocked_reference():
            self.domain = domain
            self.X = X
            self.Y = Y
            self.metas = metas
            self.W = W
            self.n_rows = self.X.shape[0]
            if ids is None:
                cls._init_ids(self)
            else:
                self.ids = ids
            self.attributes = {} if attributes is None else attributes
        return self

    def get_column(self, index, copy=False):
        """
        Return a column with values of `index`.

        If `index` is an instance of variable that does not exist in the domain
        but has `compute_value`, `get_column` calls `compute_value`. Otherwise,
        it returns a view into the table unless `copy` is set to `True`.

        Args:
            index (int or str or Variable): attribute
            copy (bool): if set to True, ensure the result is a copy, not a view

        Returns:
            column (np.array): data column
        """
        if isinstance(index, Variable) and index not in self.domain:
            if index.compute_value is None:
                raise ValueError(f"variable {index.name} is not in domain")
            return _compute_column(index.compute_value, self)

        mapper = None
        if not isinstance(index, Integral):
            if isinstance(index, DiscreteVariable) \
                    and index.values != self.domain[index].values:
                mapper = index.get_mapper_from(self.domain[index])
            index = self.domain.index(index)

        col = self._get_column_view(index)
        if sp.issparse(col):
            col = col.toarray().reshape(-1)
        if col.dtype == object and self.domain[index].is_primitive():
            col = col.astype(np.float64)
        if mapper is not None:
            col = mapper(col)
        if copy and col.base is not None:
            col = col.copy()
        return col

    def _compute_contingency(self, col_vars=None, row_var=None):
        raise NotImplementedError()