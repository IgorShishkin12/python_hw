from tensor.py import Tensor


class Matrix(Tensor):
    def __init__(self, shape, data):
        if len(shape) != 2:
            raise ValueError("Matrix must be 2-dimensional")
        super().__init__(shape, data)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            row_idx, col_idx = idx
            row_slice = self._convert_to_slice(row_idx, self.shape[0])
            col_slice = self._convert_to_slice(col_idx, self.shape[1])
            return self._getitem_slice_slice(row_slice, col_slice)
        elif isinstance(idx, slice):
            return self._getitem_slice(idx)
        elif isinstance(idx, int):
            return self._getitem_int(idx)
        elif isinstance(idx, list):
            return self._getitem_list(idx)
        else:
            raise TypeError("Invalid index type")

    def _convert_to_slice(self, idx, dim_size):
        if isinstance(idx, int):
            return slice(idx, idx + 1)
        elif isinstance(idx, list):
            return idx
        elif isinstance(idx, slice):
            return idx
        else:
            raise TypeError("Invalid index type")

    def _getitem_slice_slice(self, row_slice, col_slice):
        if isinstance(row_slice, list):
            rows = row_slice
        else:
            rows = range(*row_slice.indices(self.shape[0]))

        if isinstance(col_slice, list):
            cols = col_slice
        else:
            cols = range(*col_slice.indices(self.shape[1]))

        return [[self.data[r * self.strides[0] + c * self.strides[1]] for c in cols] for r in rows]

    def __repr__(self):
        rows = self.shape[0]
        cols = self.shape[1]
        matrix_str = "\n".join(
            " ".join(f"{self.data[r * self.strides[0] + c * self.strides[1]]:2}" for c in range(cols))
            for r in range(rows)
        )
        return f"Matrix(shape={self.shape}):\n{matrix_str}"
