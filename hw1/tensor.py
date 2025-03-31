class Tensor:
    def __init__(self, shape, data):
        self.shape = shape
        self.data = data
        self.strides = self._calculate_strides(shape)

    def _calculate_strides(self, shape):
        strides = [1]
        for dim in reversed(shape[1:]):
            strides.insert(0, strides[0] * dim)
        return strides

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self._getitem_tuple(idx)
        elif isinstance(idx, slice):
            return self._getitem_slice(idx)
        elif isinstance(idx, int):
            return self._getitem_int(idx)
        elif isinstance(idx, list):
            return self._getitem_list(idx)
        else:
            raise TypeError("Invalid index type")

    def _getitem_tuple(self, idx):
        if len(idx) != len(self.shape):
            raise IndexError("Invalid number of indices")
        index = sum(i * s for i, s in zip(idx, self.strides))
        return self.data[index]

    def _getitem_slice(self, idx):
        start, stop, step = idx.indices(self.shape[0])
        return [self.data[i * self.strides[0]] for i in range(start, stop, step)]

    def _getitem_int(self, idx):
        if idx < 0:
            idx += self.shape[0]
        start = idx * self.strides[0]
        stop = start + self.strides[0]
        return self.data[start:stop]

    def _getitem_list(self, idx):
        return [self._getitem_int(i) for i in idx]

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"
