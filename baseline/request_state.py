"""요청마다 별도의 상태 딕셔너리를 사용한다."""
from collections.abc import MutableMapping
from contextvars import ContextVar


class RequestState(MutableMapping):
    def __init__(self):
        self._current = ContextVar("disclosure_request", default=None)

    def _data(self):
        data = self._current.get()
        if data is None:
            data = {}
            self._current.set(data)
        return data

    def reset(self):
        self._current.set({})

    def __getitem__(self, key):
        return self._data()[key]

    def __setitem__(self, key, value):
        self._data()[key] = value

    def __delitem__(self, key):
        del self._data()[key]

    def __iter__(self):
        return iter(self._data())

    def __len__(self):
        return len(self._data())
