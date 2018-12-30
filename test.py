from typing import Iterable, Generator


def f(x: None) -> Generator[int, str]:
    for i in range(5):
        yield (i, str(i), 13.1)

