"""
    A helper class to load and index Lassy analyses.
"""

from xml.etree.cElementTree import ElementTree, parse
from os import path, walk
from glob import glob
from typing import overload, Iterator
from builtins import slice


class Lassy:
    def __init__(self, treebank_dir: str) -> None:
        if path.isdir(treebank_dir):
            self.treebank_dir = treebank_dir
        else:
            raise NotADirectoryError(f'{treebank_dir} is not a directory')

        self.filelist = [y for x in walk(self.treebank_dir) for y in glob(path.join(x[0], '*.xml'))
                         if not y.split('/')[-1] in IGNORED_FILES]
        print(f'Found {len(self)} files.')

    def __len__(self) -> int:
        return len(self.filelist)

    @overload
    def __getitem__(self, item: int | str | None) -> tuple[ElementTree, str] | None: ...

    @overload
    def __getitem__(self, item: slice) -> list[tuple[ElementTree, str]]: ...

    def __getitem__(self, item):
        if isinstance(item, str):
            return self[next((i for i in range(len(self.filelist)) if self.filelist[i].endswith(item)), None)]
        if isinstance(item, int):
            return parse(name := self.filelist[item]), name.split('/')[-1],
        if isinstance(item, slice):
            return [self[i] for i in range(*item.indices(len(self)))]
        if item is None:
            return
        raise TypeError(f'{item} is not an integer or slice')

    def __iter__(self) -> Iterator[tuple[ElementTree, str]]:
        return (self.__getitem__(i) for i in range(len(self)))

    def __index__(self):
        return self.filelist.index


IGNORED_FILES = {
    'WR-P-E-H-0000000050.p.3.s.1.xml',
    'WR-P-E-H-0000000036.p.3.s.1.xml',
    'WS-U-E-A-0000000027.p.3.s.1.xml',
    'WR-P-E-H-0000000052.p.3.s.1.xml',
    'wiki-356.p.22.s.1.xml',
    'wiki-356.p.20.s.1.xml',
    'WR-P-P-C-0000000049.txt-243.xml',
    'WS-U-E-A-0000000234.p.17.s.7.xml',
    'WR-P-E-H-0000000047.p.3.s.1.xml',
    'WS-U-E-A-0000000218.p.17.s.8.xml',
    'WS-U-E-A-0000000212.p.12.s.3.xml',
    'WR-P-E-H-0000000027.p.3.s.1.xml',
    'WS-U-E-A-0000000023.p.25.s.6.xml',
    'WR-P-P-I-0000000065.p.1.s.2.xml',
    'WR-P-P-I-0000000063.p.1.s.2.xml',
    'WR-P-P-I-0000000064.p.1.s.2.xml',
    'WS-U-E-A-0000000217.p.36.s.1.xml',
    'WR-P-E-H-0000000049.p.3.s.1.xml',
    'wiki-889.p.16.s.2.xml',
    'wiki-889.p.14.s.2.xml',
    'wiki-889.p.18.s.2.xml',
    'WR-P-P-I-0000000051.p.1.s.2.xml',
    'WR-P-P-I-0000000059.p.1.s.3.xml',
    'WR-P-P-C-0000000047.txt-52.xml',
    'WR-P-P-I-0000000049.p.1.s.2.xml',
    'WS-U-E-A-0000000222.p.28.s.11.xml',
    'WR-P-P-I-0000000057.p.1.s.2.xml',
    'WR-P-E-I-0000016944.p.3.s.288.xml',
    'WR-P-E-H-0000000013.p.3.s.1.xml',
    'WR-P-P-I-0000000068.p.1.s.2.xml',
    'WR-P-E-H-0000000020.p.3.s.1.xml',
    'WR-P-P-I-0000000050.p.1.s.2.xml',
    'wiki-135.p.16.s.1.xml',
    'WS-U-E-A-0000000045.p.1.s.2.xml',
    'WS-U-E-A-0000000045.p.2.s.2.xml',
    'WR-P-E-H-0000000040.p.3.s.1.xml',
    'WS-U-E-A-0000000225.p.34.s.2.xml',
    'WR-P-P-C-0000000062.p.167.s.2.xml',
    'WR-P-P-C-0000000062.p.39.s.1.xml',
    'WR-P-E-H-0000000055.p.3.s.1.xml',
    'WS-U-E-A-0000000031.p.1.s.2.xml',
    'WS-U-E-A-0000000031.p.2.s.2.xml',
    'WR-P-P-I-0000000062.p.1.s.2.xml',
    'WR-P-E-C-0000000036.p.32.s.1.xml',
    'WR-P-E-C-0000000036.p.58.s.1.xml',
    'WR-P-E-C-0000000036.p.41.s.1.xml',
    'WR-P-E-C-0000000036.p.97.s.1.xml',
    'WR-P-E-C-0000000036.p.101.s.1.xml',
    'WR-P-E-C-0000000036.p.147.s.1.xml',
    'WR-P-E-C-0000000036.p.18.s.1.xml',
    'WR-P-E-C-0000000036.p.65.s.1.xml',
    'WR-P-E-C-0000000036.p.100.s.1.xml',
    'WR-P-E-C-0000000036.p.145.s.1.xml',
    'WR-P-E-C-0000000036.p.74.s.1.xml',
    'WR-P-E-C-0000000036.p.106.s.1.xml',
    'WR-P-E-C-0000000036.p.44.s.1.xml',
    'WR-P-E-C-0000000036.p.56.s.1.xml',
    'WR-P-E-C-0000000036.p.120.s.1.xml',
    'WR-P-E-C-0000000036.p.112.s.1.xml',
    'WR-P-E-C-0000000036.p.109.s.1.xml',
    'WR-P-E-C-0000000036.p.77.s.1.xml',
    'WR-P-E-C-0000000036.p.68.s.1.xml',
    'WR-P-E-C-0000000036.p.99.s.1.xml',
    'WR-P-E-C-0000000036.p.35.s.1.xml',
    'WR-P-E-C-0000000036.p.48.s.1.xml',
    'WR-P-E-C-0000000036.p.88.s.1.xml',
    'WR-P-E-C-0000000036.p.38.s.1.xml',
    'WR-P-E-C-0000000036.p.29.s.1.xml',
    'WR-P-E-C-0000000036.p.95.s.1.xml',
    'WR-P-E-C-0000000036.p.139.s.1.xml',
    'WR-P-E-C-0000000036.p.23.s.1.xml',
    'WR-P-E-C-0000000036.p.87.s.1.xml',
    'WR-P-E-C-0000000036.p.60.s.1.xml',
    'WR-P-E-C-0000000036.p.96.s.1.xml',
    'WR-P-E-C-0000000036.p.144.s.1.xml',
    'WR-P-E-C-0000000036.p.59.s.1.xml',
    'WR-P-E-C-0000000036.p.146.s.1.xml',
    'WR-P-E-C-0000000036.p.21.s.1.xml',
    'WR-P-E-C-0000000036.p.103.s.1.xml',
    'WR-P-E-C-0000000036.p.19.s.1.xml',
    'WR-P-E-C-0000000036.p.57.s.1.xml',
    'WR-P-E-C-0000000036.p.80.s.1.xml',
    'WR-P-E-C-0000000036.p.98.s.1.xml',
    'WR-P-E-C-0000000036.p.24.s.1.xml',
    'WR-P-E-C-0000000036.p.85.s.1.xml',
    'WR-P-E-C-0000000036.p.115.s.1.xml',
    'WR-P-E-C-0000000036.p.61.s.1.xml',
    'WR-P-E-C-0000000036.p.22.s.1.xml',
    'WR-P-E-C-0000000036.p.49.s.1.xml',
    'WR-P-E-C-0000000036.p.46.s.1.xml',
    'WR-P-E-C-0000000036.p.20.s.1.xml',
    'WR-P-E-C-0000000036.p.63.s.1.xml',
    'WR-P-E-C-0000000036.p.143.s.1.xml',
    'WR-P-E-C-0000000036.p.26.s.1.xml',
    'WR-P-E-C-0000000036.p.71.s.1.xml',
    'WR-P-E-C-0000000036.p.83.s.1.xml',
    'WR-P-E-C-0000000036.p.118.s.1.xml',
    'WR-P-E-C-0000000036.p.62.s.1.xml',
    'WR-P-P-I-0000000056.p.1.s.2.xml',
    'WR-P-E-H-0000000009.p.3.s.1.xml',
    'WR-P-P-I-0000000053.p.1.s.2.xml',
    'wiki-5716.p.2.s.3.xml',
    'WR-P-E-H-0000000051.p.3.s.1.xml',
    'WR-P-P-I-0000000052.p.1.s.2.xml',
    'WR-P-P-I-0000000060.p.1.s.2.xml',
    'WR-P-P-I-0000000054.p.1.s.2.xml',
    'WR-P-E-C-0000000021.p.53.s.1.xml',
    'WR-P-E-C-0000000021.p.50.s.1.xml',
    'WR-P-E-C-0000000021.p.67.s.1.xml',
    'WR-P-E-C-0000000021.p.76.s.1.xml',
    'WR-P-E-C-0000000021.p.56.s.1.xml',
    'WR-P-E-C-0000000021.p.75.s.1.xml',
    'WR-P-E-C-0000000021.p.72.s.1.xml',
    'WR-P-E-C-0000000021.p.59.s.1.xml',
    'WR-P-E-C-0000000021.p.65.s.1.xml',
    'WR-P-E-C-0000000021.p.62.s.1.xml',
    'WR-P-E-C-0000000021.p.61.s.1.xml',
    'WR-P-E-C-0000000021.p.74.s.1.xml',
    'WR-P-E-C-0000000021.p.52.s.1.xml',
    'WR-P-E-C-0000000021.p.64.s.1.xml',
    'WR-P-E-C-0000000021.p.71.s.1.xml',
    'WR-P-E-C-0000000021.p.66.s.1.xml',
    'WR-P-E-C-0000000021.p.73.s.1.xml',
    'WR-P-E-C-0000000021.p.51.s.1.xml',
    'WR-P-E-C-0000000021.p.70.s.1.xml',
    'WR-P-E-C-0000000021.p.60.s.1.xml',
    'WR-P-E-C-0000000021.p.49.s.1.xml',
    'WR-P-E-C-0000000021.p.54.s.1.xml',
    'WR-P-E-C-0000000021.p.55.s.1.xml',
    'WR-P-E-C-0000000021.p.68.s.1.xml',
    'WR-P-E-C-0000000021.p.58.s.1.xml',
    'WR-P-E-C-0000000021.p.63.s.1.xml',
    'wiki-1617.p.27.s.2.xml',
    'WS-U-E-A-0000000020.p.44.s.8.xml',
    'WR-P-P-I-0000000058.p.1.s.3.xml',
    'WS-U-E-A-0000000025.p.40.s.9.xml',
    'WR-P-P-I-0000000066.p.1.s.2.xml',
    'WR-P-P-C-0000000051.txt-2.xml',
    'WR-P-P-C-0000000051.txt-308.xml',
    'WR-P-P-C-0000000051.txt-316.xml',
    'WR-P-P-C-0000000051.txt-64.xml',
    'WR-P-P-I-0000000067.p.1.s.2.xml',
    'wiki-1941.p.37.s.2.xml',
    'wiki-1941.p.33.s.2.xml',
    'wiki-1941.p.30.s.2.xml',
    'wiki-1941.p.28.s.2.xml',
    'wiki-1941.p.29.s.1.xml',
    'WR-P-P-I-0000000055.p.1.s.2.xml',
    'WR-P-P-G-0000000020.p.8.s.1.xml',
    'WR-P-P-I-0000000048.p.1.s.2.xml',
    'dpc-vla-001175-nl-sen.p.136.s.3.xml',
    'WS-U-E-A-0000000245.p.17.s.6.xml',
    'WS-U-E-A-0000000220.p.35.s.15.xml',
    'WS-U-E-A-0000000237.p.4.s.4.xml',
    'wiki-1941.p.5.s.1.xml',
    'dpc-kam-001286-nl-sen.p.10.s.2.xml',
    'WS-U-E-A-0000000211.p.17.s.9.4.xml',
    'wiki-7543.p.22.s.1.xml',
    'wiki-7543.p.22.s.2.xml'
}