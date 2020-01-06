import os
import xml.etree.cElementTree as et
from glob import glob
from warnings import warn

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from LassyExtraction.graphutils import *


def is_public(filename: str) -> bool:
    return 'wiki' in filename or 'WR-P-E-J' in filename


class Lassy(Dataset):
    """
        Lassy dataset. A wrapper that feeds samples into the extraction algorithm.
    """

    def __init__(self, root_dir: str = '/home/kokos/Projects/Lassy 4.0/LassySmall',
                 treebank_dir: str = '/Treebank', transform: Optional[Compose] = None,
                 ignore: Optional[str] = 'LassyExtraction/utils/ignored.txt') -> None:

        if os.path.isdir(root_dir) and os.path.isdir(root_dir+treebank_dir):
            self.root_dir = root_dir
            self.treebank_dir = root_dir + treebank_dir
        else:
            raise ValueError('%s and %s must be existing directories' % (root_dir, treebank_dir))

        self.ignored = []
        if ignore is not None:
            try:
                with open(ignore, 'r') as f:
                    self.ignored = list(map(lambda x: x[0:-1], f.readlines()))
                    print('Ignoring {} samples..'.format(len(self.ignored)))
            except FileNotFoundError:
                warn('Could not open the ignore file.')

        self.filelist = [y for x in os.walk(self.treebank_dir) for y in glob(os.path.join(x[0], '*.[xX][mM][lL]'))
                         if y.split('/')[-1] not in self.ignored]
        self.transform = transform

        print('Dataset constructed with {} samples.'.format(len(self.filelist)))

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, file_or_idx: Union[int, str]) -> Any:
        idx, file = self.match_file_idx(file_or_idx)

        parse = et.parse(file)
        sample = (idx, file, parse)

        if self.transform:
            return self.transform(sample)

        return sample

    def match_file_idx(self, file_or_index: Union[int, str]) -> Tuple[int, str]:
        if isinstance(file_or_index, str):
            return self.filelist.index(file_or_index), file_or_index
        elif isinstance(file_or_index, int):
            return file_or_index, self.filelist[file_or_index]
        else:
            raise TypeError('Index must be int or str.')
