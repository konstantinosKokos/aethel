from torch.utils.data import Dataset
from torchvision.transforms import Compose

from src.utils.typevars import *
from warnings import warn

from glob import glob
import os
import xml.etree.cElementTree as et


class Lassy(Dataset):
    """
        Lassy dataset. A wrapper that feeds samples into the extraction algorithm.
    """

    def __init__(self, root_dir: str = '/home/kokos/Documents/Projects/Lassy/LassySmall 4.0',
                 treebank_dir: str = '/Treebank', transform: Optional[Compose] = None,
                 ignore: Optional[str] = 'src/utils/ignored.txt') -> None:

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
                         if y not in self.ignored]
        self.transform = transform

        print('Dataset constructed with {} samples.'.format(len(self.filelist)))

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: Union[int, str]) -> Any:
        if isinstance(index, int):
            file = self.index_to_filename(index)
        elif isinstance(index, range):
            return [self.__getitem__(i) for i in index]
        elif isinstance(index, str):
            file = index
        else:
            raise TypeError('Index must be int, str or range')

        parse = et.parse(file)
        parse.getroot().set('type', 'Tree')
        sample = (index, file, parse)

        if self.transform:
            return self.transform(sample)

        return sample

    def index_to_filename(self, index: int) -> str:
        return self.filelist[index]
