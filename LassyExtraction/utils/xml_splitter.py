from typing import List, Tuple
import os
from glob import glob


def split_xml(uncompressed: str) -> List[Tuple[str, str]]:
    """
        Takes a wall of text and splits it into xml parts.

    :param uncompressed:
    :type uncompressed:
    :return:
    :rtype:
    """

    def getname(xml_block: str) -> str:
        """
            Finds the sentence name attribute of an xml part.
        :param xml_block:
        :type xml_block:
        :return:
        :rtype:
        """
        return xml_block.split('sentid="')[1].split('"')[0]

    xmls = uncompressed.split('</alpino_ds>')[:-1]
    xmls = list(map(lambda x: x + '</alpino_ds>', xmls))
    names = list(map(getname, xmls))
    return list(zip(xmls, names))


def store_xml(item: Tuple[str, str], prefix: str = '') -> None:
    if len(item[0]):
        with open(prefix + item[1] + '.xml', 'w') as f:
            f.write(item[0])


def get_files(rootdir: str, extension: str) -> List[str]:
    filelist = [file for dir in os.walk(rootdir) for file in glob(os.path.join(dir[0], '*.' + extension))]
    return filelist


if __name__ == '__main__':
    file_extension = input('File extension?')
    filelist = get_files(os.getcwd(), file_extension)
    print('Found {} uncompressed files.'.format(len(filelist)))
    for file in filelist:
        with open(file, 'r') as f:
            r = f.read()
            samples = split_xml(r)
            for sample in samples[:1]:
                store_xml(sample, '/'.join(file.split('/')[:-1]) + '/')
