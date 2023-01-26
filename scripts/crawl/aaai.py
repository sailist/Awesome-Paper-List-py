import fitz  # pip install pymupdf
import os
from bs4 import BeautifulSoup
import re
from collections import Counter
from base import Crawl


def iter_group(div):
    res = []
    for i in list(div.children):
        if i == '\n':
            continue

        rstr = i.get_text()

        if rstr is not None:
            if re.search(' *[0-9]+:.*', rstr):
                yield ' '.join(res)
                res = []
            elif re.search('[,;()]', rstr):
                continue
            rstr = re.sub('-­‐', '-', rstr)
            res.append(' '.join(rstr.split()))
    yield ' '.join(res)


class AAAI(Crawl):
    def __init__(self, year, file) -> None:
        super().__init__()
        self.year = year
        self.file = file

    def parse(self):
        doc = fitz.open(self.file)
        res = []
        for page in doc:
            soup = BeautifulSoup(page.get_text('html'), features='lxml')
            res.extend(list(iter_group(soup.div)))

        res = [i.strip() for i in res]
        res = [i for i in res if re.search('^[0-9]+:', i)]
        res = [re.sub('^[0-9]+: +', '', i) for i in res]
        for item in res:
            self.append_item(self.year, item)


if __name__ == '__main__':
    AAAI('2019', '../../archive/AAAI-19_Accepted_Papers.pdf').start()

    AAAI('2020', '../../archive/AAAI-20-Accepted-Paper-List.pdf').start()

    AAAI('2021',
         '../../archive/AAAI-21_Accepted-Paper-List.Main_.Technical.Track_.pdf').start()

    AAAI('2022',
         '../../archive/AAAI-22_Accepted_Paper_List_Main_Technical_Track.pdf').start()
