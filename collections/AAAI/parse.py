import fitz  # pip install pymupdf
import os
from bs4 import BeautifulSoup
import re


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
            elif re.search('[,;]', rstr):
                continue
            rstr = re.sub('-­‐', '-', rstr)
            res.append(' '.join(rstr.split()))
    yield ' '.join(res)


def write_list(name, f):
    with open(f'{name}.md', 'w') as w:
        doc = fitz.open(f)
        res = []
        for page in doc:
            soup = BeautifulSoup(page.get_text('html'))
            res.extend(list(iter_group(soup.div)))

        res = [i.strip() for i in res]
        res = [i for i in res if re.search('^[0-9]+:', i)]
        res = [i.replace(':', '.', 1) for i in res]
        w.write('\n'.join(res))
    print(f'Totally write {len(res)} papers.')


write_list('2019', '../../archive/AAAI-19_Accepted_Papers.pdf')

write_list('2020', '../../archive/AAAI-20-Accepted-Paper-List.pdf')

write_list(
    '2021',
    '../../archive/AAAI-21_Accepted-Paper-List.Main_.Technical.Track_.pdf')

write_list(
    '2022',
    '../../archive/AAAI-22_Accepted_Paper_List_Main_Technical_Track.pdf')
