import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin


def iter_paper(c):
    tmp = []
    for i in c:
        if i.name == 'dt':
            yield tmp
            tmp = []
        if i != '\n':
            tmp.append(i)
    yield tmp


def process_group(g, fmt):
    global count

    root = 'https://www.ecva.net/'

    if len(g) == 0 or len(g) == 1 or g[0].name != 'dt':
        # print(g)
        return ''

    href = g[0].a

    title = href.get_text()
    url = urljoin(root, href.attrs['href'])

    mstr = f'{title}'.strip()

    refs = g[2].find_all('a')
    rres = [f'[[link]({url})]']
    for r in refs:
        ctt = r.get_text()
        if ctt in {'pdf', 'supplementary material'}:
            rurl = urljoin(root, r.attrs['href'])
        elif ctt in {'arXiv', 'video'}:
            rurl = r.attrs['href']
        elif ctt in {'bibtex', 'DOI'}:
            continue
        else:
            print('ignore', ctt, f'of {r}')
            continue

        rstr = f"[{ctt}]({rurl})"
        rres.append(rstr)
    rres = ', '.join(rres)
    if fmt == 'md':
        count += 1
        return f'{count}. {mstr} | {rres} \n'
    elif fmt == 'txt':
        return f'{mstr} \n'


def write_file():
    global count

    res = requests.get("https://www.ecva.net/papers.php")
    soup = BeautifulSoup(res.content.decode())
    years = [
        list(i.dl.children)
        for i in soup.find_all('div', class_='accordion-content')
    ]
    for year in years:
        count = 0
        paper_group = list(iter_paper(year))
        md = [process_group(i, fmt='md') for i in paper_group]
        txt = [process_group(i, fmt='txt') for i in paper_group]
        md = [i for i in md if i != '']
        name = re.search('eccv_([0-9]{4})', md[0]).group(1)
        print(f'Crawl {name}')
        with open(f'{name}.md', 'w') as w, open(f'{name}.txt', 'w') as w2:
            w.write(''.join(md))
            w2.write(''.join(txt))
            print(f' - Totally write {count} papers')


write_file()