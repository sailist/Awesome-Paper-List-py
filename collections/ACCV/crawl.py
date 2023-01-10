import re
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
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

    title = href.string
    url = urljoin(root, href.attrs['href'])

    mstr = f'{title}'.strip()

    refs = g[2].find_all('a')
    rres = [f'[[link]({url})]']
    for r in refs:
        if r.string in {'pdf', 'supplementary material'}:
            rurl = urljoin(root, r.attrs['href'])
        elif r.string in {'arXiv'}:
            rurl = r.attrs['href']
        elif r.string in {'bibtex','DOI'}:
            continue
        else:
            continue
            # print(r.string)
            

        rstr = f"[[{r.string}]({rurl})]"
        rres.append(rstr)
    rres = ' '.join(rres)
    if fmt == 'md':
        count += 1      
        return f'{count}. {mstr} | {rres} \n'
    elif fmt == 'txt':
        return f'{mstr} \n'


def write_file():
    global count
    count = 0
    
    res = requests.get('https://www.ecva.net/papers.php')
    soup = BeautifulSoup(res.content.decode())
    res = soup.find_all('dl')
    for r in res:
        name = re.search('20[0-9][0-9]',r.a.attrs['href']).group()
        print(f'Process {name}')
        with open(f'{name}.md', 'w') as w, open(f'{name}.txt', 'w') as w2:
            paper_group = list(iter_paper(r.children))
            w.write(''.join([process_group(i, fmt='md') for i in paper_group]))
            w2.write(''.join([process_group(i, fmt='txt')
                             for i in paper_group]))
            print(f' - Totally write {count} papers')

write_file()