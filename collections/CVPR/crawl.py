import requests
from bs4 import BeautifulSoup
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

    root = 'https://openaccess.thecvf.com/'

    if len(g) == 0 or len(g) == 1 or g[0].name != 'dt':
        # print(g)
        return ''

    href = g[0].a

    title = href.string
    url = urljoin('https://openaccess.thecvf.com/', href.attrs['href'])

    mstr = f'{title}'

    refs = g[2].find_all('a')
    rres = [f'[[link]({url})]']
    for r in refs:
        if r.string in {'pdf', 'supp'}:
            rurl = urljoin(root, r.attrs['href'])
        elif r.string in {'arXiv'}:
            rurl = r.attrs['href']
        elif r.string in {'bibtex'}:
            continue
        else:
            # print(r.string)
            pass

        rstr = f"[[{r.string}]({rurl})]"
        rres.append(rstr)
    rres = ' '.join(rres)
    if fmt == 'md':
        count += 1      
        return f'{count}. {mstr} | {rres} \n'
    elif fmt == 'txt':
        return f'{mstr} \n'


count = 0


def write_file(name, links):
    global count
    count = 0
    print(f'Crawl for {name}')
    with open(f'{name}.md', 'w') as w, open(f'{name}.txt', 'w') as w2:
        for link in links:
            res = requests.get(link)
            content = res.content.decode()
            soup = BeautifulSoup(content, features="lxml")
            paper_group = list(iter_paper(soup.dl.children))
            w.write(''.join([process_group(i, fmt='md') for i in paper_group]))
            w2.write(''.join([process_group(i, fmt='txt')
                             for i in paper_group]))
        print(f' - Totally write {count} papers')


write_file('2022', [
    'https://openaccess.thecvf.com/CVPR2022?day=2022-06-21',
    'https://openaccess.thecvf.com/CVPR2022?day=2022-06-22',
    'https://openaccess.thecvf.com/CVPR2022?day=2022-06-23',
    'https://openaccess.thecvf.com/CVPR2022?day=2022-06-24',
])

write_file('2021', [
    'https://openaccess.thecvf.com/CVPR2021?day=2021-06-21',
    'https://openaccess.thecvf.com/CVPR2021?day=2021-06-22',
    'https://openaccess.thecvf.com/CVPR2021?day=2021-06-23',
    'https://openaccess.thecvf.com/CVPR2021?day=2021-06-24',
    'https://openaccess.thecvf.com/CVPR2021?day=2021-06-25',
])

write_file('2020', [
    'https://openaccess.thecvf.com/CVPR2020.py?day=2020-06-16',
    'https://openaccess.thecvf.com/CVPR2020.py?day=2020-06-17',
    'https://openaccess.thecvf.com/CVPR2020.py?day=2020-06-18',
])

write_file('2019', [
    'https://openaccess.thecvf.com/CVPR2019.py?day=2019-06-18',
    'https://openaccess.thecvf.com/CVPR2019.py?day=2019-06-19',
    'https://openaccess.thecvf.com/CVPR2019.py?day=2019-06-20',
])
