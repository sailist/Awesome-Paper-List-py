import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin


def process_group(i, fmt='md', root=''):
    global count
    if fmt == 'md':
        count += 1
        link = i.attrs["href"]
        link = urljoin(root, link)
        return f'{count}. {i.get_text()} | [link]({link}) \n'
    elif fmt == 'txt':
        return f"{i.get_text()} \n"


def write_file(name, link):
    global count
    count = 0
    print(f'Crawl for {name}')
    with open(f'{name}.md', 'w') as w, open(f'{name}.txt', 'w') as w2:
        res = requests.get(link)
        content = res.content.decode()
        soup = BeautifulSoup(content, features="lxml")
        ps = soup.find('div', class_='container-fluid').find_all('li')
        ps = [i.a for i in ps]
        w.write(''.join([process_group(i, fmt='md', root=link) for i in ps]))
        w2.write(''.join([process_group(i, fmt='txt') for i in ps]))
        print(f' - Totally write {count} papers')


root = "https://papers.nips.cc/"
res = requests.get(root)
soup = BeautifulSoup(res.content.decode(), features="lxml")
lists = soup.find('div', class_='col-sm').find_all('a')
match_year = re.compile('([0-9]{4})')
for item in lists:
    link = urljoin(root, item.attrs['href'])
    name = match_year.search(item.get_text()).group()
    write_file(name, link)
