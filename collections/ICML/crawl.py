import requests
from bs4 import BeautifulSoup


def parse_2021(soup):
    ps = soup.find_all('div', class_='maincard')
    links = [[
        i.find('div', class_='maincardBody').get_text(),
        i.find('a', class_='paper-pdf-link')
    ] for i in ps]
    links = [(i[0], i[1].attrs['href']) for i in links if i[1] is not None]
    return links


def parse_2022(soup):
    ps = soup.find_all('div', class_='maincard')
    links = [[
        i.find('div', class_='maincardBody').get_text(),
        i.find('a', class_='href_PDF')
    ] for i in ps]
    links = [(i[0], i[1].attrs['href']) for i in links if i[1] is not None]
    return links


def parse_2020(soup):
    ps = soup.find_all('div', class_='maincard')
    links = [[
        i.find('div', class_='maincardBody').get_text(),
    ] for i in ps]
    links = [(i[0], ) for i in links]
    return links


def process_group(link, fmt='md'):
    global count
    if fmt == 'md':
        count += 1
        if len(link) == 1:
            return f'{count}. {link[0]} \n'
        return f'{count}. {link[0]} | [link]({link[1]}) \n'
    elif fmt == 'txt':
        return f'{link[0]} \n'


def write_file(name, links):
    global count
    count = 0
    print(f'Crawl for {name}')
    with open(f'{name}.md', 'w') as w, open(f'{name}.txt', 'w') as w2:
        w.write(''.join([process_group(i, fmt='md') for i in links]))
        w2.write(''.join([process_group(i, fmt='txt') for i in links]))
        print(f' - Totally write {count} papers')


res = requests.get("https://icml.cc/Conferences/2022/Schedule?type=Poster")
soup = BeautifulSoup(res.content.decode(), features="lxml")
links = parse_2022(soup)
write_file('2022', links)

res = requests.get("https://icml.cc/Conferences/2021/Schedule?type=Poster")
soup = BeautifulSoup(res.content.decode(), features="lxml")
links = parse_2021(soup)
write_file('2021', links)

res = requests.get("https://icml.cc/Conferences/2020/Schedule?type=Poster")
soup = BeautifulSoup(res.content.decode(), features="lxml")
links = parse_2020(soup)
write_file('2020', links)