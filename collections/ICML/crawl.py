import requests
from bs4 import BeautifulSoup
from collections import Counter


class Crawl:
    def __init__(self, year, link) -> None:
        self.dic = {}
        self.year = year
        self.link = link

    def get_soup(self):
        res = requests.get(self.link)
        return BeautifulSoup(res.content.decode(), features="lxml")

    def parse(self):
        pass

    def start(self):
        self.parse()
        self.write()

    def append_item(self, year, title, attrs=None, type=None):
        if type is not None:
            self.dic.setdefault((year, type), []).append([title, attrs])
        else:
            self.dic.setdefault((year, ''), []).append([title, attrs])

    def write(self):
        cc = Counter()
        for k, v in self.dic.items():
            year = k[0]
            k = '_'.join([i for i in k if i is not None]).strip('_')
            res = []
            for i, (title, attrs) in enumerate(v, start=1):
                if attrs is None:
                    res.append(f'{i}. {title}')
                else:
                    attrs = ', '.join(
                        [f"[{k}]({v})" for k, v in attrs.items()])
                    res.append(f'{i}. {title} | {attrs}')
            cc[year] += len(res)
            with open(f'{k}.md', 'w') as w:
                w.write('\n'.join(res))
                print(f' - write {len(res)} papers for {k}.')
        for k, v in cc.items():
            print(f'total {v} papers in {k}.')


class Crawl2022(Crawl):
    def parse(self):
        soup = self.get_soup()
        ps = soup.find_all('div', class_='maincard')
        links = [[
            i.find('div', class_='maincardBody').get_text().strip(),
            i.find('a', class_='href_PDF')
        ] for i in ps]

        for item in links:
            if item[1] is None:
                continue
            self.append_item('2022', item[0], {'link': item[1].attrs['href']})


class Crawl2021(Crawl):
    def parse(self):
        soup = self.get_soup()
        ps = soup.find_all('div', class_='maincard')
        links = [[
            i.find('div', class_='maincardBody').get_text().strip(),
            i.find('a', class_='paper-pdf-link')
        ] for i in ps]

        for item in links:
            if item[1] is None:
                continue
            self.append_item('2021', item[0], {'link': item[1].attrs['href']})


class Crawl2020(Crawl):
    def parse(self):
        soup = self.get_soup()
        ps = soup.find_all('div', class_='maincard')
        links = [[
            i.find('div', class_='maincardBody').get_text().strip(),
        ] for i in ps]

        for item in links:
            self.append_item('2020', item[0])




Crawl2022("2022", "https://icml.cc/Conferences/2022/Schedule?type=Poster").start()
Crawl2021("2021", "https://icml.cc/Conferences/2021/Schedule?type=Poster").start()
Crawl2020("2020", "https://icml.cc/Conferences/2020/Schedule?type=Poster").start()
