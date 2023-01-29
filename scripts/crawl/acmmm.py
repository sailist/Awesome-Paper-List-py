import re
from tqdm import tqdm
from bs4 import BeautifulSoup
import requests
from collections import Counter
from base import Crawl, COLLECTION_ROOT
from urllib.parse import urljoin


class ACMMM(Crawl):
    def __init__(self, year, link) -> None:
        super().__init__()
        self.year = year
        self.link = link

    def parse_2020(self):
        res = requests.get("https://2020.acmmm.org/main-track-list.html")
        soup = BeautifulSoup(res.content.decode(), features="lxml")
        ps = soup.find_all('p')
        ps = [p.get_text().strip() for p in ps]
        ps = [re.sub('^([0-9]+) +- +', '', i) for i in ps]
        for item in ps:
            self.append_item('2020', item)

    def parse_2021(self):
        with open('2021.html') as r:
            soup = BeautifulSoup(r.read(), features="lxml")
            ps = soup.find_all('p')
            ps = [p.get_text().strip() for p in ps]
            ps = [
                re.sub('^([0-9]+) *', '', i) for i in ps
                if re.search('^([0-9]+) *', i)
            ]
            for item in ps:
                self.append_item('2021', item)

    def parse(self):
        year = self.year
        res = requests.get(self.link)
        soup = BeautifulSoup(res.content.decode(), features='lxml')
        sessions = [i for i in soup.find_all(
            'a', class_='section__title') if i.attrs['href'] != '#']
        for session in tqdm(sessions):
            res = requests.get(urljoin(
                "https://dl.acm.org/doi/proceedings/10.1145/3503161", session.attrs['href']))
            soup = BeautifulSoup(res.content.decode(), features='lxml')
            session_name = re.search(
                'SESSION:(.*)', session.get_text()).group(1).lower().strip().replace(' ', '_')
            session_name = re.sub('[-+:_]+', '_', session_name)
            r = soup.find('a', id=session.attrs['id'])
            items = r.find_parent().find_all('h5', 'issue-item__title')
            for item in items:
                title = item.get_text()
                link = item.a.attrs['href']
                link = urljoin(self.link, link)
                self.append_item(year, title,
                                 attrs={'link': link, 'group': session_name})


ACMMM('2022', 'https://dl.acm.org/doi/proceedings/10.1145/3503161').start()
ACMMM('2021', 'https://dl.acm.org/doi/proceedings/10.1145/3474085').start()
ACMMM('2020', 'https://dl.acm.org/doi/proceedings/10.1145/3394171').start()
ACMMM('2019', 'https://dl.acm.org/doi/proceedings/10.1145/3343031').start()
ACMMM('2018', 'https://dl.acm.org/doi/proceedings/10.1145/3240508').start()
