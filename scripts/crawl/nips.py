import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
from urllib.parse import urljoin
from collections import Counter
from base import Crawl


class NIPS(Crawl):
    def parse(self):
        root = "https://papers.nips.cc/"
        res = requests.get(root)
        soup = BeautifulSoup(res.content.decode(), features="lxml")
        lists = soup.find('div', class_='col-sm').find_all('a')
        match_year = re.compile('([0-9]{4})')
        for item in tqdm(lists):
            link = urljoin(root, item.attrs['href'])
            year = match_year.search(item.get_text()).group()
            res = requests.get(link)
            content = res.content.decode()
            soup = BeautifulSoup(content, features="lxml")
            ps = soup.find('div', class_='container-fluid').find_all('li')

            for item in ps:
                item = item.a
                link = item.attrs["href"]
                if 'http' not in link:
                    link = urljoin(root, link)
                pdf_link = link.replace('Abstract.html', 'Paper.pdf').replace('hash','file')
                title = item.get_text().strip()
                self.append_item(year,
                                 title,
                                 attrs={'link': link, 'pdf': pdf_link})
                self.append_download_item(year, title, pdf_link)


NIPS().start()
