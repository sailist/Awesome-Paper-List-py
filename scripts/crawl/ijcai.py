import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from base import Crawl
from joblib import Memory



class IJCAI(Crawl):
    def parse_v0(self, root):
        res = requests.get(root)
        year = re.search('[0-9]{4}', root).group()
        soup = BeautifulSoup(res.content.decode(), features='lxml')
        ps = soup.find_all('p')
        ps = [i for i in ps if len(i.find_all('a')) >= 2]
        for item in ps:
            attrs = {}
            for href in item.find_all('a'):
                link = href.attrs['href']
                link = urljoin(root, link)
                key = href.get_text().strip().lower()
                if link.endswith('pdf'):
                    title = key
                    key = 'pdf'

                key = {
                    'pdf': 'pdf',
                    'abstract': 'link'
                }[key]
                attrs[key] = link
                if key == 'pdf':
                    self.append_download_item(year, title, link)
            self.append_item(year, title, attrs)

    def parse_v1(self, root):
        res = requests.get(root)
        year = re.search('[0-9]{4}', root).group()
        soup = BeautifulSoup(res.content.decode(), features='lxml')
        ps = soup.find_all('p')
        ps = [i for i in ps if len(i.find_all('a')) >= 2]
        for item in ps:
            brindex = [i for i,content in enumerate(item.contents) if content.name == 'br']
            if len(brindex) == 0:
                print(item)
                continue
            title = ''.join([i.get_text() for i in item.contents[:brindex[0]]])
            title = title.strip()
            attrs = {}
            for href in item.find_all('a'):
                link = href.attrs['href']
                link = urljoin(root, link)
                key = href.get_text().strip().lower()
                try:
                    key = {
                        'pdf': 'pdf',
                        'abstract': 'link'
                    }[key]
                except:
                    print(item)
                attrs[key] = link
                if key == 'pdf':
                    self.append_download_item(year, title, link)
            self.append_item(year, title, attrs)

    def parse_v2(self, root):
        res = requests.get(root)
        year = re.search('[0-9]{4}', root).group()
        soup = BeautifulSoup(res.content.decode(), features='lxml')
        ps = soup.find_all('div', class_='paper_wrapper')
        for item in ps:
            title = item.find('div', class_='title').get_text()
            attrs = {}
            for href in item.find_all('a'):
                link = href.attrs['href']
                link = urljoin(root, link)
                key = href.get_text().strip().lower()
                key = {
                    'pdf': 'pdf',
                    'details': 'link',
                }[key]
                attrs[key] = link
                if key == 'pdf':
                    self.append_download_item(year, title, link)
            self.append_item(year, title, attrs)

    def parse(self):
        res = requests.get("https://www.ijcai.org/past_proceedings")
        soup = BeautifulSoup(res.content.decode(), features='lxml')
        content = soup.find('div', class_='field-items')
        proceedings = [i.attrs['href'] for i in content.find_all(
            'a') if 'proceedings' in i.attrs['href']]
        for item in tqdm(proceedings):
            year = int(re.search('[0-9]{4}', item).group())
            if year >= 2017: # (PDF | Details)
                self.parse_v2(item)
            elif year >= 2014: # PDF | Abstract
                self.parse_v1(item)

IJCAI().start()
