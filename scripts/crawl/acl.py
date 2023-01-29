import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from base import Crawl


class ACL(Crawl):
    def __init__(self, link=None, year=None, type='strong') -> None:
        super().__init__()
        self.link = link
        self.year = year
        self.ttype = type

    def parse(self):
        year = self.year
        res = requests.get(self.link)
        soup = BeautifulSoup(res.content.decode(), features='lxml')
        lis = soup.find_all('p', class_='align-items-stretch')
        root = 'https://aclanthology.org/'
        for item in lis:
            title = item.strong.get_text()
            link = item.strong.a.attrs['href']
            link = urljoin(root, link)
            attrs = {'link': link}

            for k in item.find('span',
                               class_='list-button-row').find_all('a'):
                link = k.attrs['href']
                key = k.get_text().strip()
                if 'http' not in link:
                    link = urljoin(root, link)
                if key == '' and 'title' in k.attrs:
                    key = k.attrs['title'].lower().strip()
                if key == '':
                    print(k, key, link)
                attrs[key] = link
                if key == 'pdf':
                    self.append_download_item(year, title, link)
            self.append_item(year, title, attrs=attrs)


ACL('https://aclanthology.org/events/acl-2010/', '2010').start()
ACL('https://aclanthology.org/events/acl-2011/', '2011').start()
ACL('https://aclanthology.org/events/acl-2012/', '2012').start()
ACL('https://aclanthology.org/events/acl-2013/', '2013').start()
ACL('https://aclanthology.org/events/acl-2014/', '2014').start()
ACL('https://aclanthology.org/events/acl-2015/', '2015').start()
ACL('https://aclanthology.org/events/acl-2016/', '2016').start()
ACL('https://aclanthology.org/events/acl-2017/', '2017').start()
ACL('https://aclanthology.org/events/acl-2018/', '2018').start()
ACL('https://aclanthology.org/events/acl-2019/', '2019').start()
ACL('https://aclanthology.org/events/acl-2020/', '2020').start()
ACL('https://aclanthology.org/events/acl-2021/', '2021').start()
ACL('https://aclanthology.org/events/acl-2022/', '2022').start()
