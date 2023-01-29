from bs4 import BeautifulSoup
import json
import re
import requests
from urllib.parse import urljoin
from base import Crawl


class ICCV(Crawl):
    def __init__(self, year, link) -> None:
        super().__init__()
        self.year = year
        self.link = link

    def parse(self):
        pass


ICCV("2019", "https://api.openreview.net/notes?invitation=ICLR.cc%2F2019%2FConference%2F-%2FBlind_Submission&details=replyCount%2Cinvitation%2Coriginal%2CdirectReplies&limit=1000&offset=0").start()
ICCV("2019", "https://api.openreview.net/notes?invitation=ICLR.cc%2F2019%2FConference%2F-%2FBlind_Submission&details=replyCount%2Cinvitation%2Coriginal%2CdirectReplies&limit=1000&offset=1000").start()
