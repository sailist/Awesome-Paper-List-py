import re
from bs4 import BeautifulSoup
import requests

print('ACM MM 2021')
with open('2021.html') as r:
    soup = BeautifulSoup(r.read(), features="lxml")
    ps = soup.find_all('p')
    with open('2021.md', 'w') as w:
        ps = [p.get_text().strip() for p in ps]
        ps = [
            re.sub('^([0-9]+) ', '\\1. ', i) for i in ps
            if re.search('^[0-9]', i)
        ]
        w.write('\n'.join(ps))
    print(f'    Total {len(ps)} papers.')
    
#

res = requests.get("https://2020.acmmm.org/main-track-list.html")
soup = BeautifulSoup(res.content.decode(), features="lxml")
ps = soup.find_all('p')
print('ACM MM 2020')
with open('2020.md', 'w') as w:
    ps = [p.get_text().strip() for p in ps]
    ps = [i.replace(' - ', '. ', 1) for i in ps]
    w.write('\n'.join(ps))
    print(f'    Total {len(ps)} papers.')
