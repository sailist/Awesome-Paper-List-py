import requests
from bs4 import BeautifulSoup
import re

# 2022

for ename,link in [
    ['main-track','https://ijcai-22.org/main-track-accepted-papers/'],
    ['special-track-for-good','https://ijcai-22.org/special-track-on-ai-for-good-accepted-papers/'],
    ['special-track-arts-and-creativity','https://ijcai-22.org/special-track-on-ai-the-arts-and-creativity-accepted-papers/'],
    ['survey-track','https://ijcai-22.org/survey-track-accepted-papers/'],
    ['journal-track','https://ijcai-22.org/journal-track-accepted-papers/'],
    ['best-papers-from-sisconf','https://ijcai-22.org/best-papers-from-sister-conferences-accepted-papers/'],
    # ['doctoral-consortium','https://ijcai-22.org/doctoral-consortium-accepted-papers/'], # other process logic
    ['demo-track','https://ijcai-22.org/demo-track-accepted-papers/'],
]:
    print(f'Process 2022 for {ename}')
    res = requests.get(link)
    soup = BeautifulSoup(res.content.decode(),features="lxml")
    ps = soup.find_all('div',class_='paper_wrapper')
    ps = [i.get_text('\n').split('\n')[:2] for i in ps]
    
    ps = [''.join([re.sub('.*([0-9]+)','\1. ',i[0]),i[1]]) for i in ps]
    with open(f'2022-{ename}.md','w') as w:
        w.write('\n'.join(ps))
    print(f'    Totally {len(ps)} papers.')
    
    
# 2021 

print('Process 2021')

res = requests.get("https://ijcai-21.org/program-main-track/")
soup = BeautifulSoup(res.content.decode(),features="lxml")
ps = [i.get_text() for i in soup.find_all('div',class_='gl-title')]
ps = [re.sub('^#([0-9]+) ','\\1. ',i) for i in ps]

with open(f'2021.md','w') as w:
    w.write('\n'.join(ps))
    
print(f'    Totally {len(ps)} papers.')


# 2020

print('Process 2020')
res = requests.get("https://static.ijcai.org/2020-accepted_papers.html")
soup = BeautifulSoup(res.content.decode(),features="lxml")
ps = [i.strong.get_text() for i in soup.find_all('li',class_='paper')]
ps = [f'{i}. {t}' for i,t in enumerate(ps,start=1)]

with open(f'2020.md','w') as w:
    w.write('\n'.join(ps))

print(f'    Totally {len(ps)} papers.')