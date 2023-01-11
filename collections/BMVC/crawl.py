import requests
from bs4 import BeautifulSoup

def process_group(g, fmt):
    global count

    if len(g) == 1:
        return ''

    
    title = g.strong.get_text()

    mstr = f'{title}'

    refs = g.find_all('a')
    
    rres = []
    for r in refs:
        ctt = r.get_text()
        if ctt in {'Paper', 'Supplemental', 'Code'}:
            rurl = r.attrs['href']
        elif 'Poster Session' in ctt:
            continue
        elif 'Oral Session' in ctt:
            rres.append('[Oral]')
            continue
        else:
            print('ignore', ctt, f'of {r}')
            continue

        rstr = f"[{ctt}]({rurl})"
        rres.append(rstr)
    rres = ', '.join(rres)
    if fmt == 'md':
        count += 1
        return f'{count}. {mstr} | {rres} \n'
    elif fmt == 'txt':
        return f'{mstr} \n'

def write_file(name, link):
    global count
    count = 0
    print(f'Crawl for {name}')
    with open(f'{name}.md', 'w') as w, open(f'{name}.txt', 'w') as w2:
        res = requests.get(link)
        soup = BeautifulSoup(res.content.decode())
        ps = soup.find('table',class_='table')

        paper_group = ps.find_all('tr')
        w.write(''.join([process_group(i, fmt='md') for i in paper_group]))
        w2.write(''.join(
            [process_group(i, fmt='txt') for i in paper_group]))
        print(f' - Totally write {count} papers')

        
write_file('2021',"https://www.bmvc2021-virtualconference.com/programme/accepted-papers/")
write_file('2020',"https://www.bmvc2020-conference.com/programme/accepted-papers/")