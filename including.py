import os

COLL = 'collections'


with open('DETAILS.md', 'w', encoding='utf-8') as w:
    confs = os.listdir(COLL)
    for conf in confs:
        years = os.listdir(os.path.join(COLL, conf))
        w.write('<details open>\n')
        w.write(f"<summary> {conf} </summary>\n")
        res = []
        for year in sorted(years, reverse=True):
            if not year.endswith('.md'):
                continue
            year_ = year.replace(".md", "")
            res.append(
                f'<a href="https://github.com/sailist/Awosome-Paper-List-py/blob/master/collections/{conf}/{year}.md">{year_}</a>')
        res = ', '.join(res)
        w.write(f'<li>{res}</li>')
        w.write('\n</details>\n\n')
