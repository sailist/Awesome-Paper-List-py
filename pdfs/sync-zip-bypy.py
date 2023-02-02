import zipfile
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from bypy import ByPy


cur_dir = os.path.dirname(os.path.abspath(__file__))
sync_f = os.path.join(cur_dir, '.sync')


def upload(root, fs):
    bp = ByPy()
    year = os.path.basename(root)
    conf = os.path.basename(os.path.dirname(root))
    if year.isnumeric():
        zipf = os.path.join(os.path.dirname(root), f'{year}.zip')
        print(f'compress {conf}/{year}')
        with zipfile.ZipFile(zipf, 'w') as w:
            for f in tqdm(fs):
                absf = os.path.join(root, f)
                w.write(absf, f)

        tgt = os.path.join('paper-pdfs-zip', conf, f'{year}.zip')
        bp.upload(zipf, tgt)
        os.remove(zipf)
        with open(f'.{year}.sync', 'w'):
            pass


def main():
    tasks = []
    for root, _, fs in os.walk(cur_dir):
        year = os.path.basename(root)
        if year.isnumeric() and not os.path.exists(os.path.join(root, f'.{year}.sync')):
            tasks.append(delayed(upload)(root, fs))

    Parallel(-1, verbose=10)(tasks)


if __name__ == '__main__':
    main()
