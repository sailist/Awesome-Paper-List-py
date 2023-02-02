# Awosome-Paper-List

This repository collects the list of accepted paper from (deep learning) conference. All lists are crawled by python scripts for later maintenance and analysis. Welcome to contribute.

# Conference collections

Including IJCAI, ECCV, ACCV, NIPS, ACMMM, WACV, ICML, CVPR, AAAI, ICCV, ACL, BMVC. See [DETAILS.md](./DETAILS.md) for details.

# Find related work

A simple script are provided under `focus/` folder, which can use multiple regular experssions to filter papers you may be interested:

```
cd focus
# edit match_lis
python filter.py
# a xlsx file will be generated.
```

# Download pdf files

You can run the following script to download papers from all supported conferences:

```
git clone --depth=1 https://github.com/sailist/Awosome-Paper-List-py
cd Awosome-Paper-List-py/scripts/crawl
bash scripts/bash.sh
```

or execute each python file to crawl and download papers for each conference:

```
cd Awosome-Paper-List-py/scripts/crawl
python cvpr.py
```

In addition, compressed and uncompressed versions can also be downloaded from Baidu Cloud

you can also download [zip version::kfnd](https://pan.baidu.com/s/17mq6Kth4pVu7inuxd-pdEQ) or [uncompressed version::3mm9](https://pan.baidu.com/s/1yDs3E1ClbCLzwSemTfz95w) from baidupan.

> Downloaded pdfs are uploaded via [bypy](https://github.com/houtianze/bypy), and the corresponding scripts are `pdfs/sync-zip-bypy.py` and `pdfs/sync-bypy.py`


# Reference

- https://github.com/yarkable/Awesome-Computer-Vision-Paper-List

# Other Tools

- [Conference-Acceptance-Rates](https://github.com/lixin4ever/Conference-Acceptance-Rate)
- [ccfddl - Conference-Deadlines](https://ccfddl.github.io/)
