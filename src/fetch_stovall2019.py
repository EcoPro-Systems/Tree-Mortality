#!/usr/bin/env python
"""
Stovall, Shugart & Yang (2019) tree mortality at NEON SOAP and TEAK.

About 2 M trees segmented from the NEON 2013 lidar canopy height model, each
with the fraction of its crown classified dead in NAIP 2009, 2010, 2012,
2014 and 2016, a dead flag (> 37.5% of the crown) and the first dead year
(figshare doi:10.6084/m9.figshare.7609193, version 4, CC BY 4.0). Downloads
ALLtrees_v2.csv (the file earlier comparisons used), checks its md5 against
the figshare record, and saves the record as record.json.

    python fetch_stovall2019.py $E/stovall2019
"""
import hashlib
import json
import click
import requests
from pathlib import Path

ARTICLE = 'https://api.figshare.com/v2/articles/7609193/versions/4'
FILE = 'ALLtrees_v2.csv'


def md5sum(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
def main(outputdir):
    outputdir.mkdir(parents=True, exist_ok=True)
    r = requests.get(ARTICLE, timeout=60)
    r.raise_for_status()
    rec = r.json()
    (outputdir / 'record.json').write_text(json.dumps(rec, indent=1))
    f = next(x for x in rec['files'] if x['name'] == FILE)
    out = outputdir / FILE
    if not (out.exists() and md5sum(out) == f['computed_md5']):
        with requests.get(f['download_url'], stream=True, timeout=600) as r:
            r.raise_for_status()
            tmp = out.with_suffix('.part')
            with open(tmp, 'wb') as g:
                for b in r.iter_content(1 << 22):
                    g.write(b)
        tmp.rename(out)
    got = md5sum(out)
    if got != f['computed_md5']:
        raise click.ClickException(f'{FILE}: md5 {got} != '
                                   f'{f["computed_md5"]}')
    click.echo(f'{out}: {out.stat().st_size} bytes, md5 {got} OK')


if __name__ == '__main__':
    main()
