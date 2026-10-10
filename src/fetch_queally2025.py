#!/usr/bin/env python
"""
Gridded tree mortality at NEON SOAP and TEAK from Queally et al. (2025, Global
Change Biology, doi:10.1111/gcb.70246), Zenodo doi:10.5281/zenodo.13436293
(CC BY 4.0).

The archive is a single 3.1 GB zip. Only a few small members are needed, so
they are read with HTTP range requests through zipfile (whose CRC check
verifies each member) instead of downloading the whole archive:
  data/raw/mortality.tif          30 m crown-mean mortality, bands
                                  mort_16 (Stovall et al. 2019, NAIP 2016)
                                  and mort_17 (Hemming-Schroeder et al.
                                  2023, 2017), masked where high-incidence-
                                  angle NAIP made the estimates unreliable
  data/raw/mortality_names.txt    its band names
  data/processed/sites.tif        1 = SOAP, 2 = TEAK, on the same grid
  README.md
Members are written flat into OUTPUTDIR with the Zenodo record as
record.json.

    python fetch_queally2025.py $E/queally2025
"""
import io
import json
import time
import zipfile
import click
import requests
from pathlib import Path

RECORD = 'https://zenodo.org/api/records/13436293'
MEMBERS = ('data/raw/mortality.tif', 'data/raw/mortality_names.txt',
           'data/processed/sites.tif', 'README.md')


class RangeReader(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests"""

    def __init__(self, url, session):
        self.s = session
        r = self.s.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        self.url, self.n, self.pos = r.url, int(r.headers['Content-Length']), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.n}
        self.pos = base[whence] + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.n:
            return 0
        end = min(self.n, self.pos + len(b)) - 1
        for attempt in range(6):
            try:
                r = self.s.get(self.url, timeout=120, headers={
                    'Range': f'bytes={self.pos}-{end}'})
                r.raise_for_status()
                d = r.content
                if len(d) == end - self.pos + 1:
                    break
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
        else:
            raise IOError(f'range {self.pos}-{end} failed')
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
def main(outputdir):
    outputdir.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    r = s.get(RECORD, timeout=60)
    r.raise_for_status()
    rec = r.json()
    (outputdir / 'record.json').write_text(json.dumps(rec, indent=1))
    f = rec['files'][0]
    click.echo(f'{f["key"]}: {f["size"]} bytes, {f["checksum"]}')
    z = zipfile.ZipFile(io.BufferedReader(
        RangeReader(f['links']['self'], s), 1 << 18))
    root = z.namelist()[0].split('/')[0]
    for m in MEMBERS:
        info = z.getinfo(f'{root}/{m}')
        out = outputdir / Path(m).name
        out.write_bytes(z.read(info))  # raises on a CRC mismatch
        click.echo(f'{out}: {info.file_size} bytes, CRC OK')


if __name__ == '__main__':
    main()
