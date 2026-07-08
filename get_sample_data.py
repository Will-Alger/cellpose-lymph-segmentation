"""
Download the sample nuclei images (with ground-truth masks) into testdata/.

These are the images the README examples use (base: gray_2D.png, 225 GT cells).
They come from Cellpose's public test bundle. Requires cellpose to be installed.

    python get_sample_data.py
"""

import os
import zipfile

from cellpose.utils import download_url_to_file

URL = "https://osf.io/download/s52q3/"


def main():
    os.makedirs("testdata", exist_ok=True)
    zip_path = os.path.join("testdata", "data.zip")
    print("Downloading sample data...")
    download_url_to_file(URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall("testdata")
    print("Sample data ready in testdata/ (e.g. testdata/data/2D/gray_2D.png)")


if __name__ == "__main__":
    main()
