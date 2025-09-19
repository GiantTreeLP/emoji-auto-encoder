import os
import subprocess
import sys
from multiprocessing.pool import Pool
from typing import List, Callable
from urllib import request


def create_twitter_url(emoji: str):
    return f"https://raw.githubusercontent.com/twitter/twemoji/gh-pages/v/14.0.2/svg/{emoji.lower()}.svg"


def download_twitter(format_list: List[str], directory: str):
    download_format(format_list, create_twitter_url, directory)


def download_emoji(url_lambda: Callable[[str], str], emoji: str, directory: str, extension: str):
    save_location = f"{directory}{emoji}.{extension}"
    if os.path.exists(save_location) and os.path.isfile(save_location):
        return
    from urllib.error import URLError
    try:
        url = url_lambda(emoji)
        print(f"Downloading {url} -> {save_location}")
        req = request.urlopen(url)
        if req.getcode() == 200:
            open(save_location, "wb").write(req.read())
        else:
            print(f"Couldn't download {req.geturl()}, reason: {req.info()}")
    except URLError as e:
        print(print(f"Couldn't download {url}, reason: {e.reason}"))


def download_format(format_list: List[str], url_lambda: Callable[[str], str], directory: str, extension: str = None):
    os.makedirs(directory, exist_ok=True)
    if extension is None:
        extension = os.path.splitext(url_lambda("undefined"))[1][1:]
    with Pool() as p:
        p.starmap(download_emoji, [(url_lambda, emoji, directory, extension) for emoji in format_list])
        p.close()
        p.join()


def convert_svg_to_png(src: str, dest: str):
    os.makedirs(dest, exist_ok=True)
    for file in os.listdir(src):
        source_file = f"{src}{file}"
        destination_file = f"{dest}{os.path.splitext(file)[0]}.png"
        if os.path.exists(destination_file) and os.path.isfile(destination_file):
            continue
        print(f"Converting: {source_file} -> {destination_file}")
        subprocess.call(["../convert.exe" if sys.platform == "win32" else "convert",
                         "-background", "None",
                         "-size", "128x128",
                         source_file, destination_file],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def convert_png_to_bw(src: str, dest: str):
    os.makedirs(dest, exist_ok=True)
    for file in os.listdir(src):
        source_file = f"{src}{file}"
        destination_file = f"{dest}{os.path.splitext(file)[0]}.png"
        if os.path.exists(destination_file) and os.path.isfile(destination_file):
            continue
        print(f"Converting: {source_file} -> {destination_file}")
        subprocess.call(["../convert.exe" if sys.platform == "win32" else "convert",
                         "-size", "128x128",
                         source_file,
                         "-background", "white",
                         "-alpha", "remove",
                         "-colorspace", "gray",
                         "-depth", "8",
                         "-type", "grayscale",
                         destination_file],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def convert_jpg_to_bw(src: str, dest: str, quality: int):
    os.makedirs(dest, exist_ok=True)
    for file in os.listdir(src):
        source_file = f"{src}{file}"
        destination_file = f"{dest}{os.path.splitext(file)[0]}.jpg"
        if os.path.exists(destination_file) and os.path.isfile(destination_file):
            continue
        print(f"Converting: {source_file} -> {destination_file}")
        subprocess.call(["../convert.exe" if sys.platform == "win32" else "convert",
                         "-size", "128x128",
                         source_file,
                         "-background", "white",
                         "-alpha", "remove",
                         "-colorspace", "gray",
                         "-depth", "8",
                         "-type", "grayscale",
                         "-quality", f"{quality}",
                         destination_file],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    emoji_list = [
        # face-smiling
        "1F600",
        "1F603",
        "1F604",
        "1F601",
        "1F606",
        "1F605",
        "1F923",
        "1F602",
        "1F642",
        "1F643",
        "1FAE0",
        "1F609",
        "1F60A",
        "1F607",
        # face-affection
        "1F970",
        "1F60D",
        "1F929",
        "1F618",
        "1F617",
        "263A",
        "1F61A",
        "1F619",
        "1F972",
        # face-tongue
        "1F60B",
        "1F61B",
        "1F61C",
        "1F92A",
        "1F61D",
        "1F911",
        # face-hand
        "1F917",
        "1F92D",
        "1FAE2",
        "1FAE3",
        "1F92B",
        "1F914",
        "1FAE1",
        # face-neutral-skeptical
        "1F910",
        "1F928",
        "1F610",
        "1F611",
        "1F636",
        "1FAE5",
        "1F636",
        "1F60F",
        "1F612",
        "1F644",
        "1F62C",
        "1F62E",
        "1F925",
        "1FAE8",
        "1F642",
        "1F642",
        # face-sleepy
        "1F60C",
        "1F614",
        "1F62A",
        "1F924",
        "1F634",
        "1FAE9",
        # face-unwell
        "1F637",
        "1F912",
        "1F915",
        "1F922",
        "1F92E",
        "1F927",
        "1F975",
        "1F976",
        "1F974",
        "1F635",
        "1F635",
        "1F92F",
        # face-hat
        "1F920",
        "1F973",
        "1F978",
        # face-glasses
        "1F60E",
        "1F913",
        "1F9D0",
        # face-concerned
        "1F615",
        "1FAE4",
        "1F61F",
        "1F641",
        "2639",
        "1F62E",
        "1F62F",
        "1F632",
        "1F633",
        "1FAEA",
        "1F97A",
        "1F979",
        "1F626",
        "1F627",
        "1F628",
        "1F630",
        "1F625",
        "1F622",
        "1F62D",
        "1F631",
        "1F616",
        "1F623",
        "1F61E",
        "1F613",
        "1F629",
        "1F62B",
        "1F971",
        # face-negative
        "1F624",
        "1F621",
        "1F620",
        "1F92C",
        "1F608",
        "1F47F",
        "1F480",
        "2620",
        # face-costume
        "1F4A9",
        "1F921",
        "1F479",
        "1F47A",
        "1F47B",
        "1F47D",
        "1F47E",
        "1F916",
        # cat-face
        "1F63A",
        "1F638",
        "1F639",
        "1F63B",
        "1F63C",
        "1F63D",
        "1F640",
        "1F63F",
        "1F63E",
        # monkey-face
        "1F648",
        "1F649",
        "1F64A",
    ]
    download_twitter(emoji_list, "../emojis/twemoji/svg/")
    convert_svg_to_png("../emojis/twemoji/svg/", "../emojis/twemoji/png/")
    convert_png_to_bw("../emojis/twemoji/png/", "../emojis/twemoji/png_bw/")
    convert_jpg_to_bw("../emojis/twemoji/png/", "../emojis/twemoji/jpg/", quality=20)
    convert_jpg_to_bw("../emojis/twemoji/png/", "../emojis/twemoji/jpg_1/", quality=1)
