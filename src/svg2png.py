import os
import subprocess
from multiprocessing.pool import Pool
from typing import List, Callable
from urllib import request


def create_google_url(emoji: str):
    return f"https://github.com/googlei18n/noto-emoji/raw/master/png/128/emoji_u{emoji.lower()}.png"


def download_google(format_list: List[str], directory: str):
    download_format(format_list, create_google_url, directory)


def create_twitter_url(emoji):
    return f"https://github.com/twitter/twemoji/raw/gh-pages/2/svg/{emoji.lower()}.svg"


def download_twitter(format_list: List[str], directory: str):
    download_format(format_list, create_twitter_url, directory)


def create_emojione_url(emoji: str):
    return f"https://api.emojione.com/emoji/{emoji.lower()}/download/128"


def download_emojione(format_list: List[str], directory: str):
    download_format(format_list, create_emojione_url, directory, "png")


def create_emojitwo_url(emoji):
    return f"https://github.com/EmojiTwo/emojitwo/raw/master/png/128/{emoji.lower()}.png"


def download_emojitwo(format_list: List[str], directory: str):
    download_format(format_list, create_emojitwo_url, directory)


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
        subprocess.call(["../convert.exe",
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
        subprocess.call(["../convert.exe",
                         "-size", "128x128",
                         source_file,
                         "-background", "white",
                         "-alpha", "remove",
                         "-colorspace", "gray",
                         "-depth", "8",
                         "-type", "grayscale",
                         destination_file],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    emoji_list = [
        # subgroup: face-positive
        "1F600",
        "1F601",
        "1F602",
        "1F923",
        "1F603",
        "1F604",
        "1F605",
        "1F606",
        "1F609",
        "1F60A",
        "1F60B",
        "1F60E",
        "1F60D",
        "1F618",
        "1F970",
        "1F617",
        "1F619",
        "1F61A",
        "1F642",
        "1F917",
        "1F929",
        # subgroup: face-neutral
        "1F914",
        "1F928",
        "1F610",
        "1F611",
        "1F636",
        "1F644",
        "1F60F",
        "1F623",
        "1F625",
        "1F62E",
        "1F910",
        "1F62F",
        "1F62A",
        "1F62B",
        "1F634",
        "1F60C",
        "1F61B",
        "1F61C",
        "1F61D",
        "1F924",
        "1F612",
        "1F613",
        "1F614",
        "1F615",
        "1F643",
        "1F911",
        "1F632",
        # subgroup: face-negative
        "2639",
        "1F641",
        "1F616",
        "1F61E",
        "1F61F",
        "1F624",
        "1F622",
        "1F62D",
        "1F626",
        "1F627",
        "1F628",
        "1F629",
        # "1F92F",
        "1F62C",
        "1F630",
        "1F631",
        # "1F975",
        # "1F976",
        "1F633",
        "1F92A",
        "1F635",
        "1F621",
        "1F620",
        # "1F92C",
        # subgroup: face-sick
        "1F637",
        "1F912",
        "1F915",
        # "1F922",
        "1F92E",
        "1F927",
        # subgroup: face-role
        "1F607",
        "1F920",
        "1F973",
        "1F974",
        "1F97A",
        "1F925",
        "1F92B",
        "1F92D",
        "1F9D0",
        "1F913",
    ]
    download_google(emoji_list, "../emojis/google-noto/png/")
    download_twitter(emoji_list, "../emojis/twemoji/svg/")
    download_emojione(emoji_list, "../emojis/emojione/png/")
    # download_emojitwo(emoji_list, "../emojis/emojitwo/png/")
    convert_svg_to_png("../emojis/twemoji/svg/", "../emojis/twemoji/png/")
    convert_png_to_bw("../emojis/twemoji/png/", "../emojis/twemoji/png_bw/")
