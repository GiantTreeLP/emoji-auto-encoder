import os
from multiprocessing.pool import Pool
from typing import List, Callable
from urllib import request


def create_google_url(emoji: str):
    return f"https://github.com/googlei18n/noto-emoji/raw/master/png/128/emoji_u{emoji.lower()}.png"


def download_google(format_list: List[str], directory: str):
    download_format(format_list,
                    create_google_url,
                    directory)


def create_twitter_url(emoji):
    return f"https://github.com/twitter/twemoji/raw/gh-pages/2/svg/{emoji.lower()}.svg"


def download_twitter(format_list: List[str], directory: str):
    download_format(format_list,
                    create_twitter_url,
                    directory)


def create_emojitwo_url(emoji):
    return f"https://github.com/EmojiTwo/emojitwo/raw/master/png/128/{emoji.lower()}.png"


def download_emojitwo(format_list: List[str], directory: str):
    download_format(format_list,
                    create_emojitwo_url,
                    directory)


def download_emoji(url_lambda: Callable[[str], str], emoji: str, directory: str, extension: str = None):
    if extension is None:
        extension = os.path.splitext(url_lambda("undefined"))[1][1:]
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
    with Pool() as p:
        p.starmap(download_emoji, [(url_lambda, emoji, directory, extension) for emoji in format_list])
        p.close()
        p.join()


if __name__ == '__main__':
    emoji_list = [
        "1f600",
        "1f601",
        "1f602",
        "1f603",
        "1f604",
        "1f605",
        "1f606",
        "1f609",
        "1f642",
        "1f643",
        "1f923",
    ]
    download_google(emoji_list, "../emojis/google-noto/png/")
    download_twitter(emoji_list, "../emojis/twemoji/svg/")
    download_emojitwo(emoji_list, "../emojis/emojitwo/png/")
