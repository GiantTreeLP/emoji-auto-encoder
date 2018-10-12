import os
from typing import List, Callable
from urllib import request


def download_google(format_list: List[str], directory: str):
    download_format(format_list,
                    lambda emoji: f"https://raw.githubusercontent.com/googlei18n/noto-emoji/"
                                  f"master/svg/emoji_u{emoji.lower()}.svg",
                    directory)


def download_twitter(format_list: List[str], directory: str):
    download_format(format_list,
                    lambda emoji: f"https://raw.githubusercontent.com/twitter/twemoji/"
                                  f"gh-pages/svg/{emoji.lower()}.svg",
                    directory)


def download_emojitwo(format_list: List[str], directory: str):
    download_format(format_list,
                    lambda emoji: f"https://raw.githubusercontent.com/EmojiTwo/emojitwo/"
                                  f"master/svg/{emoji.lower()}.svg",
                    directory)


def download_format(format_list: List[str], url_lambda: Callable[[str], str], directory: str):
    os.makedirs(directory, exist_ok=True)
    for emoji in format_list:
        save_location = directory + emoji + ".svg"
        if os.path.exists(save_location) and os.path.isfile(save_location):
            continue
        from urllib.error import URLError
        try:
            url = url_lambda(emoji)
            req = request.urlopen(url)
            if req.getcode() == 200:
                open(save_location, "wb").write(req.read())
            else:
                print(f"Couldn't download {req.geturl()}, reason: {req.info()}")
        except URLError as e:
            print(print(f"Couldn't download {url}, reason: {e.reason}"))


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
    download_google(emoji_list, "../emojis/google-noto/svg/")
    download_twitter(emoji_list, "../emojis/twemoji/svg/")
    download_emojitwo(emoji_list, "../emojis/emojitwo/svg/")
