import re


def pre_caption(caption, max_words):
    # remove unwanted punctuation
    caption = (
        re.sub(
            r"([,.'!?\"()*#:;~])",
            "",
            caption.lower(),
        )
        .replace("-", " ")
        .replace("/", " ")
        .replace("<person>", "person")
    )
    # remove extra spaces
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    # trim newline and extra spaces at ends
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")
    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])
    return caption
