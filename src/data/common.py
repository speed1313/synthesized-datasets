import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    text = "".join(char for char in text if char.isprintable())
    text = text.replace("。 ", "。")
    text = text.strip()
    return text
