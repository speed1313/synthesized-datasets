import datasets as ds
from src.data.common import normalize_text

jawiki: ds.Dataset = ds.load_dataset("hpprc/jawiki", split="train")


jawiki = jawiki.filter(
    lambda x: (not x["is_disambiguation_page"])
    and (not x["is_violent_page"])
    and (not x["is_sexual_page"])
    and ("一覧" not in x["title"]),
    num_proc=16,
)
jawiki = jawiki.select_columns(["id", "title", "paragraphs"])
jawiki = jawiki.add_column("source", ["jawiki" for _ in range(len(jawiki))])

books = ds.load_dataset("hpprc/jawiki-books", split="train")
books = books.select_columns(["id", "title", "paragraphs"])
books = books.add_column("source", ["books" for _ in range(len(books))])

# wiktionary = ds.load_dataset("hpprc/jawiki-wiktionary", split="train")
# wiktionary = wiktionary.select_columns(["id", "title", "paragraphs"])
# wiktionary = wiktionary.add_column("source", ["wiktionary" for _ in range(len(wiktionary))])

# dataset = ds.concatenate_datasets([jawiki, books, wiktionary])

dataset = ds.concatenate_datasets([jawiki, books])


def make_chunked_texts(texts: list[str]):
    current_text = ""
    output_texts = []
    for text in texts:
        current_text = (current_text + "\n" + text).strip()
        if len(current_text) >= 100:
            output_texts.append(normalize_text(current_text))
            current_text = ""
    return output_texts


def process(x: dict[str, list]) -> dict:
    data = []

    for passage_id, title, paragraphs, source in zip(x["id"], x["title"], x["paragraphs"], x["source"]):
        current_section_title = None
        current_texts = []

        for paragraph in paragraphs:
            if not (paragraph["title"] == current_section_title or paragraph["title"] is current_section_title):
                current_texts = [normalize_text(t) for t in current_texts]
                current_texts = [t for t in current_texts if t.strip() != ""]

                for text in make_chunked_texts(current_texts):
                    data.append(
                        {
                            "passage_id": passage_id,
                            "title": title,
                            "section_title": current_section_title,
                            "text": text,
                            "source": source,
                        }
                    )

                current_section_title = paragraph["title"]
                current_texts = [paragraph["text"]]
            else:
                current_texts.append(paragraph["text"])

        current_texts = [normalize_text(t) for t in current_texts]
        current_texts = [t for t in current_texts if t.strip() != ""]
        if len(current_texts) > 0:
            for text in make_chunked_texts(current_texts):
                data.append(
                    {
                        "passage_id": passage_id,
                        "title": title,
                        "section_title": current_section_title,
                        "text": text,
                        "source": source,
                    }
                )

    return {
        "passage_id": [d["passage_id"] for d in data],
        "title": [d["title"] for d in data],
        "section_title": [d["section_title"] for d in data],
        "text": [d["text"] for d in data],
        "source": [d["source"] for d in data],
    }


dataset = dataset.map(
    process,
    num_proc=16,
    batched=True,
    batch_size=4096,
    remove_columns=dataset.column_names,
)

dataset = dataset.select_columns(["source", "passage_id", "title", "section_title", "text"])

dataset.push_to_hub("hpprc/jawiki-paraphrases", "collection", private=True)
