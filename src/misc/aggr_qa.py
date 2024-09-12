from pathlib import Path

import datasets as ds


def load_dataset(data_dir: str, model: str):
    data_files = [str(path) for path in Path(data_dir).glob("*.jsonl")]

    for _ in range(10):
        try:
            dataset = ds.load_dataset(
                "json",
                data_files=data_files,
                num_proc=16,
                split="train",
            )
            break
        except Exception:
            continue
    else:
        raise Exception(f"Failed to load dataset: {model}")

    dataset = dataset.select_columns(
        [
            "query",
            "answer",
            "text",
        ]
    )
    dataset = dataset.filter(lambda x: 50 <= len(x["text"]) <= 1000, num_proc=16)
    dataset = dataset.sort(["text", "query"])

    def process(x: dict[str, list]) -> dict[str, list]:
        query_texts = set()
        queries, answers, texts = [], [], []
        for query, answer, text in zip(x["query"], x["answer"], x["text"]):
            if ((query, text) not in query_texts) and (answer in text):
                query_texts.add((query, text))
                queries.append(query)
                texts.append(text)
                answers.append(answer)

        return {
            "query": queries,
            "text": texts,
            "answer": answers,
            "length": [len(text) for text in texts],
            "model": [model] * len(queries),
        }

    dataset = dataset.map(
        process,
        num_proc=16,
        batched=True,
        batch_size=10000,
        remove_columns=dataset.column_names,
    )

    print(model, data_dir)
    print(dataset)
    return dataset


def load_auto_wiki_qa():
    dataset = ds.load_dataset("cl-nagoya/auto-wiki-qa", split="train")
    dataset = dataset.select_columns(["query", "answer", "title", "text"])

    def process(x):
        text = x["title"].strip() + " " + x["text"].replace("\n", "")
        return {
            "text": text,
            "model": "swallow-mx-beam3",
            "length": len(text),
        }

    dataset = dataset.map(process, num_proc=16)
    dataset = dataset.select_columns(
        [
            "query",
            "answer",
            "text",
            "model",
            "length",
        ]
    )
    dataset = dataset.filter(lambda x: 50 <= len(x["text"]) <= 1000, num_proc=16)
    print("auto-wiki-qa")
    print(dataset)
    return dataset


def load_auto_wiki_qa_nemotron():
    dataset = ds.load_dataset("cl-nagoya/auto-wiki-qa-nemotron", split="train")
    dataset = dataset.select_columns(["query", "answer", "title", "text"])

    def process(x):
        text = x["title"].strip() + " " + x["text"].replace("\n", "")
        return {
            "text": text,
            "model": "nemotron-4-340b",
            "length": len(text),
        }

    dataset = dataset.map(process, num_proc=16)
    dataset = dataset.select_columns(
        [
            "query",
            "answer",
            "text",
            "model",
            "length",
        ]
    )
    dataset = dataset.filter(lambda x: 50 <= len(x["text"]) <= 1000, num_proc=16)
    print("auto-wiki-qa-nemotron")
    print(dataset)
    return dataset


dataset = ds.DatasetDict(
    {
        "gemma2-9b": load_dataset("./datasets/wiki_qa/gemma2_9b", "gemma2-9b"),
        "gemma2-27b": load_dataset("./datasets/wiki_qa/gemma2_27b", "gemma2-27b"),
        "phi3.5-mini": load_dataset("./datasets/wiki_qa/phi3_5_mini", "phi3.5-mini"),
        "tanuki-8x8b": load_dataset("../llm-translator-tanuki/datasets/wiki_qa_tanuki", "tanuki-8x8b"),
        "swallow-mx-beam3": load_auto_wiki_qa(),
        "nemotron-4-340b": load_auto_wiki_qa_nemotron(),
        "swallow-mx": load_dataset("../llm-translator/datasets/wiki_qa_noisy", "swallow-mx"),
        "sarashina2-70b": load_dataset("../llm-translator/datasets/wiki_qa_noisy_sarashina", "sarashina2-70b"),
        "sarashina2-13b": load_dataset("../llm-translator/datasets/wiki_qa_noisy_sarashina_13b", "sarashina2-13b"),
        "swallow-mx-2": load_dataset("./datasets/wiki_qa/swallow_mx", "swallow-mx"),
        "sarashina2-70b-2": load_dataset("./datasets/wiki_qa/sarashina2_70b", "sarashina2-70b"),
    }
)
print(dataset)
dataset: ds.Dataset = ds.concatenate_datasets(list(dataset.values()))
dataset = dataset.filter(lambda x: x["answer"] in x["text"], num_proc=16)


def count_kono(text: str) -> int:
    return text.count("この") - text.count("どこの")


dataset = dataset.filter(lambda x: count_kono(x["query"]) <= 0, num_proc=16)

ng_phrases = ["この文章", "下記の"]
dataset = dataset.filter(lambda x: all(phrase not in x["query"] for phrase in ng_phrases), num_proc=16)

model_names = list(sorted(set(dataset["model"])))
dataset = dataset.cast_column("model", ds.ClassLabel(names=model_names))

dataset = dataset.sort(["length", "text", "model"])
dataset = dataset.remove_columns(["length"])
dataset = dataset.select_columns(["query", "answer", "text", "model"])

dataset.push_to_hub("hpprc/auto-wiki-qa-noisy", private=True)
