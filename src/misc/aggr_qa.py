import datasets as ds


def load_dataset(data_dir: str, model: str):
    dataset = ds.load_dataset(
        "json",
        data_dir=data_dir,
        num_proc=16,
        split="train",
    )

    dataset = dataset.map(
        lambda x: {
            "length": len(x["text"]),
            "model": model,
        },
        num_proc=16,
    )
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
    dataset = ds.load_dataset("hpprc/auto-wiki-qa-nemotron", split="train")
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
        "tanuki-8x8b": load_dataset("../llm-translator-tanuki/datasets/wiki_qa_tanuki", "tanuki-8x8b"),
        "swallow-mx-beam3": load_auto_wiki_qa(),
        "nemotron-4-340b": load_auto_wiki_qa_nemotron(),
        "swallow-mx": load_dataset("../llm-translator/datasets/wiki_qa_noisy", "swallow-mx"),
        "sarashina2-70b": load_dataset("../llm-translator/datasets/wiki_qa_noisy_sarashina", "sarashina2-70b"),
        "sarashina2-13b": load_dataset("../llm-translator/datasets/wiki_qa_noisy_sarashina_13b", "sarashina2-13b"),
        "swallow-mx-2": load_dataset("./datasets/wiki_qa/swallow_mx", "swallow-mx"),
        "gemma2-9b": load_dataset("./datasets/wiki_qa/gemma2_9b", "gemma2-9b"),
        "gemma2-27b": load_dataset("./datasets/wiki_qa/gemma2_27b", "gemma2-27b"),
        "sarashina2-70b-2": load_dataset("./datasets/wiki_qa/sarashina2_70b", "sarashina2-70b"),
    }
)
print(dataset)
dataset: ds.Dataset = ds.concatenate_datasets(list(dataset.values()))
dataset = dataset.filter(lambda x: x["answer"] in x["text"], num_proc=16)

model_names = list(sorted(set(dataset["model"])))
dataset = dataset.cast_column("model", ds.ClassLabel(names=model_names))

dataset = dataset.sort(["length", "text", "model"])
dataset = dataset.remove_columns(["length"])

dataset.push_to_hub("hpprc/auto-wiki-qa-noisy", private=True)
