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

    dataset = dataset.sort(["text"])

    def process(x: dict[str, list]) -> dict[str, list]:
        text_set = set()
        indices, texts = [], []
        for idx, text in zip(x["id"], x["text"]):
            if text not in text_set:
                text_set.add(text)
                indices.append(idx)
                texts.append(text)

        return {
            "id": indices,
            "text": texts,
            "model": [model] * len(indices),
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


dataset = ds.DatasetDict(
    {
        "gemma2-27b": load_dataset("./datasets/wiki_paraphrase/gemma2_27b", "gemma2-27b"),
        "gemma2-9b": load_dataset("./datasets/wiki_paraphrase/gemma2_9b", "gemma2-9b"),
    }
)

print(dataset)
dataset: ds.Dataset = ds.concatenate_datasets(list(dataset.values()))

model_names = list(sorted(set(dataset["model"])))
dataset = dataset.cast_column("model", ds.ClassLabel(names=model_names))

dataset = dataset.sort(["id", "model", "text"])

dataset.push_to_hub("hpprc/jawiki-paraphrases", "generated", private=True)
