import random
import uuid
from pathlib import Path

import click
from vllm import LLM, SamplingParams

import datasets as ds
from src.data.common import normalize_text


def make_input_text(passage: str, tokenizer) -> str:
    passage = passage.strip().replace("\n", "")
    messages = [
        {
            "role": "user",
            "content": f"""
あなたは卓越した日本語話者です。以下の指示に従い、与えられた日本語Wikipediaの文章を言い換えてください。

### 指示
1. 日本語Wikipediaの文章を同じ意味の別の文章に言い換えてください
2. 言い換えた文章は元の文章と異なるものである必要があります。ただし、意味が全く変化しないようにしてください
3. 語順や文の順番を変えても構いませんが、内容を追加したり削除したりすることはできません
4. 文章全体で内容と意味が変化しないのであれば、文の順番をできるだけ入れ替えてください。できるだけ言い換えてください
5. 言い換えた文章のみを出力してください

文章: {passage}
""".strip(),
        },
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return input_text


def create_dataset(texts: list[str], llm: LLM, tokenizer) -> ds.Dataset:
    sampling_params = SamplingParams(
        temperature=0.99,
        top_p=0.95,
        max_tokens=8192,
    )

    inputs_text = [make_input_text(t, tokenizer) for t in texts]

    responses = llm.generate(
        inputs_text,
        sampling_params=sampling_params,
    )
    output_texts: list[str] = [response.outputs[0].text.strip() for response in responses]
    output_texts = [normalize_text(t) for t in output_texts]

    return output_texts


@click.command()
@click.option("--dtype", type=str, default="bf16")
@click.option("--tp", type=int, default=4)
def main(dtype: str, tp: int):
    model_name = "google/gemma-2-27b-it"
    root_dir = Path("datasets/wiki_paraphrase2/gemma2_27b")

    batch_size = 10000
    # batch_size = 100
    max_file_size = 1_000_000

    if dtype == "bf16":
        dtype = "bfloat16"
        enable_prefix_caching = True
    elif dtype == "fp16":
        dtype = "float16"
        enable_prefix_caching = False
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    rng = random.SystemRandom()
    seed = rng.randint(0, 2**32 - 1)

    llm = LLM(
        model_name,
        trust_remote_code=True,
        tensor_parallel_size=tp,
        quantization=None,
        dtype=dtype,
        gpu_memory_utilization=0.95,
        seed=seed,
        enforce_eager=False,
        use_v2_block_manager=False,
        enable_prefix_caching=enable_prefix_caching,
    )
    tokenizer = llm.get_tokenizer()

    dataset: ds.Dataset = ds.load_dataset("hpprc/jawiki-paraphrases2", "collection", split="train")
    texts: list[str] = dataset["text"]
    indices = [idx for idx in range(len(texts))]

    print(f"Processing Dataset: {len(indices)} / {len(dataset)} samples")

    root_dir.mkdir(parents=True, exist_ok=True)

    idx = str(uuid.uuid4())
    save_path = root_dir / f"{idx}.jsonl"

    while True:
        try:
            sub_indices = random.sample(indices, batch_size)
            sub_texts = [texts[i] for i in sub_indices]
            output_texts = create_dataset(sub_texts, llm, tokenizer)

            sub_dataset = ds.Dataset.from_dict(
                {
                    "id": sub_indices,
                    "text": output_texts,
                }
            )

            if save_path.exists():
                prev_dataset = ds.load_dataset("json", data_files=str(save_path), split="train")
                sub_dataset = ds.concatenate_datasets([prev_dataset, sub_dataset])

            sub_dataset = sub_dataset.select_columns(["id", "text"])
            sub_dataset.to_json(save_path, force_ascii=False, lines=True)

            if len(sub_dataset) >= max_file_size:
                idx = str(uuid.uuid4())
                save_path = root_dir / f"{idx}.jsonl"

        except Exception as e:
            print(e)
            with (root_dir / f"{idx}.log").open("a+") as f:
                f.write(f"{e}\n")
            continue


if __name__ == "__main__":
    main()
