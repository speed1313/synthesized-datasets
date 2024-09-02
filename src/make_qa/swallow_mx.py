import random
import uuid
from datetime import datetime
from pathlib import Path
import click

from vllm import LLM, SamplingParams

import datasets as ds


def make_input_text(passage: str, tokenizer) -> str:
    passage = passage.strip().replace("\n", "")
    prompt = f"""
文章に基づいて質問とそれに対する回答を作成してください。

## 一般的な指示
- 日本語Wikipediaの文章から質問と回答を作成してください
- 質問と回答は**必ず日本語で**出力してください、それ以外は出力してはいけません

## 質問についての指示
- 質問は**単純かつ簡潔**にしてください
- 質問はその質問単体で意味が伝わるものにしてください。**代名詞(この、それ、あれなど)を絶対に使わないでください**
- "What"や"Yes/No"を回答する質問をしてください、"How"についての質問は避けてください

## 回答についての指示
- 回答は**単純かつ簡潔**にしてください
- 回答は可能な限り1つの単語やフレーズを出力してください
- 回答は文章内に含まれるものにしてください
- 回答できない場合には「回答不能」と出力してください

文章: バリトン（独: Bariton〈バーリトン〉、英: baritone〈バリトーン〉、仏: baryton〈バリトン〉、伊: baritono〈バリートノ〉）は、男声のバスとテノールの中間の声域およびそれを受け持つ歌手。男声を音域で二分する場合はバスの側に分類される。 典型的なバリトンの音域は概ねG2～G4[1]、合唱ではA2～F4くらいである。記譜はバス記号が用いられることが多く、バリトン記号は現在はあまり用いられない。
質問: 男性の声の音域で、テノールとバスの間に当たるのは何でしょう？
回答: バリトン

文章: これに対し海底の電信は、電線を覆う絶縁物質に適した材料を選び出す必要があったので、陸路に比べて敷設が遅れていた。しかし、マレー半島の樹木から採れるガタパーチャと呼ばれる樹液が使用できるということが分かり、海底ケーブル敷設の道は開けた。初の海底ケーブルは、1850年、ブレット兄弟（兄John Watkins Brett、弟Jacob W. Brett）によりドーバー海峡に敷かれた。このケーブルは不慮の事故により翌日切断されたが、翌1851年に再び敷設され、英国とフランスをケーブルで結ぶことに成功した。
質問: 海底ケーブルが初めて結ばれたのはどこ？
回答: 英国とフランス

文章: {passage}
質問: """
    return prompt


def create_dataset(texts: list[str], llm: LLM, tokenizer, sampling_params) -> ds.Dataset:
    inputs_text = [make_input_text(t, tokenizer) for t in texts]

    responses = llm.generate(
        inputs_text,
        sampling_params=sampling_params,
    )
    output_texts: list[str] = [response.outputs[0].text.strip() for response in responses]

    queries = []
    answers = []

    for output_text in output_texts:
        query, answer = None, None

        try:
            query, answer, *_ = output_text.split("回答: ")
            query = query.strip().replace("質問: ", "", 1).strip()
            if "\n" in answer:
                answer = answer.split("\n")[0].strip()
            query, answer = query.strip().replace("\n", ""), answer.strip().replace("\n", "")

        except Exception:
            pass

        queries.append(query)
        answers.append(answer)

    dataset = ds.Dataset.from_dict(
        {
            "query": queries,
            "answer": answers,
            "text": texts,
        }
    )

    def filter_fn(x: dict) -> bool:
        return (
            x["answer"] is not None
            and x["query"] is not None
            and isinstance(x["answer"], str)
            and isinstance(x["query"], str)
            and len(x["answer"]) > 0
            and len(x["query"]) > 0
            and len(x["answer"]) <= 100
            and len(x["query"]) <= 100
            and x["answer"] != x["query"]
            and x["answer"] not in x["query"]
            and x["query"] not in x["answer"]
        )

    dataset = dataset.filter(filter_fn)
    return dataset


@click.command()
@click.option("--dtype", type=str, default="bf16")
def main(dtype: str):
    model_name = "tokyotech-llm/Swallow-MX-8x7b-NVE-v0.1"
    root_dir = Path("datasets/wiki_qa/swallow_mx")
    batch_size = 10000
    max_file_size = 1_000_000

    if dtype == "bf16":
        dtype = "bfloat16"
        enable_prefix_caching = True
        enable_chunked_prefill = True
    elif dtype == "fp16":
        dtype = "float16"
        enable_prefix_caching = False
        enable_chunked_prefill = False
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    llm = LLM(
        model_name,
        trust_remote_code=True,
        tensor_parallel_size=4,
        quantization=None,
        dtype=dtype,
        gpu_memory_utilization=0.9,
        seed=int(datetime.now().timestamp()),
        enforce_eager=True,
        # we should disable prefix-caching and chunked prefilling to avoid errors
        # ref: https://github.com/vllm-project/vllm/issues/2729#issuecomment-2290484761
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
    )
    tokenizer = llm.get_tokenizer()

    dataset: ds.Dataset = ds.load_dataset("hpprc/emb", "qa-collection", split="train")
    dataset = dataset.filter(lambda x: 50 <= len(x["text"]) <= 1000)
    texts: list[str] = dataset["text"]

    print(f"Processing Dataset: {len(dataset)} samples")

    sampling_params = SamplingParams(
        temperature=0.99,
        top_p=0.95,
        max_tokens=128,
    )

    root_dir.mkdir(parents=True, exist_ok=True)

    idx = str(uuid.uuid4())
    save_path = root_dir / f"{idx}.jsonl"

    while True:
        try:
            sub_texts = random.sample(texts, batch_size)
            sub_dataset = create_dataset(sub_texts, llm, tokenizer, sampling_params)

            if save_path.exists():
                prev_dataset = ds.load_dataset("json", data_files=str(save_path), split="train")
                sub_dataset = ds.concatenate_datasets([prev_dataset, sub_dataset])

            sub_dataset = sub_dataset.select_columns(["query", "answer", "text"])
            sub_dataset.to_json(save_path, force_ascii=False, lines=True)

            if len(sub_dataset) >= max_file_size:
                idx = str(uuid.uuid4())
                save_path = root_dir / f"{idx}.jsonl"

        except Exception as e:
            with (root_dir / f"{idx}.log").open("a+") as f:
                f.write(f"{e}\n")
            continue


if __name__ == "__main__":
    main()
