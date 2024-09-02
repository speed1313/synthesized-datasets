import random
import uuid
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams

import datasets as ds


def make_input_text(passage: str, tokenizer) -> str:
    passage = passage.strip().replace("\n", "")
    prompt = f"""文章: バリトン（独: Bariton〈バーリトン〉、英: baritone〈バリトーン〉、仏: baryton〈バリトン〉、伊: baritono〈バリートノ〉）は、男声のバスとテノールの中間の声域およびそれを受け持つ歌手。男声を音域で二分する場合はバスの側に分類される。 典型的なバリトンの音域は概ねG2～G4[1]、合唱ではA2～F4くらいである。記譜はバス記号が用いられることが多く、バリトン記号は現在はあまり用いられない。
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

    sampling_params = SamplingParams(
        temperature=0.99,
        top_p=0.95,
        max_tokens=256,
    )

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
            query = query.strip().replace("質問:", "", 1).strip()
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


def main():
    model_name = "sbintuitions/sarashina2-13b"
    root_dir = Path("datasets/wiki_qa/sarashina2_13b")
    batch_size = 10000
    max_file_size = 1_000_000

    llm = LLM(
        model_name,
        trust_remote_code=True,
        tensor_parallel_size=4,
        quantization=None,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        seed=int(datetime.now().timestamp()),
        enforce_eager=True,
        use_v2_block_manager=False,
        enable_prefix_caching=True,
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
