import pandas as pd
from datasets import Dataset, DatasetDict
import datasets
import json 
import os

default_path = "/home/elicer/O-LoRA/CL_Benchmark"
data_path = [["BoolQA/BoolQA","BoolQA"], ["NLI/CB","NLI_CB"], ["SC/amazon","sc_amazon"]]

def load_dataset_from_json(json_path):
    # JSON 파일을 pandas DataFrame으로 읽기
    df = pd.read_json(json_path, orient='records')
    return Dataset.from_pandas(df)

def push_dataset():
    # 데이터셋 로딩
    for d_path,name in data_path:
        path = default_path + "/" + d_path
        train_dataset = load_dataset_from_json(f"{path}/train.json")
        dev_dataset = load_dataset_from_json(f"{path}/dev.json")
        test_dataset = load_dataset_from_json(f"{path}/test.json")

        # 데이터셋 결합
        dataset = DatasetDict({
            "train": train_dataset,
            "validation": dev_dataset,
            "test": test_dataset
        })

        dataset.save_to_disk(f"./data/{name}")


if __name__ == "__main__":
    push_dataset()

