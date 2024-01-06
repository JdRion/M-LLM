import pandas as pd
from datasets import Dataset

def preprocessing_BoolQA(dataset):
    dataset = dataset.map(lambda x:  {'text': f"""instruction: According to the following passage, is the question true or false? Choose one from the option.\n 
        sentence: {x['sentence']} \n 
        label: {x['label']}""" })
    return dataset

def preprocessing_CB(dataset):
    dataset = dataset.map(lambda x:  {'text': f"""What is the logical relationship between the \"sentence 1\" and the \"sentence 2\"? Choose one from the option.\n 
        sentence: {x['sentence']} \n 
        label: {x['label']}""" })

    return dataset

def preprocessing_SC(dataset):
    dataset = dataset.map(lambda x:  {'text': f""""What is the sentiment of the following paragraph? Choose one from the option.\n 
        sentence: {x['sentence']} \n 
        label: {x['label']}""" })

    return dataset

def preprocessing_dataset(dataset, dataset_name):
    if dataset_name == "BoolQA":
        return preprocessing_BoolQA(dataset)
    elif dataset_name == "CB":
        return preprocessing_CB(dataset)
    elif datasest_name == "SC":
        return preprocessing_SC(datasest)


def preprare_dataset(path):
    df = pd.read_csv(path)
    df = df.sample(2200, random_state=42)

    train_df = df[:2000]
    val_df = df[2000:2100]
    test_df = df[2100:2200]
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    if "BoolQA" in path:
        dataset_name = "BoolQA"
        train_dataset = preprocessing_dataset(train_dataset,dataset_name)
        val_dataset = preprocessing_dataset(val_dataset,dataset_name)
        test_dataset = preprocessing_dataset(test_dataset,dataset_name)
    elif "CB" in paht:
        dataset_name = "CB"
        train_dataset = preprocessing_dataset(train_dataset,dataset_name)
        val_dataset = preprocessing_dataset(val_dataset,dataset_name)
        test_dataset = preprocessing_dataset(test_dataset,dataset_name)
    elif "sc" in path:
        dataset_name = "SC"
        train_dataset = preprocessing_dataset(train_dataset,dataset_name)
        val_dataset = preprocessing_dataset(val_dataset,dataset_name)
        test_dataset = preprocessing_dataset(test_dataset,dataset_name)
    else:
        raise ValueError("Unknown dataset type")
    
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    preprare_dataset()



