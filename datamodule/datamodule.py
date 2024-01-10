import pandas as pd
from datasets import Dataset

def preprocessing_BoolQA(dataset):
    dataset = dataset.map(lambda x:  {'text': f"""instruction: According to the following passage, is the question true or false? Choose one from the option.\n 
        sentence: {x['sentence']} \n 
        label: {x['label']}""" })
    return dataset

def preprocessing_NLI(dataset):
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
    elif dataset_name == "NLI":
        return preprocessing_NLI(dataset)
    elif dataset_name == "SC":
        return preprocessing_SC(dataset)

def map_tokenizer(dataset,tokenizer):
    text_data = {'text': dataset['text']}
    dataset = Dataset.from_dict(text_data)
    dataset = dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)      

    return dataset  

def preprare_train_dataset(paths, tokenizer):
    for name,path in paths.items():
        df = pd.read_csv(path)
        df = df.sample(2200, random_state=42)
        train_df = df[:2000]
        train_dataset = Dataset.from_pandas(train_df)

        if "BoolQA" in path:
            dataset_name = "BoolQA"
            f_train_dataset = preprocessing_dataset(train_dataset,dataset_name)
            f_train_dataset = map_tokenizer(f_train_dataset, tokenizer)
        elif "NLI" in path:
            dataset_name = "NLI"
            s_train_dataset = preprocessing_dataset(train_dataset,dataset_name)
            s_train_dataset = map_tokenizer(s_train_dataset, tokenizer)
        elif "sc" in path:
            dataset_name = "SC"
            t_train_dataset = preprocessing_dataset(train_dataset,dataset_name)
            t_train_dataset = map_tokenizer(t_train_dataset, tokenizer)
        else:
            raise ValueError("Unknown dataset type")
    
    return f_train_dataset, s_train_dataset, t_train_dataset

def prepare_test_dataset(paths):
    for name,path in paths.items():
            df = pd.read_csv(path)
            df = df.sample(2200, random_state=42)
            train_df = df[2000:2100]            
            train_dataset = Dataset.from_pandas(train_df)

            if "BoolQA" in path:
                dataset_name = "BoolQA"
                f_test_dataset = preprocessing_dataset(train_dataset,dataset_name)
            elif "NLI" in path:
                dataset_name = "NLI"
                s_test_dataset = preprocessing_dataset(train_dataset,dataset_name)
            elif "sc" in path:
                dataset_name = "SC"
                t_test_dataset = preprocessing_dataset(train_dataset,dataset_name)
            else:
                raise ValueError("Unknown dataset type")
        
    return [f_test_dataset, s_test_dataset, t_test_dataset]

if __name__ == "__main__":
    preprare_train_dataset()



