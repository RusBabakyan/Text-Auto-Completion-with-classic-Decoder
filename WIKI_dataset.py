from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
import nltk
nltk.download('punkt')
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, data, tok, max_length = 30):
        self.data = data
        self.tok = tok
        self.max_length = max_length

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx]

        data_tokens = self.tok(data, return_tensors = 'pt', max_length = self.max_length, truncation=True, padding = 'max_length')

        data_input_ids = data_tokens['input_ids'][0]
        data_attention_mask = data_tokens['attention_mask'][0]

        input_ids = data_input_ids[:-1]
        labels_ids = data_input_ids[1:]

        return input_ids, labels_ids, data_attention_mask[1:].bool()
    
def WikipediaDataset(path="wikipedia", name="20220301.simple", split=0.2):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(path, name)['train']['text']
    VOCAB_SIZE = len(tokenizer)

    if split:
        train_data, test_data = train_test_split(dataset, test_size=split, random_state=42)
        return (train_data, test_data), tokenizer, VOCAB_SIZE
    
    return dataset, tokenizer, VOCAB_SIZE


    # def __init__(self, path="wikipedia", name="20220301.simple"):
    #     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    #     tokenizer.pad_token = tokenizer.eos_token
    #     dataset = load_dataset(path, name)
    #     self.VOCAB_SIZE = len(tokenizer)
    #     super().__init__(dataset['train']['text'], tokenizer)
        


