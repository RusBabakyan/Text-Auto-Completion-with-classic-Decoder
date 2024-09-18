import torch
from decoder_models import TransformerDecoder
from transformers import GPT2Tokenizer

# small:
    # D_MODEL = 512
    # N_HEADS = 8
    # N_BLOCKS = 3

# big 
    # D_MODEL = 800
    # N_HEADS = 10
    # N_BLOCKS = 6
class PredictionModel():
    D_MODEL = 512
    N_HEADS = 8
    N_BLOCKS = 3
    
    def __init__(self, path, max_length=1):
        self.max_length = max_length
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        VOCAB_SIZE = len(self.tokenizer)

        self.model = TransformerDecoder(VOCAB_SIZE, self.D_MODEL, self.N_HEADS, self.N_BLOCKS)
        self.model.load_state_dict(torch.load(path, weights_only=False))
        self.model.to(self.DEVICE)
        
    def __call__(self, x):
        # x = x.rstrip(' ')
        input_ids = self.__tokenize(x)
        prediction = self.__generate_batch(input_ids, is_decode=True)
        if x[-1] == ' ':
            return prediction[len(x)+1:]
        return prediction[len(x):].replace('\n', '')
        

    def __tokenize(self, data, max_length=30):
        # data_tokens = self.tokenizer(data, return_tensors = 'pt', max_length = self.max_length, truncation=True)
        data_tokens = self.tokenizer(data, return_tensors = 'pt')
        data_input_ids = data_tokens['input_ids'][0]
        return data_input_ids

    @torch.no_grad()
    def __generate_batch(self, ids, is_decode = False):

        self.model.eval()

        if not isinstance(ids, torch.Tensor):
            ids =  torch.tensor(ids, device = self.DEVICE)
            
        if ids.shape:
            symbols = ids.detach().to(self.DEVICE).view(1,len(ids))

        else:
            symbols = ids.detach().to(self.DEVICE).view(1,1)

        for _ in range(self.max_length):
            cur_symbols = self.model(symbols).argmax(-1)[:, -1][:, None]
            symbols = torch.cat([symbols, cur_symbols], dim = 1)
            
        if is_decode:
            symbols = self.tokenizer.batch_decode(symbols, skip_special_tokens=True)
        return symbols[0]