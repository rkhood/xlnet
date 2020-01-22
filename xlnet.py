import torch
from transformers import XLNetModel, XLNetTokenizer


def get_model(output_dir):
    '''
    output_dir: path to hugging face directory
    '''
    model = XLNetModel.from_pretrained(output_dir)
    tokeniser = XLNetTokenizer.from_pretrained(output_dir)
    return model, tokeniser


def get_vectors(data, model, tokeniser):
    vectors = []
    for text in data:
        input_ids = torch.tensor(
                [tokeniser.encode(text, add_special_tokens=True)])

        # cls token
        with torch.no_grad():
            vectors.append(
                    model(input_ids)[0].squeeze().tolist()[0])
