import torch
from transformers import XLNetConfig, XLNetModel, XLNetTokenizer


def save_model(model_dir, saved_dir, n_labels, task, model_type='xlnet-base-cased'):
    """
    save checkpoints as pytorch model: model.ckpt.index
                                       model.ckpt.data-00000-of-00001
    """
    config = XLNetConfig.from_pretrained(
            model_type,
            num_labels=n_labels,
            finetuning_task=task,
            )
    model = XLNetModel.from_pretrained(
            model_dir,
            config=config,
            from_tf=True,
            )
    tokeniser = XLNetTokenizer.from_pretrained(
            model_dir,
            config=config,
            from_tf=True,
            )
    model.save_pretrained(saved_dir)
    # print model params
    for param in model.state_dict():
        print(param, "\t", model.state_dict()[param].size())


def get_model(model_dir, saved_dir, n_labels, task, model_type='xlnet-base-cased'):
    config = XLNetConfig.from_pretrained(
            model_type,
            num_labels=n_labels,
            finetuning_task=task,
            )
    model = XLNetModel(config=config)
    model.load_state_dict(torch.load(saved_dir+'/pytorch_model.bin'))
    model.eval() # set dropout and batch normalisation layers to evaluation
    tokeniser = XLNetTokenizer.from_pretrained(
            model_dir,
            config=config,
            from_tf=True,
            )
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
    return vectors
