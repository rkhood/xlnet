# XLNet

Using Google's [XLNet](https://arxiv.org/abs/1906.08237?source=techstories.org) transformer model to create contextualised sentence embeddings.  Two methods utilising XLNet are included:

1. Using [Hugging Face's](https://github.com/huggingface/transformers) transformer package, and a pre-trained model:
	
	`pip install transformers`
2. Fine-tuning the model and converting checkpoints to a PyTorch model 
