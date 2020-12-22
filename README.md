# PyTorch-NAQANet
PyTorch implementation of [NAQANet](https://arxiv.org/pdf/1903.00161.pdf).

## Answer abilities
The current implementation only handle *passage_span_extraction* and *counting* answer types.  
The logic for *addition-subtraction* is implemented but not tested.  
*question_span_extraction* has not been implemented.  

## Embeddings
We use pretrained GloVE word embeddings, while character embedding are trained with the model.

## Dataset  
The model is trained on [DROP](https://arxiv.org/pdf/1903.00161.pdf) dataset.  

## Usage  
To train the model on cuda device, run  
`python3 train_naqanet.py --use_gpu -g <device_id>`

Before training the model:  
1. Downloaded DROP train and eval datasets and put them in *data/drop* folder.  
2. run `python3 setup_drop.py`
