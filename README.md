# NAQANet PyTorch
[NAQANet](https://arxiv.org/pdf/1903.00161.pdf) PyTorch implementation.

## Answer abilities
The current implementation only handle *passage_span_extraction* and *counting* answer types.  
The logic for *addition-subtraction* is implemented but not tested.  
*question_span_extraction* has not been implemented.  

## Embeddings
We use GloVE word embeddings, while character embeddings are trained with the model.

## Dataset  
The model is trained on [DROP](https://arxiv.org/pdf/1903.00161.pdf) dataset.  

## Usage  
To train and evaluate the model on cuda device, run  
`python3 train_naqanet.py --use_gpu -g <device_id>`

Before training the model:  
1. Download DROP train and eval datasets and put them in *data/drop* folder.  
2. run `python3 setup_drop.py`

## Implementation differences
My current implementation reduce the number of layers in the stack of encoders from 6 to 1, in order to avoid memory issues on the GPU

## Performance
The current implementation with batch size 4, epochs 30 reach:
F1 = 34.02, EM = 30.53

On both metrics around 13 points are lost w.r.t. to the results on paper. This can be explained by the following reasons:

* 2 out of 4 answer abilities have been removed
* Reduced number of layers from 6 to 1 in the stack of encoders before the output layer
* I might have missed something in the implementation  
  
Batch size is reduced to 4 to avoid memory issues with the GPU. Paper reccomends this value equals 16. All other hyperparameters are kept as suggested in the original paper  

## Aknowledgments
The QANet part is mainly based on the following repositories:  
* https://github.com/andy840314/QANet-pytorch-
* https://github.com/BangLiu/QANet-PyTorch
* https://github.com/heliumsea/QANet-pytorch
