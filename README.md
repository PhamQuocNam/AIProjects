# Aspect-based Sentiment Analysis

> Aspect Based Sentiment Analysis, PyTorch Implementations.

To install requirements, run `pip install -r requirements.txt`.

## Usage

### Training

```sh
python train.py --model_name bert_model --dataset laptop --epochs 50
```

### Inference
In the [infer.py](./infer.py), please enter your input in the main() function, and run by write on the terminal:

```sh
python infer.py --model_name bert_model
```

* Refer to [infer_example.py](./infer_example.py) for both non-BERT-based models and BERT-based models.


### Tips

* For non-BERT-based models, training procedure is not very stable.
* BERT-based models are more sensitive to hyperparameters (especially learning rate) on small data sets
* Fine-tuning on the specific task is necessary for releasing the true power of BERT.




