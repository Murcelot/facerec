# Face recognition model with resistant to deepfake attacks
My model for Kryptonit hackaton.
Usually to defend from deepfake attacks while face recognition one use 2 models. First model to check photo for deepfake, second model used to make embedding.
In this hackaton it was suggested to make one model, that could be distinquish deepfakes photos of original photos.

Here you can see my model architecture in models/TranSSL.py
It is based on spherenet and transformer with positional encodings and tokenizer.

In train.py you can see training proccess. It has warmup scheduler, augmentations and label smoothing for more stable transformers learning.
