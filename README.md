# Embedding Graphs on a Grassmann Manifold

This repository is the official implementation of [Embedding Graphs on a Grassmann Manifold](submitted to ECML-PKDD 2021).

## Requirements

To install requirements:

```
pip install -r requirements.txt
```

## Graph Classification
To reproduce the results in Table 1 of the main text, you can use the following command:

```
python main.py
```
Other hyperparameters include: --dataset, --lr, --wd, --conv_hid --fc_dim, --s, --drop_ratio, --pRatio.

## Node Clustering
To reproduce the results in Table 3 of the main text, you can use the following command:

```
python train_vae.py
```
The default arguments are for reproducing the results of Pubmed.

## Contributing
Copyright (c) <2020> <NeurIPS>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
