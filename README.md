# **Deep Clustering for Data Cleaning and Integration**

**Article:** https://arxiv.org/abs/2305.13494

This repo provides the following:
* An evaluation of existing deep clustering algorithms on several data integration problems, specifically schema inference, entity resolution, and domain discovery.
* Comparing them to baselines that use different non-deep clustering techniques.



# Note: 

#### In this repo we used the modified implementations (suitable for data integration tasks) of following deep clustering algorithms:

* [SDCN](https://github.com/bdy9527/SDCN)
* [EDESC](https://github.com/JinyuCai95/EDESC-pytorch)
* [SHGP](https://github.com/kepsail/SHGP)

#### And, the following embedding techniques:

* [EmbDi](https://gitlab.eurecom.fr/cappuzzo/embdi)
* [TabTransformer](https://github.com/jrzaurin/pytorch-widedeep)
* [Tabnet](https://github.com/jrzaurin/pytorch-widedeep)
* [SBERT](https://www.sbert.net/docs/hugging_face.html)
* [FastText](https://fasttext.cc/docs/en/crawl-vectors.html)

#### For original implementations please see the links given above.

**Special thanks** to the authors (of all deep clustering and embedding approaches mentioned above) for providing their implementations publically.


## Requirements
* Python 3.7 
* pytorch-cuda=11.7 or above
* To reproduce results efficiently you will need GPU a100 or v100.



## Run

There are two major parts to this project: For each data integration problem, i.e., schema inference, entity resolution, and domain discovery:


 1. Develop a dense embedding matrix (X.txt) from raw data (using SBERT, FastText, TabTransformer, and Tabnet to embed Tables for schema inference, EmbDi and SBERT to embed rows for entity resolution, and EmbDi and SBERT to embed columns for domain discovery).
 2. A dense embedding matrix (X.txt) will then be used in deep clustering algorithms (SDCN, EDESC and SHGP) as input to perform clustering.

## Steps

Here we show demo steps to re-produce results for schema inference using schema-level web tables data (SBERT + SDCN), schema+instance level Table union search data (SBERT + SDCN) and for entity resolution for schema+instance level data (EmbDi + SDCN). 


1. Compile schema inference/schema + instances/Preprocessing.ipynb to get schema level information from tables.
2. The generated TextPre1.csv can be used to produce a dense embedding matrix (X.txt) using SBERT by compiling schema inference/schema only/SBERT+FastText.py
3. We have X.txt feature vector which will be used to get clustering results in SDCN by compiling.

   3.1. DC/SDCN/calcu_graph.py to generate KNN graph
  
   3.2. DC/SDCN/data/pretrain.py for pretraining or AE version (please see hyperparameter tables below) 
  
   3.3. DC/SDCN/sdcn.py for clustering results (we considered Q distribution as our final results)
  
   3.4. Please updated nb_dimension = 768  accordingly # for SBERT 786  
4. Compile entity resolution/ER.py to get row embedding matrix (X.txt) using EmbDi and compile entity resolution/ER_SBERT/ER_SBERT.py to get row embeddings using SBERT.
5. Repeat step (3) with input as row embedding matrix (X.txt) to get clustering results.
6. Compile TUS.py and generate_labels.py to obtain labels using the Louvain method. We first need to download TUS data (benchmark.sqlite and groundtruth.sqlite) from [here](https://github.com/RJMillerLab/table-union-search-benchmark).
7. Compile Transformer.py to get table embedding matrix (X.txt) using Tabnet or TabTransformer.
8. Repeat step (3) with input as table embedding matrix (X.txt) to get clustering results.
9. Please repeat (3) for each embedding obtained (see folders --> schema inference/, entity resolution/ and domain discovery/) or see full_data/ to get the combination of all embeddings for all problems except DD due to the size limit. We are only able to provide ready-to-use embeddings for FasTtext.
10. Forexample compile domain discovery/DD.py to get column embedding matrix (X.txt) using EmbDi and compile domain discovery/DD_SBERT(H+B)/DD_SBERT.py to get column embeddings of schema+instance level evidence using SBERT.
11. When applying SDCN with tabular transformers for schema+instance level data, compile schema inference/schema + instances/SI_transformers/SI_transformers.py
12. To get results with standard clustering algorithms compile SC/SC.py



## Hyperparameters
In order to reproduce results in the paper, parameters (given in Table 1 and 2) can be adopted:

#### Table 1: Hyperparameters used for web tables, Music Brainz and Camera datasets

| DC Method | Task | Embedding     | P-train epochs | Training Epochs | Z   | Layer Size | #layers | Initialization |
|-----------|------|---------------|----------------|-----------------|-----|------------|---------|----------------|
| SDCN      | SI   | SBERT         | 30             | 97              | 100 | 1000       | 2       | Birch         |
| SDCN      | SI   | FastText      | 30             | 100             | 100 | 1000       | 2       | kmean         |
| SDCN      | SI   | TabNet        | 30             | 100             | 100 | 1000       | 2       | kmean         |
| SDCN      | SI   | TabTransformer| 30             | 100             | 100 | 1000       | 2       | kmean         |
| EDESC     | SI   | SBERT         | 30             | 65              | 104 | 1000       | 2       | Birch         |
| EDESC     | SI   | FastText      | 30             | 43              | 104 | 1000       | 2       | kmean         |
| EDESC     | SI   | TabNet        | 30             | 96              | 104 | 1000       | 2       | kmean         |
| EDESC     | SI   | TabTransformer| 30             | 73              | 104 | 1000       | 2       | kmean         |
| SHGP      | SI   | SBERT         | -              | 50              | -   | 1000       | -       | kmean         |
| SHGP      | SI   | FastText      | -              | 50              | -   | 1000       | -       | kmean         |
| SHGP      | SI   | TabNet        | -              | 50              | -   | 1000       | -       | kmean         |
| SHGP      | SI   | TabTransformer| -              | 50              | -   | 1000       | -       | kmean         |
| SDCN      | ER   | EmbDi         | 100            | -               | 100 | 1000       | 2       | Birch         |
| EDESC     | ER   | EmbDi         | 100            | 4               | 684 | 1000       | 2       | Birch         |
| SDCN      | ER   | SBERT         | 100            | -               | 100 | 1000       | 2       | Birch         |
| EDESC     | ER   | SBERT         | 100            | 76              | 684 | 1000       | 2       | Birch         |
| SHGP      | ER   | SBERT         | -              | 50              | -   | 1000       | -       | kmean         |
| SHGP      | ER   | EmbDi         | -              | 50              | -   | 1000       | -       | kmean         |
| SDCN      | DD   | SBERT         | 30             | 88              | 100 | 1000       | 2       | Birch         |
| SDCN      | DD   | FastText      | 30             | 2               | 100 | 1000       | 2       | Birch         |
| SDCN      | DD   | EmbDi         | 30             | 1               | 100 | 1000       | 2       | kmean         |
| SDCN      | DD   | SBERT (H+B)   | 30             | 98              | 100 | 1000       | 2       | Birch         |
| EDESC     | DD   | SBERT         | 30             | 74              | 112 | 1000       | 2       | Birch         |
| EDESC     | DD   | FastText      | 30             | 100             | 112 | 1000       | 2       | kmean         |
| EDESC     | DD   | EmbDi         | 30             | 1               | 112 | 1000       | 2       | kmean         |
| EDESC     | DD   | SBERT (H+B)   | 30             | 61              | 100 | 1000       | 2       | Birch         |
| SHGP      | DD   | SBERT         | -              | 50              | -   | 1000       | -       | kmean         |
| SHGP      | DD   | FastText      | -              | 50              | -   | 1000       | -       | kmean         |
| SHGP      | DD   | EmbDi         | -              | 50              | -   | 1000       | -       | kmean         |
| SHGP      | DD   | SBERT (H+B)   | -              | 50              | -   | 1000       | -       | kmean         |


#### Table 2: Hyperparameters used for TUS, Geographic Settlements and Monitor
| DC Method | Task | Embedding       | P-train epochs | Training Epochs | Z    | Layer Size | #layers | Initialization |
|-----------|------|-----------------|----------------|-----------------|------|------------|---------|----------------|
| SDCN      | SI   | SBERT           | 30             | 5               | 100  | 1000       | 2       | kmean         |
| SDCN      | SI   | FastText        | 30             | 10              | 100  | 1000       | 2       | kmean         |
| SDCN      | SI   | TabNet          | 30             | 100             | 100  | 1000       | 2       | kmean         |
| SDCN      | SI   | TabTransformer  | 30             | 68              | 100  | 1000       | 2       | kmean         |
| EDESC     | SI   | SBERT           | 30             | 8               | 111  | 1000       | 2       | kmean         |
| EDESC     | SI   | FastText        | 30             | 83              | 111  | 1000       | 2       | kmean         |
| EDESC     | SI   | TabNet          | 30             | 52              | 111  | 1000       | 2       | kmean         |
| EDESC     | SI   | TabTransformer  | 30             | 100             | 111  | 1000       | 2       | kmean         |
| SHGP      | SI   | SBERT           | -              | 50              | -    | 1000       | -       | kmean         |
| SHGP      | SI   | FastText        | -              | 50              | -    | 1000       | -       | kmean         |
| SHGP      | SI   | TabNet          | -              | 50              | -    | 1000       | -       | kmean         |
| SHGP      | SI   | TabTransformer  | -              | 50              | -    | 1000       | -       | kmean         |
| SDCN      | ER   | EmbDi           | 200            | -               | 100  | 1000       | 2       | -             |
| EDESC     | ER   | EmbDi           | 100            | 4               | 100  | 1000       | 2       | kmean         |
| SHGP      | ER   | EmbDi           | -              | 50              | -    | 1000       | -       | kmean         |
| SDCN      | ER   | SBERT           | 100            | -               | 100  | 1000       | 2       | -             |
| EDESC     | ER   | SBERT           | 30             | 4               | 786  | 1000       | 2       | kmean         |
| SHGP      | ER   | SBERT           | -              | 50              | -    | 1000       | -       | kmean         |
| SDCN      | DD   | SBERT           | 30             | -               | 100  | 1000       | 2       | kmean         |
| SDCN      | DD   | FastText        | 30             | -               | 100  | 1000       | 2       | kmean         |
| SDCN      | DD   | EmbDi           | 30             | 6               | 100  | 1000       | 2       | kmean         |
| SDCN      | DD   | SBERT (H+B)     | 50             | -               | 100  | 1000       | 2       | Birch         |
| EDESC     | DD   | SBERT           | 30             | 100             | 243  | 1000       | 2       | kmean         |
| EDESC     | DD   | FastText        | 30             | 4               | 81   | 1000       | 2       | kmean         |
| EDESC     | DD   | EmbDi           | 30             | 5               | 243  | 1000       | 2       | kmean         |
| EDESC     | DD   | SBERT (H+B)     | 30             | 1               | 243  | 1000       | 2       | Birch         |
| SHGP      | DD   | SBERT           | -              | 50              | -    | 1000       | -       | kmean         |
| SHGP      | DD   | FastText        | -              | 50              | -    | 1000       | -       | kmean         |
| SHGP      | DD   | EmbDi           | -              | 50              | -    | 1000       | -       | kmean         |
| SHGP      | DD   | SBERT (H+B)     | -              | 50              | -    | 1000       | -       | kmean         |



**-** SDCN or EDESC did not manage to improve the representation, and we retained the AE training with Birch or Kmeans clustering

**Note:** 

**1.** Due to storage limitations, please unzip (Tables.zip) before compiling schema inference code.

**2.** Due to different levels of precision for floating-point arithmetic and the architectural aspect of different GPUs and CPUs, the resulting values can be slightly different. However, the overall results will be the same. We used **Nvidia A100 GPU with 80GB GPU RAM **.

## License
This work provides a modified version of the following embedding techniques and deep clustering algorithms.  The original license terms apply, and a copy of the original license files is included in the respective folder of embedding techniques and deep clustering algorithms.

* [SDCN](https://github.com/bdy9527/SDCN)
* [EDESC](https://github.com/JinyuCai95/EDESC-pytorch)
* [SHGP](https://github.com/kepsail/SHGP)
* [EmbDi](https://gitlab.eurecom.fr/cappuzzo/embdi)
* [TabTransformer](https://github.com/jrzaurin/pytorch-widedeep)
* [Tabnet](https://github.com/jrzaurin/pytorch-widedeep)
