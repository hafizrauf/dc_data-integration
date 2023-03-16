# **Deep Clustering for Data Integration Tasks**


This repo provides the following:
* An evaluation of existing deep clustering algorithms on several data integration problems, specifically schema inference, entity resolution, and domain discovery.
* Comparing them to baselines that use different non-deep clustering techniques.



# Note: 

#### In this repo we used the modified implementations (suitable for data integration tasks) of following deep clustering algorithms:

* [SDCN](https://github.com/bdy9527/SDCN)
* [EDESC](https://github.com/JinyuCai95/EDESC-pytorch)

#### Furthermore, following embedding techniques:

* [EmbDi](https://gitlab.eurecom.fr/cappuzzo/embdi)
* [TabTransformer](https://github.com/jrzaurin/pytorch-widedeep)
* [Tabnet](https://github.com/jrzaurin/pytorch-widedeep)
* [SBER](https://www.sbert.net/docs/hugging_face.html)
* [FastText](https://fasttext.cc/docs/en/crawl-vectors.html)

#### For original implementations please see the links given above.

**Special thanks** to the authors (of all deep clustering and embedding approaches mentioned above) for providing their implementations publically.


## Requirements
* Python 3.7 
* pytorch-cuda=11.7 or above
* To reproduce results efficiently you will need GPU a100 or v100.



## Run

There are two major parts to this project: For each data integration problem, i.e., schema inference, entity resolution, and domain discovery:


 1. Develop a dense embedding matrix (X.text) from raw data (using SBERT, FastText, TabTransformer, and Tabnet to embed Tables for schema inference, EmbDi to embed rows for entity resolution, and EmbDi to embed columns for domain discovery).
 2. A dense embedding matrix (X.text) will then be used in deep clustering algorithms (SDCN and EDESC) as input to perform clustering.

## Steps

Here we show demo steps to re-produce results for schema inference using schema-level data.

````
1. Compile schema inference/schema + instances/Preprocessing.ipynb to get schema level information from tables.
2. The generated TextPre1.csv can then be used to produce a dense embedding matrix (X.text) using SBERT by compiling schema inference/schema only/SBERT+FastText.py
3. We have X.text feature vector that will be used to get clustering results in SDCN by compiling DC/SDCN/data/pretrain.py for pretraining or DNN version and DC/SDCN/sdcn.py

````



## Hyperparameters
In order to reproduce results in the paper, following parameters can be adopted:

| DC Method | Task | Embedding | P-train epochs | Training Epochs | Z	Layer Size | P-train algo | train algo | 
| ---|--- |--- |--- |--- |--- |--- |---|              
|SDCN	|SI	|SBERT|	**0|	35|	100	|1000|	-|	Kmean|
|SDCN	|SI	|FastText|	28|	2|	100|	1000|	Birch|	Kmean|
|SDCN	|SI	|TabNet	|50|	1|	100|	5000|	Kmean	|Kmean|
|SDCN	|SI	|TabTransformer|	50|	7|	100|	5000|	Kmean|	Kmean|
|EDESC|	SI|	SBERT|	49|	2|	104|	500|	Birch|	Birch|
|EDESC|	SI|	FastText|	32|	2	|156|	1000|	Kmean|	kmean|
|EDESC|	SI|	TabNet|	50|	106	|130	|1000|	Birch|	Birch|
|EDESC|	SI|	TabTransformer|	50|	108	|130|	5000|	Birch|	Birch|
|SDCN	|ER	|EmbDi	**0|	34|	250|	5000	|-|	Birch|
|EDESC|	ER	|EmbDi|	50	|3	|684|	6000	|Birch|	Birch|
|SDCN	|DD	|SBERT|	**0|	8|	100	|1000	|-|	Birch|
|SDCN	|DD	|FastText|	5|	16|	100|	500|	Kmean	|Birch|
|SDCN	|DD	|EmbDi	|50	|1	|56	|2000*	|Birch	|Kmean|
|EDESC|	DD|	SBERT|	4|	5|	112	|1000|	Birch	|Birch|
|EDESC|	DD|	FastText|	49|	2|	56	|1000	|Birch|	Birch|
|EDESC|	DD|	EmbDi|	3|	7|	280	|256	|Kmean|	kmean|

*SDCN used total 6 layers with layers size of 256 and 2000.

** Specific DNN experiment with AE + (Birch or Kmean)





**Note:** Due to storage limitations, please unzip (Tables.zip) before compiling schema inference code.


