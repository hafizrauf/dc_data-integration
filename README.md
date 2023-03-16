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


**Note:** Due to storage limitations, please unzip (Tables.zip) before compiling schema inference code.


