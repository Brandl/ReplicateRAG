# RAG-DLNP

Code for a university group project.
192.039 Deep Learning for Natural Language Processing
2025S

### Group

- Juergen Brandl 
- Jihye Kang 
- Alenka Triplat 

## I. Replicate

Install the requirements first

    pip install -U transformers datasets faiss-cpu huggingface_hub fsspec
    pip install -U torch torchvision

We used the model to run answering of questions on 6 different dataset. For convience this repo comes with `qa_dataset_all_slices.tar.gz` which includes wikipedia articles from NaturalQA and TriviaQA dataset for you to load. Unpack all datasets first:

    tar -zxvf psgs_w100.tsv.gz

In replication, each approach will run prediction of Answers for the following configuration: "dataset_name", "size", "use dummy". If use_dummy set to `True` it loads only 10.000 retrieval docs for Retriever:
- "nq", "350", False
- "triviaqa", "1k", False
- "nq", "3600", False
- "nq", "11000", False
- "triviaqa", "10k", False
- "triviaqa", "30k", False

There are 5 different implementations for RAG Retriever. Each one will generate answers into folder `/replicate_answers`

### 1. Full Retriever
This loads the pre-trained RAG Retriever from [HuggingFace RAG](https://github.com/huggingface/transformers/blob/main/src/transformers/models/rag/retrieval_rag.py): "facebook/rag-token-nq", with "compressed" index. The complete retriever index requires over 76GB of RAM. To load the file, you should use at PyTorch version 2.6 or higher.
In detail for `psgs_w100.nq.compressed` dataset which is use:

    Size of downloaded dataset files: 85.23 GB
    Size of the generated dataset: 78.42 GB
    Total amount of disk used: 163.71 GB

You might have to install git LFS when loading large files

    git LFS (sudo apt-get install -y git-lfs)

To replicate the RAG approach by loading entire retriever run:

    python replicate_answers_Full_Retriever.py

### 2. Dummy Retriever

To replicate the RAG approach by using only dummy size dataset for Retriever (proof of concept), run: 

    python replicate_answers_Dummy_Retriever.py

### 3. Load Wikipedia DPR dataset and load Retriever locally

Since Full retriever may get "stuck" loading the data due to RAM and local Disk size problems, we developed an approach to load the entire Wikipedia DPR dataset locally and then load the retriever using the locally available data.

You will need to first load the dataset from Wiki DPR available at [HuggingFace Wiki DPR](https://huggingface.co/datasets/facebook/wiki_dpr) into folder: `retrieverdata/wiki_dpr` The easiest is to use wget to get the respective data and index file. We used this file for data `psgs_w100.tsv`and this one for index `psgs_w100.nq.IVF4096_HNSW128_PQ128-IP-train.faiss`

The approach requires to embed the retrieved passages from data file (limited at max length of 512) and add faiss index again before loading dataset. We use the a pretrained DPR Context Encoder and Tokenizer for that. The code is optimized to use GPU and max RAM of JupyterLab engine (e.g. using batch size of 512 for embedding, reduce if you don't have sufficient RAM and GPU available).

    python replicate_answers_local_dpr.py

### 4. Load smaller Wikipedia doc dataset and load Retriever locally

Essentially same approach as for 3. but with smaller Wikipedia dataset. You will need to first load the dataset from Wiki Snippets available at [HuggingFace Wiki Snippets](https://huggingface.co/datasets/community-datasets/wiki_snippets) into folder: `retrieverdata/rtr_wiki_small` The easiest is to use wget to get the respective data and index file.

    python replicate_answers_wiki_small.py



## II. Modern RAG

Install the requirements first

    pip install langchain langchain-community langchain-openai qdrant-client sentence-transformers datasets tqdm pandas faiss-cpu

Start qdrant as a docker container, this gives better performance for the full wikipedia dataset:

    sudo docker run -d --name qdrant -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

You also need some OpenAI API compatible Endpoint

    export OPENAI_BASE_URL=https://openrouter.ai/api/v1
    export OPENAI_API_KEY=sk-...

### Load a dataset

For convience this repo comes with qa_dataset_all_slices.tar.gz which includes wikipedia articles from NaturalQA dataset for you to load:

    python rag_query.py --load-wikipedia-slice naturalqa_slice_3600_wikipedia.json -c wikipedia_nq3600

Or if you want to load the full dataset, you can download the psgs_w100.tsv 

    wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz 
    tar -zxvf psgs_w100.tsv.gz 
    python rag_query.py --load-wikipedia-tsv psgs_w100.tsv -c wikipedia_full

At least in theory --load-wikipedia-full should load the full from wikipedia, but it broke so many times for obscure reasons, that we dont reccomend going that route.

### Query

You can test retrieval and question answering using this commnd:

    python rag_query.py -c wikipedia_nq3600 --debug "What is the most famous football team in spain"

To prepare answer files for eval the command is as follows:

    python rag_query.py -c wikipedia_nq3600  --question-file naturalqa_slice_3600_qa.json --answer-file naturalqa_slice_3600_answers.json

This will read in questions and write the answers down in the specified file.

## III. Eval

You can evaluate the generated answer with "golden" true answers by running the following script or equivalent files: first argument is for file which stores prediction, second one for gold answers as downloaded in earlier steps. Both files should be .json format following this structure:  `{int(d["id"]): (d["question"], d["answer"])}`

    python eval_rag_.py --preds_ref answers/nq_350_answers_jn.json qa_datasets_all_slices/naturalqa_slice_350_qa.json


## Sources
Original Paper:

    @article{DBLP:journals/corr/abs-2005-11401,
      author       = {Patrick Lewis and
                      Ethan Perez and
                      Aleksandra Piktus and
                      Fabio Petroni and
                      Vladimir Karpukhin and
                      Naman Goyal and
                      Heinrich K{\"{u}}ttler and
                      Mike Lewis and
                      Wen{-}tau Yih and
                      Tim Rockt{\"{a}}schel and
                      Sebastian Riedel and
                      Douwe Kiela},
      title        = {Retrieval-Augmented Generation for Knowledge-Intensive {NLP} Tasks},
      journal      = {CoRR},
      volume       = {abs/2005.11401},
      year         = {2020},
      url          = {https://arxiv.org/abs/2005.11401},
      eprinttype    = {arXiv},
      eprint       = {2005.11401},
      timestamp    = {Mon, 14 Apr 2025 22:19:01 +0200},
      biburl       = {https://dblp.org/rec/journals/corr/abs-2005-11401.bib},
      bibsource    = {dblp computer science bibliography, https://dblp.org}
    }

Datasets:

https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
https://huggingface.co/datasets/community-datasets/wiki_snippets/tree/main/wiki40b_en_100_0
https://huggingface.co/datasets/google-research-datasets/natural_questions
https://nlp.cs.washington.edu/triviaqa/

Code Samples & Image Credit

https://github.com/langchain-ai/rag-from-scratch

