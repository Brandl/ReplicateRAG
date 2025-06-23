# RAG-DLNP

## Replicate

## Modern RAG

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

## Eval