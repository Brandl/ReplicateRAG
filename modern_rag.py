#!/usr/bin/env python3
import argparse
import os
import json
import uuid
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from datasets import load_dataset
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# To do generation this needs a openai API compatible server, could be ollama, we used openrouter for convience
#os.environ['OPENAI_BASE_URL'] = "https://openrouter.ai/api/v1"
#os.environ['OPENAI_API_KEY'] = "sk-or-v1-secret"

def load_wikipedia_slice_to_qdrant(filepath, collection_name="naturalqa", max_docs=None):
    """Load Wikipedia documents from JSON file into Qdrant"""
    with open(filepath, 'r') as f:
        docs = json.load(f)
    
    # Initialize Qdrant client
    client = QdrantClient(host="localhost", port=6333, timeout=300)
    
    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create collection if it doesn't exist
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Created Qdrant collection: {collection_name}")
    except Exception as e:
        print(f"Collection exists or error: {e}")
    
    # Prepare documents for Qdrant
    documents = []
    points = []
    
    # Limit docs if max_docs is specified
    if max_docs:
        docs = docs[:max_docs]
    
    for doc in tqdm(docs, desc="Processing Wikipedia documents"):
        # Use title + all tokens, chunked if needed
        tokens = doc.get('tokens', [])
        full_text = ' '.join(tokens)
        
        # If document is short enough, use it as is
        if len(tokens) <= 8000:
            content = f"{doc['title']}\n\n{full_text}"
            documents.append(content)
        else:
            # Chunk larger documents
            chunk_size = 4000
            overlap = 500
            for i in range(0, len(tokens), chunk_size - overlap):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = ' '.join(chunk_tokens)
                content = f"{doc['title']} (chunk {i//chunk_size + 1})\n\n{chunk_text}"
                documents.append(content)
    
    # Generate embeddings in batches
    print(f"Generating embeddings for {len(documents)} documents...")
    embeddings = embedding_model.encode(documents, batch_size=32, show_progress_bar=True)
    
    # Create Qdrant points
    doc_idx = 0
    for doc in docs:
        tokens = doc.get('tokens', [])
        full_text = ' '.join(tokens)
        
        if len(tokens) <= 8000:
            content = f"{doc['title']}\n\n{full_text}"
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[doc_idx].tolist(),
                payload={
                    "doc_id": doc['id'],
                    "title": doc['title'],
                    "text": full_text,
                    "content": content,
                    "doc_type": "wikipedia"
                }
            )
            points.append(point)
            doc_idx += 1
        else:
            # Handle chunked documents
            chunk_size = 4000
            overlap = 500
            for i in range(0, len(tokens), chunk_size - overlap):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = ' '.join(chunk_tokens)
                content = f"{doc['title']} (chunk {i//chunk_size + 1})\n\n{chunk_text}"
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[doc_idx].tolist(),
                    payload={
                        "doc_id": doc['id'],
                        "title": doc['title'],
                        "text": chunk_text,
                        "content": content,
                        "doc_type": "wikipedia",
                        "chunk": i//chunk_size + 1
                    }
                )
                points.append(point)
                doc_idx += 1
    
    # Insert in batches to avoid timeouts
    batch_size = 500
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch_points)
    
    print(f"Loaded {len(points)} Wikipedia documents into Qdrant collection: {collection_name}")

def load_wikipedia_tsv_to_qdrant(tsv_path="/home/ubuntu/WIKI/downloads/data/wikipedia_split/psgs_w100.tsv", 
                                collection_name="wikipedia_full", max_docs=None):
    """Load Wikipedia dataset from TSV file into Qdrant using proper batching"""
    import pandas as pd
    
    print(f"Loading Wikipedia dataset from TSV file to Qdrant: {tsv_path}")
    
    # Initialize Qdrant client with longer timeout
    client = QdrantClient(host="localhost", port=6333, timeout=300)
    
    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create collection if it doesn't exist
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Created Qdrant collection: {collection_name}")
    except Exception as e:
        print(f"Collection exists or error: {e}")
    
    # Read file in chunks for memory efficiency
    chunk_size = 5000  # Smaller chunks for embedding generation and better batching
    column_names = ['id', 'text', 'title']
    processed_count = 0
    
    for chunk in tqdm(pd.read_csv(tsv_path, sep='\t', names=column_names, chunksize=chunk_size, 
                                  dtype=str, na_filter=False), desc="Processing Wikipedia chunks"):
        
        if max_docs and processed_count >= max_docs:
            break
            
        # Limit chunk if we're near max_docs
        if max_docs:
            remaining = max_docs - processed_count
            if remaining < len(chunk):
                chunk = chunk.head(remaining)
        
        # Prepare all data for this chunk
        documents = []
        points = []
        
        for _, row in chunk.iterrows():
            content = f"{row['title']}\n\n{row['text']}"
            documents.append(content)
        
        # Generate embeddings in batch
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = embedding_model.encode(documents, batch_size=32, show_progress_bar=True)
        
        # Create Qdrant points
        for i, (_, row) in enumerate(chunk.iterrows()):
            content = f"{row['title']}\n\n{row['text']}"
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={
                    "doc_id": row['id'],
                    "title": row['title'],
                    "text": row['text'],
                    "content": content,
                    "doc_type": "wikipedia"
                }
            )
            points.append(point)
        
        # Insert in smaller batches to avoid timeouts
        batch_size_insert = 500  # Smaller batch for upsert
        for i in range(0, len(points), batch_size_insert):
            batch_points = points[i:i + batch_size_insert]
            client.upsert(collection_name=collection_name, points=batch_points)
        
        processed_count += len(chunk)
        print(f"Processed {processed_count} documents so far...")
        
        if max_docs and processed_count >= max_docs:
            break
    
    print(f"Loaded {processed_count} Wikipedia documents into Qdrant collection: {collection_name}")
    

def load_wikipedia_full_to_qdrant(collection_name="wikipedia_full", max_docs=None):
    """Load full Wikipedia dataset from HuggingFace wiki_dpr into Qdrant"""
    print("Loading wiki_dpr dataset from HuggingFace...")
    # Use no_embeddings configuration to avoid FAISS index issues
    dataset = load_dataset("wiki_dpr", "psgs_w100.nq.no_embeddings", split="train")
    
    # Initialize Qdrant client
    client = QdrantClient(host="localhost", port=6333, timeout=300)
    
    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create collection if it doesn't exist
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Created Qdrant collection: {collection_name}")
    except Exception as e:
        print(f"Collection exists or error: {e}")
    
    total_docs = len(dataset) if max_docs is None else min(max_docs, len(dataset))
    print(f"Processing {total_docs} Wikipedia documents...")
    
    batch_size = 1000
    processed_count = 0
    
    for i in tqdm(range(0, total_docs, batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, total_docs)
        documents = []
        points = []
        
        # Collect documents for this batch
        for j in range(i, batch_end):
            doc = dataset[j]
            content = f"{doc['title']}\n\n{doc['text']}"
            documents.append(content)
        
        # Generate embeddings in batch
        print(f"Generating embeddings for batch...")
        embeddings = embedding_model.encode(documents, batch_size=32, show_progress_bar=False)
        
        # Create Qdrant points
        for idx, j in enumerate(range(i, batch_end)):
            doc = dataset[j]
            content = f"{doc['title']}\n\n{doc['text']}"
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[idx].tolist(),
                payload={
                    "doc_id": str(doc['id']),
                    "title": doc['title'],
                    "text": doc['text'],
                    "content": content,
                    "doc_type": "wikipedia"
                }
            )
            points.append(point)
        
        # Insert in smaller batches to avoid timeouts
        insert_batch_size = 500
        for k in range(0, len(points), insert_batch_size):
            batch_points = points[k:k + insert_batch_size]
            client.upsert(collection_name=collection_name, points=batch_points)
        
        processed_count += len(points)
    
    print(f"Loaded {processed_count} Wikipedia documents into Qdrant collection: {collection_name}")

def query_rag(question, collection="wikipedia_full", model="google/gemini-2.0-flash-lite-001", 
              doc_type=None, n_results=4, debug=False):
    # Qdrant query
    client = QdrantClient(host="localhost", port=6333, timeout=300)
    
    # Load embedding model for query
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embedding for question
    query_vector = embedding_model.encode([question])[0].tolist()
    
    # Search in Qdrant
    search_filter = {"must": [{"key": "doc_type", "match": {"value": doc_type}}]} if doc_type else None
    
    results = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=n_results,
        query_filter=search_filter
    )
    
    if debug:
        print(f"Retrieved {len(results)} documents:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result.payload['title']} (Score: {result.score:.3f})")
            print(f"   Preview: {result.payload['content'][:200]}...")
    
    # Format context from results
    context = "\n\n".join([result.payload['content'] for result in results])
    
    # Local RAG prompt with examples and brief answer format
    prompt = ChatPromptTemplate.from_template(
        """Answer the question using the context provided. Look carefully through all the context for relevant information. Be extremely brief and direct like these examples:

Q: when was the first robot used in surgery
A: 1983

Q: where is zimbabwe located in the world map
A: in southern Africa, between the Zambezi and Limpopo Rivers, bordered by South Africa, Botswana, Zambia and Mozambique

Q: who sings the song i don't care i love it
A: Icona Pop and Charli XCX

Only respond with --NOT FOUND-- if you truly cannot find any relevant answer in the context.

Context: {context}
Question: {question}
Answer:"""
    )
    
    llm = ChatOpenAI(model_name=model, temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({"context": context, "question": question})

def process_question_file(question_file, num_qa, answer_file, collection, model, doc_type, n_results, debug):
    """Process questions from a file and save answers"""
    with open(question_file, 'r') as f:
        questions = json.load(f)
    
    # Limit to num_qa questions if specified
    if num_qa:
        questions = questions[:num_qa]
    
    results = []
    for i, q in enumerate(questions):
        print(f"Processing question {i+1}/{len(questions)}: {q['question']}")
        answer = query_rag(q['question'], collection, model, doc_type, n_results, debug)
        result = {
            "id": q['id'],
            "question": q['question'],
            "answer": answer
        }
        results.append(result)
        print(f"Answer: {answer}\n")
    
    # Save results to answer file
    with open(answer_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} questions. Answers saved to {answer_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs='?', help="Question to ask")
    parser.add_argument("-c", "--collection", default="naturalqa_chunked")
    parser.add_argument("-m", "--model", default="google/gemini-2.0-flash-lite-001")
    parser.add_argument("-d", "--doc-type")
    parser.add_argument("--load-wikipedia-slice", help="Load Wikipedia JSON file (slice) into Qdrant")
    parser.add_argument("--load-wikipedia-full", action="store_true", help="Load full Wikipedia dataset from HuggingFace wiki_dpr into Qdrant")
    parser.add_argument("--load-wikipedia-tsv", help="Load Wikipedia dataset from TSV file into Qdrant")
    parser.add_argument("--max-docs", type=int, help="Maximum number of documents to load from wiki_dpr")
    parser.add_argument("--debug", action="store_true", help="Show retrieved documents")
    parser.add_argument("-n", "--n-results", type=int, default=4, help="Number of results to retrieve")
    parser.add_argument("--question-file", help="JSON file containing questions to process")
    parser.add_argument("--num-qa", type=int, help="Number of questions to process from the file")
    parser.add_argument("--answer-file", help="Output file to store answers")
    
    args = parser.parse_args()
    
    if args.load_wikipedia_slice:
        load_wikipedia_slice_to_qdrant(args.load_wikipedia_slice, args.collection, args.max_docs)
    elif args.load_wikipedia_full:
        load_wikipedia_full_to_qdrant(args.collection, args.max_docs)
    elif args.load_wikipedia_tsv:
        load_wikipedia_tsv_to_qdrant(args.load_wikipedia_tsv, args.collection, args.max_docs)
    elif args.question_file:
        if not args.answer_file:
            parser.error("--answer-file is required when using --question-file")
        process_question_file(args.question_file, args.num_qa, args.answer_file, 
                              args.collection, args.model, args.doc_type, 
                              args.n_results, args.debug)
    elif args.question:
        print(query_rag(args.question, args.collection, args.model, args.doc_type, 
                        n_results=args.n_results, debug=args.debug))
    else:
        parser.error("Either provide a question, use --load-wikipedia-slice, --load-wikipedia-full, or use --question-file")