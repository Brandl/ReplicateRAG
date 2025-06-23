from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
import time

from datasets import load_from_disk
from datasets import Dataset as hfDataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import torch
from torch.utils.data import Dataset, DataLoader

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())

class LocalDprRetrieverDataset:
    def __init__(self, 
                 purpose = "sample", 
                 dataset_name="wiki40b_en_100_0", 
                 retriever_ds: hfDataset = None):
        self.purpose = purpose  # Change to "sample" for the sample dataset, change to "full" for the full dataset
        self.dataset_name = dataset_name
        self.dpr_data_dir = Path("retrieverdata/wiki_dpr_download/downloads/data/wikipedia_split/psgs_w100.tsv")
        self.retriever_ds = retriever_ds
    
    def prepare_and_store_retriever_dataset(self):
        
        # Step 1. Load TSV as HF Dataset
        if self.purpose == "sample":        
            df = pd.read_csv(self.dpr_data_dir, sep="\t", header=None, names=["id", "text", "title"], nrows=10)
        else:
            df = pd.read_csv(self.dpr_data_dir, sep="\t", header=None, names=["id", "text", "title"])
        
        # Drop duplicate header row if present as first data row
        # Remove the duplicated row which is identical to column names:
        df = df[df["id"] != "id"]  # Remove rows where 'id' column literally equals 'id' string

        # Reset index using the 'id' column
        df.set_index("id", inplace=True)

        ds = hfDataset.from_pandas(df.reset_index(drop=True))
        print("Dataset loaded successfully.")

        # No reformatting required
        
        # Step 3: Embed the passages with DPR Context Encoder and tokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)
        ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

        def embed_passages(batch):
            inputs = ctx_tokenizer(
                batch["text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
                ).to(device)
            
            with torch.no_grad():
                embeddings = ctx_encoder(**inputs).pooler_output
            return {"embeddings": embeddings.cpu().numpy()}
        
        # Batch-wise encoding
        ds = ds.map(embed_passages, batched=True, batch_size=512)
        print("Embeddings completed")
        
        # Step 4: Build FAISS index
        ds.add_faiss_index(column="embeddings")
        self.retriever_ds = ds
        
        return ds

class QADataset(Dataset):
    """Generic Q&A Dataset class for both NQ and TriviaQA"""
    def __init__(self, qa_input):
        self.qa_input = qa_input
    
    def __len__(self):
        return len(self.qa_input)
    
    def __getitem__(self, idx):
        item = self.qa_input[idx]
        return item["id"], item["question"], item["answer"]

class RAGModelManagerLocalDprWiki:
    """Class to manage RAG model components and inference using local Wiki Dpr dataset
        define if for testing purpose = "sample" or "full" dataset 
        """
    
    def __init__(self, purpose = "sample"):
        self.purpose = purpose  # Change to "sample" for the sample dataset, change to "full" for the full dataset
        self.tokenizer = None
        self.retriever = None
        self.model = None
        self.device = None
        self.is_initialized = False
        self.retriever_ds = None

    def load_retriever_dataset(self):
        retriever_manager = LocalDprRetrieverDataset(purpose = self.purpose)
        ds = retriever_manager.prepare_and_store_retriever_dataset()
        self.retriever_ds = ds
    
    def initialize_model(self, device=None):
        if self.is_initialized:
            print("Model already initialized!")
            return
            
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")

        print("Loading RAG model components...")

        # Step 5: Load RAG: tokenizer, retrievers and model
        # Load tokenizer and model
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")

        # Use RAG retriever from local Wiki Dpr data
        self.load_retriever_dataset()

        self.retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq",
            index_name="custom",
            indexed_dataset=self.retriever_ds,
            use_dummy_dataset=False,
            trust_remote_code=True
        ) 
    
        self.model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=self.retriever)
        self.model.to(self.device)
        self.model.eval()
        self.is_initialized = True
        print(f"Model components loaded successfully on {self.device}!")
    
    def generate_answer(self, queries):
        """Generate answers for a batch of questions"""
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        print("Sample questions input:", queries)
        print("Type of first question:", queries[0])
        
        self.model.to(self.device)
        
        inputs = self.tokenizer(
            queries, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"]
            )
        
        answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return answers

def load_qa(dataset_type="nq", size="350", batch_size=8):
    """
    Load dataset based on type and size
    
    Args:
        dataset_type: "nq" for Natural Questions or "triviaqa" for TriviaQA
        size: dataset size (350, 3600, 11000, 1000, 10000, 30000, etc.)
    """
    dir_name = "qa_datasets_all_slices"
    
    if dataset_type.lower() == "nq":
        json_file = Path(dir_name, f"naturalqa_slice_{size}_qa.json")
    elif dataset_type.lower() == "triviaqa":
        json_file = Path(dir_name, f"triviaqa_slice_{size}_qa.json")
    else:
        raise ValueError("dataset_type must be 'nq' or 'triviaqa'")
    
    if not json_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {json_file}")
    
    with open(json_file, "r") as f:
        qa_data = json.load(f)
        dataset = QADataset(qa_data)

    def qa_collate_fn(batch):
        ids, questions, answers = zip(*batch)
        return list(ids), list(questions), list(answers)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=qa_collate_fn
    )
    
    print(f"{dataset_type.upper()} dataset (size={size}) prepared in data loader")
    return dataloader

def run_inference(rag_manager, dataloader, output_path, test_only=True, save_every=10):
    """
    Run inference on dataset
    
    Args:
        rag_manager: Initialized RAGModelManagerOwnDataset instance
        dataloader: DataLoader with Q&A data
        output_path: Path to save results
        test_only: If True, only process first batch
    """
    if not rag_manager.is_initialized:
        raise RuntimeError("RAG model not initialized!")

    rag_manager.model.to(rag_manager.device)
    rag_manager.model.eval()

    results = []
    batch_count = 0
    
    if test_only:
        # Process only first batch
        ids, questions, _ = next(iter(dataloader))
        predictions = rag_manager.generate_answer(questions)
        
        for id_, q, pred in zip(ids, questions, predictions):
            answer = pred.strip() if pred.strip() else "--NOT FOUND--"
            results.append({
                "id": id_,
                "question": q,
                "answer": answer + "\n"
            })
            print(f"\nID: {id_}\nQ: {q}\nA: {answer}\n{'-'*40}")
    else:
        # Process entire dataset
        for ids, questions, _ in tqdm(dataloader, desc="Generating answers"):
            predictions = rag_manager.generate_answer(questions)
            
            for id_, q, pred in zip(ids, questions, predictions):
                answer = pred.strip() if pred.strip() else "--NOT FOUND--"
                results.append({
                    "id": id_,
                    "question": q,
                    "answer": answer + "\n"
                })
                
            batch_count += 1

            # Save every `save_every` batches
            if batch_count % save_every == 0:
                with open(output_path, "a") as f:
                    for entry in results:
                        f.write(json.dumps(entry) + "\n")
                print(f"[Checkpoint] Saved {len(results)} entries after {batch_count} batches.")
                results = []  # Clear after saving

    # Final flush
    if results:
        with open(output_path, "a") as f:
            for entry in results:
                f.write(json.dumps(entry) + "\n")
        print(f"[Final] Saved remaining {len(results)} entries.")

def infer_multiple_datasets(rag_manager, configs, save_every_batches=10, batch_size=8):
    """
    Run inference over multiple dataset configs with optional test mode and batch saving.
    
    Args:
        rag_manager: Initialized RAGModelManager
        configs: List of tuples (dataset_type, size, test_only)
        save_every_batches: Save results to disk after every N batches
        batch_size: DataLoader batch size
    """
    for dataset_type, size, test_only in configs:
        print(f"\n{'='*50}")
        print(f"Processing {dataset_type.upper()} dataset, size={size}")
        print(f"{'='*50}")
        
        try:
            dataloader = load_qa(dataset_type, size=size, batch_size=batch_size)
            output_path = Path("answers_replicate", f"replicate_using_local_dpr_{dataset_type}_{size}_answers.json")
            if test_only:
                output_path = output_path.with_name(output_path.stem + "_test.json")

            rag_manager.model.to(rag_manager.device)
            rag_manager.model.eval()

            results = []
            start_time = time.time()

            if test_only:
                ids, questions, _ = next(iter(dataloader))
                predictions = rag_manager.generate_answer(questions)
                for id_, q, pred in zip(ids, questions, predictions):
                    answer = pred.strip() if pred.strip() else "--NOT FOUND--"
                    results.append({"id": id_, "question": q, "answer": answer + "\n"})
                    print(f"\nID: {id_}\nQ: {q}\nA: {answer}\n{'-'*40}")
            else:
                for batch_idx, (ids, questions, _) in enumerate(dataloader):
                    predictions = rag_manager.generate_answer(questions)
                    for id_, q, pred in zip(ids, questions, predictions):
                        answer = pred.strip() if pred.strip() else "--NOT FOUND--"
                        results.append({"id": id_, "question": q, "answer": answer + "\n"})

                    if (batch_idx + 1) % save_every_batches == 0:
                        with open(output_path, "w") as f:
                            json.dump(results, f, indent=2)
                        print(f"[✓] Saved after batch {batch_idx + 1} → {len(results)} entries")
            
            # Final flush
            if results:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

            elapsed = time.time() - start_time
            print(f"✅ Completed {dataset_type.upper()} size={size} in {elapsed:.2f} sec")

        except Exception as e:
            print(f"❌ Error with {dataset_type} size={size}: {e}")

def test_first_batch(dataloader, model, tokenizer, device, output_path):
    model.to(device)
    model.eval()
    
    results = []
    
    # Get only the first batch
    ids, questions, _ = next(iter(dataloader))

    # Generate predictions
    predictions = generate_answer(questions, device, model, tokenizer)

    # Fallback to --NOT FOUND--\n
    for id_, q, pred in zip(ids, questions, predictions):
        answer = pred.strip()
        if not answer:
            answer = "--NOT FOUND--"
        results.append({
                "id": id_,
                "question": q,
                "answer": answer + "\n"  # Add newline to match format
            })
        
        print(f"\nID: {id_}\nQ: {q}\nA: {answer}\n{'-'*40}")

    # Save to file
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(results)} entries to {output_path}")

def main():
    """Main function to run the Natural Questions dataset loading and model initialization."""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load retriever, tokenizer and model
    print("Loading RAG model components...")

    rag_manager = RAGModelManagerLocalDprWiki(purpose="full") # Set to "sample" for faster testing
    rag_manager.initialize_model(device=device)
    
    print("Model components loaded successfully!")

    # Define configs
    configs = [                     # dataset_type, size, test_only
        ("nq", "350", False),       # Full inference, save every 10 batches
        ("nq", "3600", False),      # Full inference, save every 10 batches
        ("triviaqa", "1k", False),  # Full inference, save every 10 batches
        ("nq", "11000", False),     # Full inference, save every 10 batches
        ("triviaqa", "10k", False), # Full inference, save every 10 batches
        ("triviaqa", "30k", False), # Full inference, save every 10 batches
    ]

    infer_multiple_datasets(rag_manager, configs)

if __name__ == "__main__":
    main()