import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, AutoTokenizer
from tqdm import tqdm
import time

class QADataset(Dataset):
    """Generic Q&A Dataset class for both NQ and TriviaQA"""
    def __init__(self, qa_input):
        self.qa_input = qa_input
    
    def __len__(self):
        return len(self.qa_input)
    
    def __getitem__(self, idx):
        item = self.qa_input[idx]
        return item["id"], item["question"], item["answer"]

class RAGModelManager:
    """Class to manage RAG model components and inference"""
    
    def __init__(self):
        self.tokenizer = None
        # self.retriever = None
        self.model = None
        self.device = None
        self.is_initialized = False
    
    def initialize_model(self, device=None, use_dummy_dataset=True):
        """Initialize RAG model components once"""
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
        
        # # trying new version as suggested on hugging face how to use it to load model directly
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
        # self.model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
        
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        self.retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq", # default: nq embeddings, split psgs-w100,
            index_name="exact", # index_name = "compressed" or "exact"
            use_dummy_dataset=use_dummy_dataset,
            trust_remote_code=True
        )
        
        self.model = RagTokenForGeneration.from_pretrained(
            "facebook/rag-token-nq", 
            retriever=self.retriever
        )
        self.model.to(self.device)
        self.model.eval()
        self.is_initialized = True
        print(f"Model components loaded successfully on {self.device}!")
    
    def generate_answer(self, queries):
        """Generate answers for a batch of questions"""
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
            
        # # Determine device
        # if device is None:
        #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # else:
        #     self.device = device
        # print(f"Using device: {self.device}")

        # Ensure model is on correct device (in case it was moved)
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
        size: dataset size (350, 3600, 11000, 1k, 10k, 30k)
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
        rag_manager: Initialized RAGModelManager instance
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
            output_path = Path(f"replicate_DummyRetriever_{dataset_type}_{size}_answers.json")
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

                # Final save
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

    rag_manager = RAGModelManager()
    rag_manager.initialize_model(device=device, use_dummy_dataset=True)  # Set to True for faster testing
    
    print("Model components loaded successfully!")

    # Define configs
    configs = [                     # dataset_type, size, test_only
        ("nq", "350", False),       # Full inference, save every 10 batches
        ("triviaqa", "1k", False),  # Full inference, save every 10 batches
        ("nq", "3600", False),      # Full inference, save every 10 batches
        ("nq", "11000", False),     # Full inference, save every 10 batches
        ("triviaqa", "10k", False), # Full inference, save every 10 batches
        ("triviaqa", "30k", False), # Full inference, save every 10 batches
    ]

    infer_multiple_datasets(rag_manager, configs)

if __name__ == "__main__":
    main()
