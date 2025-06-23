#!/usr/bin/env python3
import argparse
import re, string, math, json
from collections import Counter

# Lower, remove punctuation/articles/extra whitespace.
def normalize_answer(s):
    # if it isn’t a string, bail out as “empty”
    if not isinstance(s, str):
        return ""
        
    s = s.lower() # lowercase
    s = s.replace("_", " ") # replace underscores with spaces
    s = re.sub(r"\b(a|an|the)\b", " ", s) # remove_articles
    s = re.sub(f"[{re.escape(string.punctuation)}]", "", s)
    return " ".join(s.split()) # white_space_fix

def compute_exact(qu, a_pred, a_true):
    print(f"Question: {qu}")
    print(f"Prediction: {a_pred}")
    print(f"Reference: {a_true}\n")
    return int(normalize_answer(a_pred) == normalize_answer(a_true))
    
def compute_f1(a_pred, a_true):
    pred_tokens = normalize_answer(a_pred).split()
    true_tokens = normalize_answer(a_true).split()
    if not pred_tokens or not true_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(true_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(true_tokens)
    return 2 * precision * recall / (precision + recall)

# Compute corpus-level BLEU score for n-grams up to max_n (default 4).
# Returns BLEU score (0 to 1).
def corpus_bleu(preds, refs, max_n=4):
    # Normalize all strings
    preds_norm = [normalize_answer(p) for p in preds]
    refs_norm  = [normalize_answer(r) for r in refs]

    # Tokenize
    preds_tokens = [p.split() for p in preds_norm]
    refs_tokens  = [r.split() for r in refs_norm]
    
    # Calculate n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        clipped_matches = 0
        total_ngrams    = 0
        for pred, ref in zip(preds_tokens, refs_tokens):
            # Count n-grams in prediction and reference
            pred_ngrams = Counter(tuple(pred[i:i+n]) for i in range(len(pred)-n+1))
            ref_ngrams  = Counter(tuple(ref[i:i+n])  for i in range(len(ref)-n+1))
            total_ngrams += sum(pred_ngrams.values())
            # Clip by max reference counts
            for ng in pred_ngrams:
                clipped_matches += min(pred_ngrams[ng], ref_ngrams.get(ng, 0))
        if total_ngrams == 0:
            precisions.append(0)
        else:
            precisions.append(clipped_matches / total_ngrams)
    
    # Brevity penalty
    pred_len = sum(len(p) for p in preds_tokens)
    ref_len  = sum(len(r) for r in refs_tokens)
    if pred_len == 0:
        return 0.0
    bp = math.exp(1 - ref_len / pred_len) if pred_len < ref_len else 1.0
    
    # Geometric mean of precisions
    if min(precisions) == 0:
        geo_mean = 0
    else:
        geo_mean = math.exp(sum((1/max_n) * math.log(p) for p in precisions))
    return bp * geo_mean

# Compute average ROUGE-L F1 score over the corpus.
# Returns average ROUGE-L F1 (0 to 1).
def rouge_l_score(preds, refs):
    def lcs(a, b):
        # Compute length of longest common subsequence
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i, ca in enumerate(a, 1):
            for j, cb in enumerate(b, 1):
                dp[i][j] = dp[i-1][j-1] + 1 if ca == cb else max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]
    
    scores = []
    for pred, ref in zip(preds, refs):
        p_tokens = normalize_answer(pred).split()
        r_tokens = normalize_answer(ref).split()

        if not p_tokens or not r_tokens:
            scores.append(0.0)
            continue
        lcs_len = lcs(p_tokens, r_tokens)
        prec = lcs_len / len(p_tokens)
        rec  = lcs_len / len(r_tokens)
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        scores.append(f1)
    return sum(scores) / len(scores) if scores else 0.0

# Count how many predictions are empty after stripping whitespace
def compute_missing(preds):
    missing = 0
    for p in preds:
        # Check for placeholder before normalization
        if p.strip() == "--NOT FOUND--":
            missing += 1
            continue
        # Otherwise, normalize and check emptiness
        if not normalize_answer(p):
            missing += 1
    total   = len(preds)
    return missing / total if total else 0.0

# Returns (ids, questions, preds, refs) aligned by id.
def align_preds_refs(pred_list, ref_list):
     # maps: id → (question, answer)
    pred_map = {int(d["id"]): (d["question"], d["answer"]) for d in pred_list}
    ref_map  = {int(d["id"]): (d["question"], d["answer"]) for d in ref_list}

    common = sorted(set(pred_map) & set(ref_map))
    questions, preds, refs = [], [], []
    for i in common:
        pq, pa = pred_map[i]
        rq, ra = ref_map[i]
        # optional sanity check that the questions match
        if pq != rq:
            print(f"ID {i} had different questions → pred: {pq!r}, ref: {rq!r}")
        questions.append(pq)
        preds.append(pa)
        refs.append(ra)
    return common, questions, preds, refs

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG predictions against references."
    )
    parser.add_argument(
        "--preds_refs",
        nargs=2,
        metavar=("PRED_JSON", "REF_JSON"),
        help="Two JSON files: first is list of prediction strings, second is list of reference strings"
    )

    args = parser.parse_args()

    # Load the two JSON files, each a list of {"id":…, "prediction"/"reference":…}
    pred_file, ref_file = args.preds_refs
    raw_preds = json.load(open(pred_file, "r", encoding="utf-8"))
    raw_refs  = json.load(open(ref_file,  "r", encoding="utf-8"))

    # Project to only id, question, answer
    pred_list = [
        {"id": item["id"], "question": item["question"], "answer": item["answer"]}
        for item in raw_preds
        if all(k in item for k in ("id","question","answer"))
    ]
    ref_list = [
        {"id": item["id"], "question": item["question"], "answer": item["answer"]}
        for item in raw_refs
        if all(k in item for k in ("id","question","answer"))
    ]
    
    # Align by ID
    ids, questions, preds, refs = align_preds_refs(pred_list, ref_list)

    # Compute metrics    
    ems     = [compute_exact(q, p, r) for q, p, r in zip(questions, preds, refs)]
    f1s     = [compute_f1(p, r)   for p, r in zip(preds, refs)]
    bleu    = corpus_bleu(preds, refs)
    rouge_l = rouge_l_score(preds, refs)
    miss    = compute_missing(preds)

    print(f"Exact Match: {100*sum(ems)/len(ems):.2f}%")
    print(f"F1 Score:    {100*sum(f1s)/len(f1s):.2f}%")
    print(f"BLEU:        {bleu*100:.2f}%")
    print(f"ROUGE-L:     {rouge_l*100:.2f}%")
    print(f"Missing Answers: {miss:.2%}")

if __name__ == "__main__":
    main()