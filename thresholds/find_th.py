#!/usr/bin/env python3
import argparse
import json
from typing import List, Tuple, Dict, Optional
from datasets import load_dataset
from multiprocessing import Pool
from functools import partial

def find_threshold_for_constraints(
    pairs: List[Tuple[float, bool]], 
    min_recall: float = 0.0, 
    max_recall: float = 1.0,
    min_precision: float = 0.0, 
    max_precision: float = 1.0,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None
) -> Tuple[Optional[float], int, int]:
    """
    Returns (l, kept_count, total_count) where strict predicate is score > l.
    Evaluates all possible thresholds and selects the one that satisfies min_recall
    and has the highest precision among those satisfying all constraints.
    If not achievable, returns (None, 0, total_count).
    
    Args:
        min_score: If provided, threshold will not be set below this value
        max_score: If provided, threshold will not be set above this value
    """
    total = len(pairs)
    if total == 0:
        return None, 0, 0

    # Aggregate by score for deterministic sweep
    by_score: Dict[float, Tuple[int, int]] = {}  # score -> (total_count, correct_count)
    for s, ok in pairs:
        tot, cor = by_score.get(s, (0, 0))
        by_score[s] = (tot + 1, cor + (1 if ok else 0))

    unique_scores = sorted(by_score.keys(), reverse=True)  # descending for highest first
    total_count = sum(t for t, _ in by_score.values())
    total_true = sum(c for _, c in by_score.values())

    if total_true == 0:
        return None, 0, total_count

    # Evaluate all thresholds and find the best one
    best_threshold = None
    best_f1_weighted = -1.0
    best_kept_count = 0
    
    included_total = 0
    included_correct = 0

    for s in unique_scores:
        group_total, group_correct = by_score[s]
        included_total += group_total
        included_correct += group_correct
        
        if included_total == 0:
            continue
        
        # Check if potential threshold would be below min_score or above max_score
        potential_threshold = s - 1e-12  # Use strict >
        if min_score is not None and potential_threshold < min_score:
            continue
        if max_score is not None and potential_threshold > max_score:
            continue
            
        # Calculate metrics for samples with score > (s - epsilon)
        recall = included_correct / total_true
        precision = included_correct / included_total
        beta = 0.1
        f1_weighted = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if ((beta**2 * precision) + recall) > 0 else 0.0
        
        # Check if all constraints are satisfied
        if (min_recall <= recall <= max_recall and 
            min_precision <= precision <= max_precision):
            # This threshold satisfies constraints - check if it has better precision
            if f1_weighted > best_f1_weighted:
                best_f1_weighted = f1_weighted
                best_threshold = potential_threshold
                best_kept_count = included_total

    if best_threshold is None:
        return None, 0, total_count
    else:
        return best_threshold, best_kept_count, total_count


def _process_one_language(
    lang: str,
    dataset_name: str,
    score_field: str,
    min_recall: float,
    max_recall: float,
    min_precision: float,
    max_precision: float,
    min_score: Optional[float],
    max_score: Optional[float]
):
    """
    Process one language and return results.
    
    Returns a row:
    [lang, samples, samples_true, trivial_accuracy, precision_threshold, kept_rate, kept_samples, precision, recall, f1]
    """
    print(f"Processing {lang}")
    
    try:
        # Load dataset from Hugging Face Hub
        dataset = load_dataset(dataset_name, lang, split='train')
        # Extract pairs (score, label)
        pairs = []
        for item in dataset:
            score = item.get(score_field, 0)
            label = item.get("language_correct", False)
            if isinstance(label, bool) and score != 0:
                pairs.append((score, label))
                
    except Exception as e:
        print(f"Error loading {lang}: {e}")
        return [lang, 0, 0, 0.0, "", 0, 0.0, 0.0, 0.0]

    total = len(pairs)
    num_true = sum(1 for _, ok in pairs if ok)
    trivial_acc = (num_true / total) if total > 0 else 0.0

    l, kept_count, total_count = find_threshold_for_constraints(
        pairs, min_recall, max_recall, min_precision, max_precision, min_score, max_score
    )
    
    if l is None:
        threshold_val = ""
        kept_samples = 0
        precision_val = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        kept_samples = kept_count
        threshold_val = l
        kept_true = sum(1 for s, ok in pairs if (s > l) and ok)
        recall = (kept_true / num_true) if num_true > 0 else 0.0
        precision_val = (kept_true / kept_samples) if kept_samples > 0 else 0.0
        f1 = 2 * (precision_val * recall) / (precision_val + recall) if (precision_val + recall) > 0 else 0.0

    return [
        lang,
        total,
        num_true,
        round(trivial_acc, 6),
        threshold_val,
        kept_samples,
        round(precision_val, 6),
        round(recall, 6),
        round(f1, 6),
    ]


def main():
    parser = argparse.ArgumentParser(description="Find thresholds for language classification from Hugging Face dataset.")
    parser.add_argument("--dataset",
                        help="Hugging Face dataset name", default="HuggingFaceFW/finepdfs_lang_classification")
    parser.add_argument("--languages", required=True, nargs='+',
                        help="List of language codes to process")

    # Constraint options
    parser.add_argument("--min-recall", type=float, default=0.0,
                        help="Minimum recall constraint [0,1] (default: 0.0)")
    parser.add_argument("--max-recall", type=float, default=1.0,
                        help="Maximum recall constraint [0,1] (default: 1.0)")
    parser.add_argument("--min-precision", type=float, default=0.0,
                        help="Minimum precision constraint [0,1] (default: 0.0)")
    parser.add_argument("--max-precision", type=float, default=1.0,
                        help="Maximum precision constraint [0,1] (default: 1.0)")
    parser.add_argument("--min-score", type=float, default=None,
                        help="Minimum language score threshold - thresholds below this value will not be set (default: None)")
    parser.add_argument("--max-score", type=float, default=None,
                        help="Maximum language score threshold - thresholds above this value will not be set (default: None)")

    parser.add_argument("--score_field", default="best_page_average_score",
                        help="Score field name (default: best_page_average_score)")
    
    parser.add_argument("--th_file", type=str, required=True,
                        help="Path to JSON file where thresholds will be saved (languages as keys, thresholds as values)")
    
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes for multiprocessing (default: 4)")

    args = parser.parse_args()

    # Clamp values to valid range
    args.min_recall = max(0.0, min(1.0, args.min_recall))
    args.max_recall = max(0.0, min(1.0, args.max_recall))
    args.min_precision = max(0.0, min(1.0, args.min_precision))
    args.max_precision = max(0.0, min(1.0, args.max_precision))

    # Create a partial function with fixed arguments for multiprocessing
    process_func = partial(
        _process_one_language,
        dataset_name=args.dataset,
        score_field=args.score_field,
        min_recall=args.min_recall,
        max_recall=args.max_recall,
        min_precision=args.min_precision,
        max_precision=args.max_precision,
        min_score=args.min_score,
        max_score=args.max_score
    )

    # Process languages in parallel
    with Pool(processes=args.workers) as pool:
        results = pool.map(process_func, args.languages)

    thresholds_dict = {}
    
    for row in results:
        print(row)
        threshold_val = row[4]  # language_score_threshold is at index 4
        if threshold_val != "":
            thresholds_dict[row[0]] = threshold_val  # row[0] is the language

    # Save thresholds to JSON file
    with open(args.th_file, 'w', encoding='utf-8') as f:
        json.dump(thresholds_dict, f, indent=2)
    
    print(f"Processed {len(results)} languages.")
    print(f"Saved thresholds for {len(thresholds_dict)} languages to {args.th_file}")


if __name__ == "__main__":
    main()