import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constrained_decoding.constrained_decoder import ConstrainedDecodingConfig  # noqa: E402
from grpo.citations import parse_completion  # noqa: E402
from grpo.rewards import yesno_label  # noqa: E402
from sft.data import build_examples  # noqa: E402

logger = logging.getLogger("grpo.data")


def infer_question_type(question: str, gold_answer: str) -> str:
    """Coarse type, used only to select R_answer's scoring branch.

    The training file carries no type field (BioASQ does, and rows built from it
    keep theirs).  A gold answer that opens with a polarity word is treated as
    yes/no; everything else is scored with ROUGE-L, which is the safe default.
    """
    opening = " ".join(gold_answer.split()[:3])
    if yesno_label(opening) is not None:
        return "yesno"
    return "summary"


def build_rows(
    records: Sequence[Dict],
    decoding_config: ConstrainedDecodingConfig,
    doc_id_style: str = "synthetic",
    max_documents: int = 0,
    require_gold_citations: bool = True,
    stats: Optional[Counter] = None,
) -> List[Dict]:
    """Turn raw training records into GRPO rows."""
    stats = stats if stats is not None else Counter()
    examples = build_examples(
        records,
        config=decoding_config,
        doc_id_style=doc_id_style,
        max_documents=max_documents,
        stats=stats,
    )

    rows: List[Dict] = []
    for example in examples:
        parsed = parse_completion(example.answer)
        if require_gold_citations and not parsed.citations:
            stats["rows_dropped_no_gold_citations"] += 1
            continue

        gold_answer = parsed.answer_text
        if not gold_answer:
            stats["rows_dropped_empty_gold_answer"] += 1
            continue

        # Preserve first-seen order so the gold lists are reproducible.
        gold_doc_ids, gold_spans = [], []
        for citation in parsed.citations:
            if citation.doc_id not in gold_doc_ids:
                gold_doc_ids.append(citation.doc_id)
            if citation.snippet not in gold_spans:
                gold_spans.append(citation.snippet)

        rows.append(
            {
                # Conversational form: TRL applies the chat template itself,
                # with chat_template_kwargs carrying the same thinking-mode flag
                # the SFT stage used.
                "prompt": [{"role": "user", "content": example.prompt}],
                "question": example.question,
                "question_type": infer_question_type(example.question, gold_answer),
                "docs": example.documents,
                "gold_doc_ids": gold_doc_ids,
                "gold_spans": gold_spans,
                "gold_answer": gold_answer,
            }
        )
        stats["rows_kept"] += 1
        stats["gold_citations"] += len(parsed.citations)

    return rows


def load_dataset_splits(
    data_path: str,
    decoding_config: ConstrainedDecodingConfig,
    eval_size: int = 256,
    max_rows: int = 0,
    seed: int = 42,
    doc_id_style: str = "synthetic",
    max_documents: int = 0,
):
    """Shuffled train/eval split as HF datasets."""
    import json

    from datasets import Dataset

    with open(data_path, encoding="utf-8") as handle:
        records = json.load(handle)

    stats = Counter()
    rows = build_rows(
        records,
        decoding_config,
        doc_id_style=doc_id_style,
        max_documents=max_documents,
        stats=stats,
    )
    random.Random(seed).shuffle(rows)
    if max_rows:
        rows = rows[:max_rows]

    eval_size = min(eval_size, max(0, len(rows) // 10))
    eval_rows, train_rows = rows[:eval_size], rows[eval_size:]
    logger.info("GRPO data: %s", dict(stats))
    logger.info("train rows: %d | eval rows: %d", len(train_rows), len(eval_rows))

    type_counts = Counter(row["question_type"] for row in train_rows)
    logger.info("question types: %s", dict(type_counts))

    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows), stats
