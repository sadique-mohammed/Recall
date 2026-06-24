"""Score the dense retriever against the labelled eval set."""

from eval.eval_harness import evaluate, load_eval_set
from rag.retriever import DenseRetriever


def main() -> None:
    eval_set = load_eval_set("eval/eval_set.jsonl")
    report = evaluate(DenseRetriever(), eval_set, k_values=(1, 3, 5, 10))
    print(report.pretty())


if __name__ == "__main__":
    main()
