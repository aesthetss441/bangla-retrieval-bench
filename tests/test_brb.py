import json

from brb import report
from brb.bm25 import BM25
from brb.cli import main
from brb.metrics import evaluate, ranks, rrf
from brb.text import bangla, char_ngrams, naive


def test_naive_tokenizer_shatters_bangla_words_and_the_aware_one_does_not():
    text = "বাংলাদেশের রাজধানী ঢাকা। Dhaka 2024"
    assert bangla(text) == ["বাংলাদেশের", "রাজধানী", "ঢাকা", "dhaka", "2024"]
    assert len(naive("বাংলাদেশের")) > 1  # \w stops at vowel signs
    assert "<দে" in char_ngrams("দেশ") and set(char_ngrams("দেশ")) < set(char_ngrams("দেশের") + ["েশ>"])


def test_metrics_on_a_hand_checked_case():
    rankings = [[5, 1, 2], [9, 8, 7], [3]]
    gold = [1, 4, 3]
    assert ranks(rankings, gold).tolist() == [2, 0, 1]
    m = evaluate(rankings, gold)
    assert abs(m["mrr@10"] - (0.5 + 0 + 1) / 3) < 1e-9
    assert abs(m["recall@1"] - 1 / 3) < 1e-9 and abs(m["recall@10"] - 2 / 3) < 1e-9
    assert m["mrr@10_ci95"][0] <= m["mrr@10"] <= m["mrr@10_ci95"][1]
    assert rrf([[1, 2, 3], [3, 1, 4]], top=2) == [1, 3]


def test_bm25_ranks_the_matching_passage_first():
    docs = ["ঢাকা বাংলাদেশের রাজধানী", "পদ্মা একটি নদী", "ক্রিকেট একটি জনপ্রিয় খেলা"]
    index = BM25([bangla(d) for d in docs])
    assert index.top(bangla("বাংলাদেশের রাজধানী কোথায়"), 3)[0] == 0
    assert index.top(bangla("নদী"), 1) == [1]


def test_end_to_end_on_custom_jsonl(tmp_path, capsys):
    docs = ["ঢাকা বাংলাদেশের রাজধানী", "পদ্মা একটি নদী", "ক্রিকেট একটি জনপ্রিয় খেলা"]
    (tmp_path / "c.jsonl").write_text("\n".join(json.dumps({"id": f"d{i}", "text": t}, ensure_ascii=False) for i, t in enumerate(docs)), encoding="utf-8")
    (tmp_path / "q.jsonl").write_text(json.dumps({"text": "রাজধানী", "doc_id": "d0"}, ensure_ascii=False) + "\n" + json.dumps({"text": "জনপ্রিয় খেলা", "doc_id": "d2"}, ensure_ascii=False), encoding="utf-8")
    assert main(["run", "--systems", "bm25-bangla,bm25-char3", "--corpus", str(tmp_path / "c.jsonl"), "--queries", str(tmp_path / "q.jsonl"), "--out", str(tmp_path / "runs")]) == 0
    result = report.write(tmp_path / "runs", tmp_path / "results", tmp_path / "docs")
    assert {r["system"] for r in result["systems"]} == {"bm25-bangla", "bm25-char3"}
    assert all(r["recall@1"] == 1.0 for r in result["systems"])
    assert (tmp_path / "docs" / "data.json").exists() and "nDCG@10" in (tmp_path / "results" / "RESULTS.md").read_text(encoding="utf-8")
