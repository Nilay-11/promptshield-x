"""
PromptShield X - Phase 3.6 Dataset Builder (real-world documents).

Goal: data that supports the claim "the detector works on real-world documents".
See eval/DATASETS.md section 5 for the full rationale.

Fixes carried over from the v35 audit
  * Label fixes: standalone "I want you to act as..." rows labeled 1 go to ambiguous_v36.json;
    benign persona rows carry persona=True; standalone BIPIA strings are dropped and only used
    inside their real host documents (context_dependent=True).
  * Template hard negatives replaced by real documents; templates kept only as a probe.
  * Exact + 8-gram dedupe within and across splits; label conflicts dropped; `group` is the
    bootstrap unit.

Real-world document coverage
  * Benign genres (upstream splits kept, so a document never crosses splits):
    LEDGAR contract provisions, UNFAIR-ToS clauses, Enron emails (AESLC), Wikipedia (WikiText-2),
    arXiv ML abstracts (including real papers that *discuss* prompt injection / jailbreaks,
    flagged meta_discussion=True), BIPIA emails / tables / StackOverflow answers.
  * Every document genre appears with both labels: ~30% of chunks get an attack from the
    split's own attack pool at a sentence boundary (doc_injected=True).
  * Attack sources: deepset, Gandalf (capped so it no longer dominates), rubend18, Tensor Trust
    hijacking + extraction attacks.
  * Long-document slices in val AND test (so long-document thresholds are chosen on val):
    long benign documents, padded benign instructions, and buried attacks.

final_test_v36.json (touch once; no source used in train/val/test)
  * LLMail-Inject phase 2: real attacker-written emails that targeted an email RAG assistant
    (api_triggered = the attack actually fired against a real LLM), plus the challenge's
    benign emails for false-positive testing.
  * LLMail retrieval contexts: 10 real benign emails with/without one attacker email in the middle.
  * InjecAgent tool responses, jackhhao/jailbreak-classification, no_robots prompts,
    BBC News articles (an unseen document genre, 30% with an LLMail attack embedded).

format_suite_v36_{heldout,final}.json: specs for rendered PDF / HTML documents (visible and
hidden attacks, plus benign documents that use the same hiding mechanisms for benign text).
They are rendered and scanned end-to-end by eval/evaluate_documents_v36.py.
"""

import csv
import hashlib
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

DATA_DIR = Path("eval/data")
RAW_DIR = DATA_DIR / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

PERSONA_RE = re.compile(
    r"\b(act as|acting as|i want you to act|pretend (to be|you are)|you are now|role-?play)\b", re.I
)
STANDALONE_PERSONA_RE = re.compile(r"^\s*i want you to act as\b", re.I)
META_RE = re.compile(
    r"prompt injection|jailbreak|adversarial prompt|prompt hijack|instruction[- ]following attack|"
    r"red[- ]teaming|system prompt|indirect injection", re.I
)

LLMAIL_URL = "https://huggingface.co/datasets/microsoft/llmail-inject-challenge/resolve/main/data/{}"
INJECAGENT_URL = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/{}"
ARXIV_URL = "https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers/resolve/main/ML-Arxiv-Papers.csv"

DOC_CAPS = {"train": 1200, "val": 300, "test": 400}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def gid(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:12]


def row(text, label, source, group=None, **flags):
    text = text.strip()
    r = {
        "text": text,
        "label": int(label),
        "source": source,
        "category": "INJECTION" if label == 1 else "BENIGN",
        "group": group or gid(norm(text)),
        "persona": False,
        "context_dependent": False,
        "doc_injected": False,
        "is_long": False,
        "augmented": False,
        "meta_discussion": False,
    }
    r.update(flags)
    return r


def split_list(items, fractions, rng):
    items = list(items)
    rng.shuffle(items)
    out, start = [], 0
    for i, f in enumerate(fractions):
        end = len(items) if i == len(fractions) - 1 else start + int(round(f * len(items)))
        out.append(items[start:end])
        start = end
    return out


def fetch(url: str, name: str) -> Path:
    """Download once into eval/data/raw/."""
    path = RAW_DIR / name
    if not path.exists() or path.stat().st_size == 0:
        print(f"  downloading {name} ...", flush=True)
        with urllib.request.urlopen(url, timeout=600) as resp, open(path, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
    return path


# ----------------------------------------------------------------------
# Document helpers
# ----------------------------------------------------------------------
SENT_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+|\n+")


def embed_attack(doc: str, attack: str, rng: random.Random) -> str:
    """Insert an attack at a random sentence boundary, matching the doc's casing style."""
    if doc == doc.lower():
        attack = attack.lower()
    bounds = [m.end() for m in SENT_BOUNDARY.finditer(doc)]
    pos = rng.choice([0, len(doc)] + bounds)
    left, right = doc[:pos].rstrip(), doc[pos:].lstrip()
    return " ".join(p for p in (left, attack.strip(), right) if p)


def snap_window(text: str, start: int, end: int) -> str:
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return text[start:end].strip()


def crop_around(text: str, a_start: int, a_len: int, rng: random.Random, max_len: int = 700) -> str:
    budget = max(0, rng.randint(250, max_len) - a_len)
    left = rng.randint(0, budget)
    return snap_window(text, max(0, a_start - left), min(len(text), a_start + a_len + budget - left))


def random_window(text: str, rng: random.Random, max_len: int = 700) -> str:
    length = rng.randint(250, max_len)
    if len(text) <= length:
        return text.strip()
    s = rng.randint(0, len(text) - length)
    return snap_window(text, s, s + length)


def cap_docs(texts, n, rng, lo=150, hi=1500):
    texts = sorted({t.strip() for t in texts if lo <= len(t.strip()) <= hi})
    rng.shuffle(texts)
    return texts[:n]


# ----------------------------------------------------------------------
# Attack / prompt sources
# ----------------------------------------------------------------------
def load_deepset(rng):
    ds = load_dataset("deepset/prompt-injections")
    out = {"train": [], "val": [], "test": []}
    ambiguous = []
    train_texts = sorted({r["text"].strip() for r in ds["train"]})
    _, val_texts = split_list(train_texts, [0.85, 0.15], rng)
    val_set = set(val_texts)
    for sp in ("train", "test"):
        for r in ds[sp]:
            text, label = r["text"].strip(), int(r["label"])
            target = "test" if sp == "test" else ("val" if text in val_set else "train")
            if label == 1 and STANDALONE_PERSONA_RE.search(text):
                ambiguous.append(row(text, label, f"deepset_{target}", reason="standalone_persona_labeled_injection"))
                continue
            out[target].append(row(text, label, f"deepset_{target}", persona=bool(PERSONA_RE.search(text))))
    return out, ambiguous


def load_gandalf(rng, train_cap=300):
    ds = load_dataset("Lakera/gandalf_ignore_instructions")
    train = [r["text"] for r in ds["train"]]
    rng.shuffle(train)
    return {
        "train": [row(t, 1, "gandalf_train") for t in train[:train_cap]],
        "val": [row(r["text"], 1, "gandalf_val") for r in ds["validation"]],
        "test": [row(r["text"], 1, "gandalf_test") for r in ds["test"]],
    }


def load_rubend():
    ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split="train")
    rows = []
    for r in ds:
        p = (r.get("Prompt") or r.get("text") or "").strip()
        if p:
            rows.append(row(p, 1, "rubend18_jailbreak", persona=bool(PERSONA_RE.search(p))))
    return {"train": rows, "val": [], "test": []}


def load_tensor_trust(rng):
    attacks = set()
    for f in ("benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl",
              "benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl"):
        path = hf_hub_download("qxcv/tensor-trust", f, repo_type="dataset")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                a = json.loads(line).get("attack", "").strip()
                if 10 <= len(a) <= 2000:
                    attacks.add(a)
    tr, va, te = split_list(sorted(attacks), [0.7, 0.15, 0.15], rng)
    mk = lambda xs, sp: [row(a, 1, f"tensor_trust_{sp}") for a in xs]
    return {"train": mk(tr, "train"), "val": mk(va, "val"), "test": mk(te, "test")}


def load_awesome(rng):
    ds = load_dataset("fka/awesome-chatgpt-prompts", split="train")
    prompts = sorted({r["prompt"].strip() for r in ds if r.get("prompt")})
    tr, va, te = split_list(prompts, [0.65, 0.15, 0.20], rng)
    mk = lambda xs, sp: [row(p, 0, f"awesome_prompts_{sp}", persona=bool(PERSONA_RE.search(p))) for p in xs]
    return {"train": mk(tr, "train"), "val": mk(va, "val"), "test": mk(te, "test")}


def load_dolly(rng):
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    instr = sorted({r["instruction"].strip() for r in ds if r.get("instruction")})
    rng.shuffle(instr)
    tr, va, te = instr[:1000], instr[1000:1250], instr[1250:1600]
    pad = {"val": instr[1600:1700], "test": instr[1700:1900]}
    mk = lambda xs, sp: [row(p, 0, f"dolly_{sp}") for p in xs]
    return {"train": mk(tr, "train"), "val": mk(va, "val"), "test": mk(te, "test")}, pad


# ----------------------------------------------------------------------
# Real document sources
# ----------------------------------------------------------------------
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def load_documents(rng):
    """Returns {split: [(text, genre, meta_flag)]} with upstream splits preserved."""
    out = {"train": [], "val": [], "test": []}

    ledgar = load_dataset("coastalcph/lex_glue", "ledgar")
    for up, sp in SPLIT_MAP.items():
        out[sp] += [(t, "ledgar", False) for t in cap_docs(ledgar[up]["text"], DOC_CAPS[sp], rng)]

    tos = load_dataset("coastalcph/lex_glue", "unfair_tos")
    for up, sp in SPLIT_MAP.items():
        sents = [t.strip() for t in tos[up]["text"] if t.strip()]
        chunks = [" ".join(sents[i : i + 4]) for i in range(0, len(sents) - 3, 4)]
        out[sp] += [(c, "unfair_tos", False) for c in cap_docs(chunks, DOC_CAPS[sp] // 2, rng)]

    enron = load_dataset("Yale-LILY/aeslc")
    for up, sp in SPLIT_MAP.items():
        out[sp] += [(t, "enron_email", False) for t in cap_docs(enron[up]["email_body"], DOC_CAPS[sp], rng)]

    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    for up, sp in SPLIT_MAP.items():
        paras = [l.strip() for l in wiki[up]["text"] if l.strip() and not l.strip().startswith("=")]
        out[sp] += [(t, "wikipedia", False) for t in cap_docs(paras, DOC_CAPS[sp], rng)]

    arxiv_path = fetch(ARXIV_URL, "ml_arxiv_papers.csv")
    csv.field_size_limit(10**8)
    with open(arxiv_path, encoding="utf-8", errors="replace") as f:
        abstracts = sorted({re.sub(r"\s+", " ", r.get("abstract") or "").strip() for r in csv.DictReader(f)})
    abstracts = [a for a in abstracts if 300 <= len(a) <= 1800]
    general = [a for a in abstracts if not META_RE.search(a)]
    meta = sorted(set(load_arxiv_meta()) | {a for a in abstracts if META_RE.search(a)})
    for sp, part in zip(("train", "val", "test"), split_list(meta, [0.6, 0.15, 0.25], rng)):
        out[sp] += [(a, "arxiv_meta", True) for a in part]
    rng.shuffle(general)
    start = 0
    for sp in ("train", "val", "test"):
        out[sp] += [(a, "arxiv", False) for a in general[start : start + DOC_CAPS[sp]]]
        start += DOC_CAPS[sp]
    return out


ARXIV_API = "https://export.arxiv.org/api/query?search_query={}&start={}&max_results=200"
ARXIV_META_QUERIES = ['abs:"prompt injection"', 'abs:jailbreak AND abs:"language model"',
                      'abs:"indirect prompt injection"', 'abs:"system prompt" AND abs:attack']


def load_arxiv_meta(max_per_query=1200):
    """Real abstracts that *discuss* prompt injection / jailbreaks (benign meta-discussion).
    Fetched once from the arXiv API and cached in eval/data/raw/arxiv_meta.json."""
    import time
    import xml.etree.ElementTree as ET
    from urllib.parse import quote

    path = RAW_DIR / "arxiv_meta.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    found = {}
    for q in ARXIV_META_QUERIES:
        for start in range(0, max_per_query, 200):
            print(f"  arXiv API: {q} start={start}", flush=True)
            with urllib.request.urlopen(ARXIV_API.format(quote(q), start), timeout=120) as resp:
                root = ET.fromstring(resp.read())
            entries = root.findall("a:entry", ns)
            for e in entries:
                pid = e.findtext("a:id", "", ns).rsplit("/", 1)[-1].split("v")[0]
                found[pid] = re.sub(r"\s+", " ", e.findtext("a:summary", "", ns)).strip()
            time.sleep(3)  # arXiv API etiquette
            if len(entries) < 200:
                break
    abstracts = sorted({a for a in found.values() if 300 <= len(a) <= 1800})
    path.write_text(json.dumps(abstracts, indent=1, ensure_ascii=False), encoding="utf-8")
    return abstracts


def load_bipia_contexts(rng):
    """Real BIPIA host documents: attacked window (label 1) vs clean window (label 0)."""
    ds = load_dataset("geodesic-research/bipia", split="train")
    by_doc = defaultdict(list)
    attacks = set()
    for r in ds:
        a, ctx = (r["attack_str"] or "").strip(), r["context"] or ""
        if not a or a not in ctx:
            continue
        clean = ctx.replace(a, "", 1).strip()
        by_doc[(r["task_name"], gid(norm(clean)))].append((r, a, clean))
        attacks.add(a)

    doc_split = dict(zip(("train", "val", "test"), split_list(sorted(by_doc), [0.7, 0.15, 0.15], rng)))
    atk_split = dict(zip(("train", "val", "test"), split_list(sorted(attacks), [0.6, 0.15, 0.25], rng)))

    out = {"train": [], "val": [], "test": []}
    standalone = []
    for sp in ("train", "val", "test"):
        allowed = set(atk_split[sp])
        for key in doc_split[sp]:
            task, doc_gid = key
            entries = [e for e in by_doc[key] if e[1] in allowed]
            clean = by_doc[key][0][2]
            out[sp].append(row(random_window(clean, rng), 0, f"bipia_{task}_{sp}",
                               group=f"bipia_doc_{doc_gid}", context_dependent=True))
            for r, a, _ in rng.sample(entries, min(1, len(entries))):
                ctx = r["context"]
                out[sp].append(row(crop_around(ctx, ctx.index(a), len(a), rng), 1, f"bipia_{task}_{sp}",
                                   group=f"bipia_doc_{doc_gid}", context_dependent=True,
                                   attack_group=gid(norm(a))))
        standalone += [row(a, 1, f"bipia_standalone_{sp}", reason="context_dependent_standalone")
                       for a in atk_split[sp]]
    return out, standalone


# ----------------------------------------------------------------------
# Final-test sources (never used in train / val / test)
# ----------------------------------------------------------------------
def load_llmail(rng, n_api=500, n_judge=500):
    subs = json.loads(fetch(LLMAIL_URL.format("labelled_unique_submissions_phase2.json"),
                            "llmail_phase2.json").read_text(encoding="utf-8"))
    api, judge = [], []
    for text, lab in subs.items():
        if str(lab.get("attack_attempt")) != "True" or not (40 <= len(text) <= 3000):
            continue
        (api if lab.get("reason") == "api_triggered" else judge).append(text)
    api, judge = sorted(api), sorted(judge)
    rng.shuffle(api)
    rng.shuffle(judge)
    attacks = ([row(t, 1, "llmail_attack_api_triggered", context_dependent=True) for t in api[:n_api]]
               + [row(t, 1, "llmail_attack_judge_labeled", context_dependent=True) for t in judge[:n_judge]])

    fp = json.loads(fetch(LLMAIL_URL.format("emails_for_fp_tests.json"), "emails_for_fp_tests.json")
                    .read_text(encoding="utf-8"))
    scen = json.loads(fetch(LLMAIL_URL.format("scenarios.json"), "scenarios.json").read_text(encoding="utf-8"))
    scen_emails = [e for s in scen.values() for e in s.get("emails", [])]
    benign_emails = sorted({e.strip() for e in fp + scen_emails if e.strip()})
    benign = [row(e, 0, "llmail_benign_email") for e in benign_emails]

    # Realistic retrieval contexts: 10 emails, attacker email in the middle (LLMail scenario 2/3 geometry)
    contexts = []
    atk_texts = [r["text"] for r in attacks]
    for i in range(150):
        emails = rng.sample(benign_emails, 10)
        g = f"llmail_ctx_{i}"
        contexts.append(row("\n\n".join(emails), 0, "llmail_context_benign", group=g, is_long=True))
        atk = rng.choice(atk_texts)
        mixed = emails[:5] + [atk] + emails[5:9]
        contexts.append(row("\n\n".join(mixed), 1, "llmail_context_attacked", group=g, is_long=True,
                            context_dependent=True, attack_group=gid(norm(atk))))
    return attacks, benign, contexts


def load_injecagent():
    rows = []
    for fname in ("test_cases_dh_base.json", "test_cases_ds_base.json"):
        cases = json.loads(fetch(INJECAGENT_URL.format(fname), fname).read_text(encoding="utf-8"))
        for c in cases:
            resp_text = c["Tool Response"].strip().strip('"')
            rows.append(row(resp_text, 1, f"injecagent_{fname.split('_')[2]}",
                            group=f"injecagent_{gid(norm(c['Attacker Instruction']))}",
                            context_dependent=True))
    return rows


def load_jackhhao():
    ds = load_dataset("jackhhao/jailbreak-classification")
    rows = []
    for sp in ("train", "test"):
        for r in ds[sp]:
            p = r["prompt"].strip()
            rows.append(row(p, 1 if r["type"] == "jailbreak" else 0, "jackhhao", persona=bool(PERSONA_RE.search(p))))
    return rows


def load_no_robots():
    ds = load_dataset("HuggingFaceH4/no_robots", split="test")
    return [row(r["prompt"], 0, "no_robots") for r in ds if r["prompt"].strip()]


def load_bbc(rng, n=600):
    ds = load_dataset("SetFit/bbc-news", split="test")
    texts = cap_docs([t.replace("  ", ". ") for t in ds["text"]], n, rng, lo=300, hi=3000)
    return [(t, "bbc_news", False) for t in texts]


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------
def build_doc_rows(docs, attack_pool, sp, rng, inject_frac=0.3):
    rows = []
    for text, src, meta in docs:
        if attack_pool and rng.random() < inject_frac:
            atk = rng.choice(attack_pool)
            rows.append(row(embed_attack(text, atk, rng), 1, f"{src}_{sp}", group=gid(norm(text)),
                            doc_injected=True, meta_discussion=meta, attack_group=gid(norm(atk))))
        else:
            rows.append(row(text, 0, f"{src}_{sp}", group=gid(norm(text)), meta_discussion=meta))
    return rows


def short_attacks(rows, max_chars=400):
    return sorted({r["text"] for r in rows if r["label"] == 1 and 15 <= len(r["text"]) <= max_chars})


def build_long_slice(docs, attack_pool, pad_pool, sp, rng, n):
    texts = [t for t, _, _ in docs]
    rows = []
    for i in range(n):
        paras = rng.sample(texts, rng.randint(6, 12))
        base = "\n\n".join(paras)
        g = f"long_{gid(base)}"
        rows.append(row(base, 0, f"long_benign_doc_{sp}", group=g, is_long=True))
        atk = rng.choice(attack_pool)
        k = rng.randint(2, len(paras) - 2)
        rows.append(row("\n\n".join(paras[:k] + [atk] + paras[k:]), 1, f"long_buried_attack_{sp}", group=g,
                        is_long=True, doc_injected=True, attack_group=gid(norm(atk))))
        instr = pad_pool[i % len(pad_pool)]
        pre, post = rng.sample(texts, 3), rng.sample(texts, 3)
        rows.append(row("\n\n".join(pre + [instr] + post), 0, f"padded_benign_instruction_{sp}",
                        group=f"pad_{gid(instr)}", is_long=True))
    return rows


def augment(rows, rng, frac=0.2):
    def perturb(text):
        choice = rng.choice(["homoglyph", "leet", "spacing"])
        if choice == "homoglyph":
            return text.replace("a", "а").replace("e", "е").replace("o", "о")
        if choice == "leet":
            return text.replace("e", "3").replace("i", "1").replace("o", "0")
        words = text.split()
        idx = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
        if idx:
            j = rng.choice(idx)
            words[j] = " ".join(words[j])
        return " ".join(words)

    out = []
    for r in rng.sample(rows, int(frac * len(rows))):
        t = perturb(r["text"])
        if t != r["text"]:
            out.append(dict(r, text=t, source=f"{r['source']}_augmented", augmented=True))  # keeps group
    return out


# ----------------------------------------------------------------------
# Format suite specs (rendered + scanned by eval/evaluate_documents_v36.py)
# ----------------------------------------------------------------------
PDF_METHODS = ["visible", "microfont", "white_text", "off_canvas", "metadata"]
HTML_METHODS = ["visible", "display_none", "html_comment", "aria_hidden", "zero_width"]


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if 20 <= len(s.strip()) <= 300]


def build_format_specs(docs, attack_pool, rng, n_docs):
    """Each document is rendered once per method with an attack and once with benign text hidden the
    same way (a real sentence from another document: footnote, boilerplate, comment)."""
    texts = [t for t, _, _ in docs]
    benign_snippets = [s for t in texts for s in sentences(t)]
    specs = []
    for i, (body, genre, _) in enumerate(rng.sample(docs, min(n_docs, len(docs)))):
        g = f"fmt_{gid(body)}"
        for fmt, methods in (("pdf", PDF_METHODS), ("html", HTML_METHODS)):
            for m in methods:
                specs.append({"format": fmt, "method": m, "label": 1, "genre": genre, "group": g,
                              "body": body, "payload": rng.choice(attack_pool)})
                specs.append({"format": fmt, "method": m, "label": 0, "genre": genre, "group": g,
                              "body": body, "payload": rng.choice(benign_snippets)})
    return specs


def ngrams(text, n=8):
    w = norm(text).split()
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


def dedupe(splits_in_order):
    """Exact-normalized dedupe within splits and across earlier splits, plus 8-gram containment.
    Label conflicts within a split are dropped and returned."""
    seen_text, seen_grams = set(), set()
    removed = Counter()
    conflicts = []
    out = {}
    for name, rows in splits_in_order:
        labels = defaultdict(set)
        for r in rows:
            labels[norm(r["text"])].add(r["label"])
        kept, local = [], set()
        for r in rows:
            k = norm(r["text"])
            if len(labels[k]) > 1:
                if k not in local:
                    conflicts.append(dict(r, reason=f"label_conflict_in_{name}"))
                    local.add(k)
                removed[(name, "conflict")] += 1
                continue
            if k in local:
                removed[(name, "dup_within")] += 1
                continue
            if k in seen_text:
                removed[(name, "dup_earlier_split")] += 1
                continue
            g = ngrams(r["text"])
            # doc_injected / long rows reuse attack strings by design; their host docs are split-disjoint
            if g and not r["is_long"] and not r["doc_injected"] and len(g & seen_grams) / len(g) > 0.5:
                removed[(name, "near_dup_earlier_split")] += 1
                continue
            local.add(k)
            kept.append(r)
        out[name] = kept
        seen_text |= local
        for r in kept:
            if not r["augmented"]:
                seen_grams |= ngrams(r["text"])
    return out, removed, conflicts


def label_conflicts_across(splits):
    lab = defaultdict(set)
    for rows in splits.values():
        for r in rows:
            lab[norm(r["text"])].add(r["label"])
    return sum(1 for v in lab.values() if len(v) > 1)


CSV_FIELDS = ["text", "label", "source", "category", "group", "persona", "context_dependent",
              "doc_injected", "is_long", "augmented", "meta_discussion"]


def write_split(name, rows):
    (DATA_DIR / f"{name}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(DATA_DIR / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summarize(name, rows):
    by_src = defaultdict(Counter)
    for r in rows:
        by_src[r["source"]][r["label"]] += 1
    uniq = len({norm(r["text"]) for r in rows})
    groups = len({r["group"] for r in rows})
    print(f"\n{name}: {len(rows)} rows | unique texts {uniq} | groups {groups} | "
          f"inj {sum(r['label'] for r in rows)} | benign {sum(1 - r['label'] for r in rows)}")
    for s in sorted(by_src):
        print(f"   {s:40s} benign={by_src[s][0]:5d}  inj={by_src[s][1]:5d}")
    return {"rows": len(rows), "unique": uniq, "groups": groups,
            "by_source": {s: {"benign": c[0], "injection": c[1]} for s, c in sorted(by_src.items())}}


def build():
    rng = random.Random(SEED)
    print("Loading sources...", flush=True)
    deepset, ambiguous = load_deepset(rng)
    gandalf = load_gandalf(rng)
    rubend = load_rubend()
    tensor_trust = load_tensor_trust(rng)
    awesome = load_awesome(rng)
    dolly, pad_pools = load_dolly(rng)
    docs = load_documents(rng)
    bipia, bipia_standalone = load_bipia_contexts(rng)

    splits, pools = {}, {}
    for sp in ("train", "val", "test"):
        attack_rows = deepset[sp] + gandalf[sp] + rubend[sp] + tensor_trust[sp]
        pools[sp] = short_attacks(attack_rows)
        rows = (attack_rows + awesome[sp] + dolly[sp] + bipia[sp]
                + build_doc_rows(docs[sp], pools[sp], sp, rng))
        if sp == "train":
            rows += augment(rows, rng)
        else:
            rows += build_long_slice(docs[sp], pools[sp], pad_pools[sp], sp, rng, n=60 if sp == "val" else 100)
        rng.shuffle(rows)
        splits[sp] = rows

    print("Loading final-test sources (LLMail-Inject, InjecAgent, jackhhao, no_robots, BBC)...", flush=True)
    llm_attacks, llm_benign, llm_contexts = load_llmail(rng)
    bbc_docs = load_bbc(rng)
    llmail_pool = [r["text"] for r in llm_attacks if len(r["text"]) <= 1200]
    final = (llm_attacks + llm_benign + llm_contexts + load_injecagent() + load_jackhhao() + load_no_robots()
             + build_doc_rows(bbc_docs, llmail_pool, "final", rng))
    rng.shuffle(final)

    deduped, removed, conflicts = dedupe([
        ("train", splits["train"]), ("val", splits["val"]),
        ("test", splits["test"]), ("final", final),
    ])
    ambiguous += conflicts + bipia_standalone

    names = {"train": "train_v36", "val": "val_v36", "test": "heldout_test_v36", "final": "final_test_v36"}
    report = {"seed": SEED, "removed": {f"{k[0]}/{k[1]}": v for k, v in sorted(removed.items())}}
    for key, fname in names.items():
        write_split(fname, deduped[key])
        report[fname] = summarize(fname, deduped[key])
    report["cross_split_label_conflicts"] = label_conflicts_across(deduped)

    (DATA_DIR / "ambiguous_v36.json").write_text(json.dumps(ambiguous, indent=2, ensure_ascii=False), encoding="utf-8")
    report["ambiguous_rows"] = dict(Counter(r.get("reason", "?") for r in ambiguous))

    # Rendered-document suites
    llmail_benign_docs = [(r["text"], "llmail_email", False) for r in llm_benign]
    fmt = {
        "format_suite_v36_heldout": build_format_specs(docs["test"], pools["test"], rng, n_docs=100),
        "format_suite_v36_final": build_format_specs(bbc_docs + llmail_benign_docs, llmail_pool, rng, n_docs=60),
        # document-level thresholds are chosen here, never on heldout/final
        "format_suite_v36_val": build_format_specs(docs["val"], pools["val"], rng, n_docs=200),
    }
    for name, specs in fmt.items():
        (DATA_DIR / f"{name}.json").write_text(json.dumps(specs, indent=2, ensure_ascii=False), encoding="utf-8")
        report[name] = len(specs)
        print(f"\n{name}: {len(specs)} rendered-document specs")

    # v35 template families: regression probe only, never trained on
    tmpl_path = DATA_DIR / "heldout_test_v35.json"
    if tmpl_path.exists():
        v35 = json.loads(tmpl_path.read_text(encoding="utf-8"))
        tmpl = {norm(r["text"]): r for r in v35 if r.get("is_hard_negative")}
        tmpl_rows = [row(r["text"], 0, r["source"].replace("test_family", "template")) for r in tmpl.values()]
        (DATA_DIR / "diagnostic_templates_v36.json").write_text(
            json.dumps(tmpl_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        report["diagnostic_templates"] = len(tmpl_rows)

    print("\nRemoved:", json.dumps(report["removed"], indent=1))
    print("Ambiguous rows:", report["ambiguous_rows"])
    print("Cross-split label conflicts:", report["cross_split_label_conflicts"])
    (DATA_DIR / "dataset_report_v36.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build()
