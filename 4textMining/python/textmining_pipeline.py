"""
Comprehensive text-mining pipeline for Sejm X-term data.

Stages
1) Load interpellations and plenary statements (stenograms).
2) Clean HTML → plain text, tokenize, stem, remove stopwords.
3) Compute per-document metrics: sentiment (lexicon), Gunning FOG, length.
4) TF–IDF by club & source; topic models (LDA) per source.
5) Aggregations saved to 4textMining/data/processed for R/Quarto viz.

Run:
    python3 4textMining/python/textmining_pipeline.py
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from stopwordsiso import stopwords
from tqdm import tqdm

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "1dataDownload" / "data" / "raw" / "term10"
TABLES_ROOT = REPO_ROOT / "1dataDownload" / "data" / "tables" / "term10"
OUTPUT_DIR = REPO_ROOT / "4textMining" / "data" / "processed"
LEXICON_PATH = REPO_ROOT / "4textMining" / "data" / "slownik_wydzwieku.csv"

# Globals
STOPWORDS_PL = set(stopwords("pl")) | {
    # parliamentary fillers & titles
    "pan",
    "pani",
    "panie",
    "panią",
    "posłanka",
    "posłanki",
    "posłem",
    "posła",
    "posłowi",
    "posłów",
    "posłom",
    "poseł",
    "posłowie",
    "wysoki",
    "wysoka",
    "sejm",
    "marszałek",
    "marszałku",
    "rząd",
    "rządu",
    "klub",
    "klubu",
    "klubie",
    "klubowy",
    "klubowe",
    "klubowej",
    "klubowego",
    "klubowych",
    "sekretarz",
    "sekretarza",
    "sekretarzem",
    "sekretarzu",
    "przewodniczący",
    "przewodnicząca",
    "wicemarszałek",
    "wicemarszałku",
    "marszałkini",
    "sala",
    "salę",
    "salą",
    "głos",
    "głosy",
    "oklaski",
    "wrzawa",
    "porządek",
    "porządku",
    "druk",
    "punkt",
    "dziennego",
    "dziennej",
    "dziennym",
    "dziennych",
    "dzienny",
    "izba",
    "izbie",
    "izbo",
    "izbę",
    "oświadczenie",
    "oświadczenia",
    "oświadczeń",
    "oświadczeniu",
    "parlamentarny",
    "parlamentarna",
    "parlamentarne",
    "parlamentarnych",
    "parlamentarnego",
    "parlamentarnym",
    "parlamentarnymi",
    "interpelacja",
    "interpelacje",
    "interpelacji",
    "interpelacjami",
    "imieniu",
    "imieniem",
    "koalicja",
    "koalicji",
    "koalicję",
    "koalicyjny",
    "koalicyjna",
    "koalicyjne",
    "ustawa",
    "ustawy",
    "ustawie",
    "ustaw",
    "posiedzenie",
    "posiedzenia",
    "posiedzeniu",
    "kadencja",
    "kadencji",
    "dzień",
    "dniem",
    "dniu",
    "sprawie",
    "sprawy",
    "sprawą",
    "spraw",
    "proszę",
    "dziękuję",
    "państwo",
    "państwa",
    # html artefacts / procedural
    "pkt",
    "ust",
    "art",
    "nbsp",
    "www",
    "http",
    "https",
    "dler",
}
TOKEN_RE = re.compile(r"[a-zA-ZąćęłńóśżźĄĆĘŁŃÓŚŻŹ]+", re.UNICODE)
VOWELS = set("aeiouyąęóAEIOUYĄĘÓ")
NAME_STOPLIST: set[str] = set()
ADJ_SUFFIXES = [
    "skiego",
    "ckiego",
    "zkiego",
    "skiej",
    "ckiej",
    "zkiej",
    "skim",
    "ckim",
    "zkim",
    "ską",
    "cką",
    "zką",
    "ska",
    "cka",
    "zka",
    "ski",
    "cki",
    "zki",
    "owa",
    "owe",
    "owy",
    "owej",
    "owym",
    "ego",
    "emu",
    "ej",
    "ie",
    "ich",
    "imi",
    "ami",
    "ach",
    "ące",
    "ący",
    "ąca",
    "ejszy",
    "ejsza",
]


def normalize_ascii(text: str) -> str:
    """Strip accents to ASCII-only for looser matching (e.g., pawel vs. paweł)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def build_name_stoplist(mp_df: pd.DataFrame, stmt_df: pd.DataFrame | None = None) -> set[str]:
    """Collect first/last-name tokens to drop from features (MPs + mówcy)."""
    names = set()

    # Common Polish case endings to catch odmiany (dopełniacz, celownik, narzędnik…)
    CASE_SUFFIXES = [
        "",
        "a",
        "u",
        "owi",
        "em",
        "ie",
        "ą",
        "ę",
        "y",
        "om",
        "ami",
        "ach",
        "ów",
        "owie",
        "ego",
        "owej",
        "owym",
        "owa",
        "skie",
        "skim",
        "skiego",
        "skiej",
        "ckie",
        "ckim",
        "ckiego",
        "ckiej",
    ]

    def expand(part: str) -> set[str]:
        """Generate a rich set of name variants incl. diacritics & declensions."""
        part = part.lower()
        variants = set()

        bases = {part, normalize_ascii(part)}

        # handle ł -> l (e.g., paweł -> pawel / pawł- forms)
        if "ł" in part:
            bases.add(part.replace("ł", "l"))
        # special: names ending with "eł" (Paweł -> Pawł-)
        if part.endswith("eł"):
            bases.add(part[:-2] + "ł")
            bases.add(part[:-2] + "l")

        for base in bases:
            for suf in CASE_SUFFIXES:
                variants.add(base + suf)

        # adjectival surnames ski/cka/cki – add gendered & case forms
        for base in list(bases):
            if base.endswith("ski"):
                stem = base[:-3]
                variants |= {
                    stem + "ski",
                    stem + "ska",
                    stem + "skiego",
                    stem + "skiej",
                    stem + "skim",
                    stem + "scy",
                    stem + "skich",
                }
            if base.endswith("cki"):
                stem = base[:-3]
                variants |= {
                    stem + "cki",
                    stem + "cka",
                    stem + "ckiego",
                    stem + "ckiej",
                    stem + "ckim",
                    stem + "ccy",
                    stem + "ckich",
                }
            if base.endswith("ska"):
                stem = base[:-3]
                variants |= {stem + "ski", stem + "skiej", stem + "ską"}
            if base.endswith("cka"):
                stem = base[:-3]
                variants |= {stem + "cki", stem + "ckiej", stem + "cką"}

        return variants

    def add_names(series):
        for full in series.astype(str):
            for part in re.sub(r"[-,]", " ", full).split():
                part = part.lower()
                if len(part) > 2:
                    names.update(expand(part))

    add_names(mp_df["name"])
    if stmt_df is not None and "name" in stmt_df.columns:
        add_names(stmt_df["name"])

    # manual high-frequency politycy spoza Sejmu X, żeby nie psuli log-odds
    manual = {
        "tusk",
        "kaczyński",
        "morawiecki",
        "trzaskowski",
        "duda",
        "kiełbasa",
        "bosak",
        "mentzen",
        "hołownia",
        "czarzasty",
        "zandberg",
        "marszałkin",
        "marszałek",
        "dąbrowska",
        "banaszek",
        "dąbrowski",
        "trzaskowskiego",
        "górczewska",
        "bodnar",
        "bodnara",
        "rafala",
        "rafał",
        "pawel",
        "paweł",
    }
    names |= manual
    # add ASCII-fallbacks for all names to catch tokens po uproszczeniu znaków
    names |= {normalize_ascii(n) for n in list(names)}
    return names


@dataclass
class Document:
    doc_id: str
    source: str  # "interpelacja" | "stenogram"
    mp_id: int
    club: str
    date: pd.Timestamp
    weight: float
    text: str


def html_to_text(path: Path) -> str:
    """Extract speaker text; drop ogłoszenia, sceniczne wtręty i zapowiedzi mówców."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    # zachowaj podziały linii, żeby łatwiej odfiltrować zapowiedzi
    text = soup.get_text(separator="\n")

    # usuń elementy w nawiasach – oklaski, wrzawa, dopiski stenografa, wywoływanie posłów
    text = re.sub(
        r"\([^)]*(Oklaski|Głos z sali|Wesołość|Dzwonek|Śmiech|ha, ha|Poseł|Posłanka|Marszałek|Sekretarz)[^)]*\)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\[[^\]]+\]", " ", text)  # [Oklaski]

    cleaned: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()

        # zapowiedzi / prowadzenie obrad
        if re.search(r"^(marszał|wicemarszał|sekretar|przewodnicz|ogłaszam|otwieram|zamykam|wznawiam)", lower):
            continue
        if re.search(r"^(panie|pan)\\s+po?s(ł|e)ł", lower) or lower.startswith("poseł") or lower.startswith("posłanka"):
            continue
        if "w imieniu klubu" in lower or "w imieniu koła" in lower:
            continue
        if "klubu parlamentarnego" in lower or "klub parlamentarny" in lower:
            continue
        if "parlamentarnej grupy" in lower or "koła parlamentarnego" in lower:
            continue
        if "udzielam głosu" in lower or "głos ma" in lower or "głos zabierze" in lower:
            continue
        if "dopomóż bóg" in lower:
            continue
        if "porządku dzien" in lower:
            continue
        if re.match(r"^\\d+\\.\\s*(posiedzenie|punkt)", lower):
            continue

        cleaned.append(line)

    text = " ".join(cleaned)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Lowercase, regex-tokenize, drop stopwords and MP names; keep full forms."""
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    filtered = []
    for t in tokens:
        if len(t) <= 2:
            continue
        if t in STOPWORDS_PL:
            continue
        # require at least one vowel to skip skróty typu "sch" / "rpm"
        if not any(ch in VOWELS for ch in t):
            continue
        norm = normalize_ascii(t)
        if t in NAME_STOPLIST or norm in NAME_STOPLIST:
            continue
        filtered.append(t)
    return filtered


def fold_token(token: str) -> str:
    """Very light stemming to merge inflected adjective/nominal endings."""
    base = token
    for suf in ADJ_SUFFIXES:
        if base.endswith(suf) and len(base) - len(suf) >= 3:
            base = base[: -len(suf)]
            break
    return base


def count_syllables(word: str) -> int:
    """Approximate syllable count using vowel groups; tailored for Polish."""
    return max(1, len(re.findall(r"[aeiouyąęó]+", word.lower())))


def fog_index(text: str) -> Tuple[float, float, int, int]:
    """Gunning FOG index and supporting stats (avg sentence len, words, sentences)."""
    sentences = [s for s in re.split(r"[\\.!\?…]+", text) if s.strip()]
    words = [w for w in TOKEN_RE.findall(text)]
    if not sentences or not words:
        return 0.0, 0.0, 0, 0
    avg_sent_len = len(words) / len(sentences)
    complex_words = sum(count_syllables(w) >= 3 for w in words)
    fog = 0.4 * (avg_sent_len + 100 * complex_words / len(words))
    return fog, avg_sent_len, len(words), len(sentences)


def load_sentiment_lexicon(path: Path) -> Dict[str, float]:
    """Load Slownik Wydzwieku (valence in last column); return stem→score."""
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["word", "pos", "emo1", "emo2", "emo3", "val"],
    )
    df["stem"] = df["word"].astype(str).str.lower()
    # some stems repeat; average scores
    lex = df.groupby("stem")["val"].mean().to_dict()
    return lex


def sentiment_score(tokens: Iterable[str], lexicon: Dict[str, float]) -> float:
    scores = [lexicon[tok] for tok in tokens if tok in lexicon]
    return float(np.mean(scores)) if scores else 0.0


def build_mp_dir_map(html_root: Path) -> Dict[int, Path]:
    """Map mp_id → path inside html_by_mp, using numeric prefix in dir names."""
    mapping = {}
    for p in html_root.iterdir():
        if not p.is_dir():
            continue
        match = re.match(r"^(\d{3})-", p.name)
        if match:
            mapping[int(match.group(1))] = p
    return mapping


def load_interpellations(mp_map: Dict[int, str]) -> List[Document]:
    root = RAW_ROOT / "interpellations" / "by_num"
    records: List[Document] = []
    for folder in tqdm(list(root.iterdir()), desc="Interpelacje", unit="doc"):
        if not folder.is_dir():
            continue
        dpath, bpath = folder / "details.json", folder / "body.html"
        if not (dpath.exists() and bpath.exists()):
            continue
        details = json.loads(dpath.read_text(encoding="utf-8"))
        authors = details.get("from") or []
        if isinstance(authors, int):
            authors = [authors]
        if not authors:
            continue
        text = html_to_text(bpath)
        date = pd.to_datetime(details.get("sentDate", None))
        doc_id = f"interp_{details.get('num', folder.name)}"
        weight = 1 / len(authors)
        for mp_id in authors:
            try:
                mp_id_int = int(mp_id)
            except (TypeError, ValueError):
                continue
            club = mp_map.get(mp_id_int)
            if not club:
                continue
            records.append(
                Document(doc_id, "interpelacja", mp_id_int, club, date, weight, text)
            )
    return records


def load_transcripts(mp_map: Dict[int, str]) -> List[Document]:
    index_path = TABLES_ROOT / "fact_transcript_statement_index.csv"
    html_root = RAW_ROOT / "transcripts" / "html_by_mp"
    dir_map = build_mp_dir_map(html_root)

    df = pd.read_csv(index_path)
    df = df[(df["unspoken"] == False) & (df["memberID"] > 0)]
    records: List[Document] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Stenogramy", unit="stmt"):
        mp_id = int(row.memberID)
        club = mp_map.get(mp_id)
        base = dir_map.get(mp_id)
        if not base or not club:
            continue
        path = (
            base
            / f"p{int(row.proceedingNum)}"
            / f"d{row.date}"
            / f"s{int(row.statementNum)}.html"
        )
        if not path.exists():
            continue
        text = html_to_text(path)
        date = pd.to_datetime(row.date)
        doc_id = f"trans_{row.proceedingNum}_{row.statementNum}"
        records.append(Document(doc_id, "stenogram", mp_id, club, date, 1.0, text))
    return records


def top_terms_by_group(
    matrix, feature_names: List[str], groups: np.ndarray, top_n: int = 15
) -> pd.DataFrame:
    """Mean tf-idf per group; return top terms."""
    out = []
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        if len(idx) == 0:
            continue
        mean_vec = np.asarray(matrix[idx].mean(axis=0)).ravel()
        top_idx = mean_vec.argsort()[::-1][:top_n]
        for rank, tid in enumerate(top_idx, start=1):
            out.append(
                {
                    "group": group,
                    "term": feature_names[tid],
                    "score": float(mean_vec[tid]),
                    "rank": rank,
                }
            )
    return pd.DataFrame(out)


def lda_topics(
    count_vec,
    lda_model: LatentDirichletAllocation,
    feature_names: List[str],
    source: str,
) -> pd.DataFrame:
    rows = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_ids = topic.argsort()[::-1][:12]
        for rank, tid in enumerate(top_ids, start=1):
            rows.append(
                {
                    "source": source,
                    "topic": topic_idx + 1,
                    "term": feature_names[tid],
                    "weight": float(topic[tid]),
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)


def lda_topic_mix(doc_topic: np.ndarray, clubs: List[str], source: str) -> pd.DataFrame:
    df = pd.DataFrame(doc_topic)
    df["club"] = clubs
    mix = (
        df.groupby("club")
        .mean()
        .reset_index()
        .melt(id_vars="club", var_name="topic", value_name="proportion")
    )
    mix["topic"] = mix["topic"].astype(int) + 1
    mix["source"] = source
    return mix


def topic_labels(df: pd.DataFrame, top_n: int = 4) -> pd.DataFrame:
    """Create short human-readable labels per topic using top_n terms."""
    labels = (
        df.sort_values(["source", "topic", "rank"])
        .groupby(["source", "topic"])
        .head(top_n)
        .groupby(["source", "topic"])["term"]
        .apply(lambda terms: ", ".join(terms))
        .reset_index()
        .rename(columns={"term": "label"})
    )
    return labels


def log_odds_ratio(
    group_a: Counter, group_b: Counter, alpha: float = 1.0
) -> pd.DataFrame:
    vocab = set(group_a) | set(group_b)
    total_a = sum(group_a.values()) + alpha * len(vocab)
    total_b = sum(group_b.values()) + alpha * len(vocab)
    rows = []
    for term in vocab:
        pa = (group_a[term] + alpha) / total_a
        pb = (group_b[term] + alpha) / total_b
        rows.append((term, math.log(pa / pb)))
    return pd.DataFrame(rows, columns=["term", "log_odds"])


def prune_counter(counter: Counter, min_count: int) -> Counter:
    return Counter({k: v for k, v in counter.items() if v >= min_count})


def count_mentions(text: str, needle: str) -> int:
    """Count case/diacritic-insensitive word mentions in raw text."""
    low = normalize_ascii(text.lower())
    pattern = rf"\\b{needle}\\w*"
    return len(re.findall(pattern, low))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mp_df = pd.read_csv(TABLES_ROOT / "dim_mp.csv")
    stmt_df = pd.read_csv(TABLES_ROOT / "fact_transcript_statement_index.csv")
    mp_df["club"] = mp_df["club"].replace(
        {
            "Polska2050-TD": "Polska2050",
            "Polska2050_TD": "Polska2050",
            "PSL-TD": "PSL",
            "PSL_TD": "PSL",
        }
    )
    global NAME_STOPLIST
    NAME_STOPLIST = build_name_stoplist(mp_df, stmt_df)
    mp_map = dict(zip(mp_df.mp_id, mp_df.club))

    lexicon = load_sentiment_lexicon(LEXICON_PATH)

    # Load corpora
    interps = load_interpellations(mp_map)
    trans = load_transcripts(mp_map)
    docs = interps + trans
    corpus = pd.DataFrame([d.__dict__ for d in docs])

    # Preprocessing
    corpus["tokens"] = [
        tokenize(t) for t in tqdm(corpus["text"], desc="Tokenizacja", unit="doc")
    ]
    corpus["sentiment"] = [
        sentiment_score(toks, lexicon)
        for toks in tqdm(corpus["tokens"], desc="Sentyment", unit="doc")
    ]
    corpus["lemmas"] = [[fold_token(t) for t in toks] for toks in corpus["tokens"]]
    fog_cols = [fog_index(t) for t in tqdm(corpus["text"], desc="FOG", unit="doc")]
    corpus[["fog_index", "avg_sentence_len", "word_count", "sentence_count"]] = (
        pd.DataFrame(fog_cols, index=corpus.index)
    )
    corpus["token_str"] = corpus["lemmas"].apply(lambda toks: " ".join(toks))
    corpus["tusk_cnt"] = [count_mentions(t, "tusk") for t in corpus["text"]]
    corpus["kaczynski_cnt"] = [count_mentions(t, "kaczynski") for t in corpus["text"]]

    # Save doc-level metrics
    corpus[
        [
            "doc_id",
            "source",
            "club",
            "date",
            "weight",
            "sentiment",
            "fog_index",
            "avg_sentence_len",
            "word_count",
            "sentence_count",
        ]
    ].to_parquet(OUTPUT_DIR / "docs_metrics.parquet", index=False)

    # TF-IDF – liczone na poziomie klubów (po scaleniu wszystkich tekstów w jeden dokument)
    tfidf_rows = []
    for source in corpus["source"].unique():
        sub = corpus[corpus["source"] == source]
        club_docs = sub.groupby("club")["token_str"].apply(lambda s: " ".join(s)).reset_index()
        tfidf_vectorizer = TfidfVectorizer(
            min_df=5,
            max_df=0.6,
            max_features=8000,
            token_pattern=r"(?u)\b\w+\b",
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(club_docs["token_str"])
        feats = tfidf_vectorizer.get_feature_names_out().tolist()

        for idx, club in enumerate(club_docs["club"]):
            row = tfidf_matrix[idx].toarray().ravel()
            top_ids = row.argsort()[::-1][:20]
            for rank, tid in enumerate(top_ids, start=1):
                tfidf_rows.append(
                    {
                        "group": club,
                        "term": feats[tid],
                        "score": float(row[tid]),
                        "rank": rank,
                        "source": source,
                    }
                )

    pd.DataFrame(tfidf_rows).to_csv(OUTPUT_DIR / "tfidf_top_terms.csv", index=False)

    # Bi-/tri-gramy TF-IDF (również agregacja na poziomie klubów)
    ngram_rows = []
    for source in corpus["source"].unique():
        sub = corpus[corpus["source"] == source]
        club_docs = sub.groupby("club")["token_str"].apply(lambda s: " ".join(s)).reset_index()
        nvec = TfidfVectorizer(
            min_df=4,
            max_df=0.45,
            max_features=6000,
            ngram_range=(2, 3),
            token_pattern=r"(?u)\b\w+\b",
        )
        nmat = nvec.fit_transform(club_docs["token_str"])
        feats = nvec.get_feature_names_out().tolist()
        for idx, club in enumerate(club_docs["club"]):
            row = nmat[idx].toarray().ravel()
            top_ids = row.argsort()[::-1][:6]
            for rank, tid in enumerate(top_ids, start=1):
                ngram_rows.append(
                    {
                        "group": club,
                        "term": feats[tid],
                        "ngram_len": len(feats[tid].split()),
                        "score": float(row[tid]),
                        "rank": rank,
                        "source": source,
                    }
                )
    pd.DataFrame(ngram_rows).to_csv(OUTPUT_DIR / "tfidf_top_ngrams.csv", index=False)

    # Raw word counts (for wordclouds)
    wc_rows = []
    for (source, club), df_sub in corpus.groupby(["source", "club"]):
        counts = Counter()
        for toks in df_sub["tokens"]:
            counts.update(toks)
        for term, cnt in counts.most_common(200):
            wc_rows.append({"source": source, "club": club, "term": term, "count": cnt})
    pd.DataFrame(wc_rows).to_csv(OUTPUT_DIR / "word_counts.csv", index=False)

    # LDA per source (unigram + bigram for bardziej zrozumiałe tematy)
    lda_topics_all = []
    lda_mix_all = []
    for source in corpus["source"].unique():
        sub = corpus[corpus["source"] == source]
        cv = CountVectorizer(
            max_features=4000,
            min_df=60,
            max_df=0.5,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
        )
        dtm = cv.fit_transform(sub["token_str"])
        lda = LatentDirichletAllocation(
            n_components=6,
            learning_method="batch",
            learning_decay=0.7,
            max_iter=16,
            random_state=42,
            n_jobs=-1,
        ).fit(dtm)

        feature_names = cv.get_feature_names_out().tolist()
        lda_topics_all.append(lda_topics(dtm, lda, feature_names, source))

        doc_topic = lda.transform(dtm)
        lda_mix_all.append(lda_topic_mix(doc_topic, sub["club"].tolist(), source))

    lda_topics_df = pd.concat(lda_topics_all, ignore_index=True)
    lda_mix_df = pd.concat(lda_mix_all, ignore_index=True)
    lda_topics_df.to_csv(OUTPUT_DIR / "lda_topics.csv", index=False)
    lda_mix_df.to_csv(OUTPUT_DIR / "lda_topic_mix.csv", index=False)

    topic_labels(lda_topics_df, top_n=4).to_csv(
        OUTPUT_DIR / "lda_topic_labels.csv", index=False
    )

    # Sentiment & complexity aggregations
    agg = (
        corpus.groupby(["club", "source"])
        .apply(
            lambda d: pd.Series(
                {
                    "n_docs": len(d),
                    "sentiment_mean": np.average(d.sentiment, weights=d.weight),
                    "sentiment_median": d.sentiment.median(),
                    "sentiment_std": d.sentiment.std(ddof=0),
                    "fog_mean": np.average(d.fog_index, weights=d.weight),
                    "fog_median": d.fog_index.median(),
                    "fog_p25": d.fog_index.quantile(0.25),
                    "fog_p75": d.fog_index.quantile(0.75),
                    "fog_std": d.fog_index.std(ddof=0),
                    "words_mean": np.average(d.word_count, weights=d.weight),
                    "words_median": d.word_count.median(),
                }
            )
        )
        .reset_index()
    )
    agg.to_csv(OUTPUT_DIR / "style_metrics.csv", index=False)

    # Dumbbell (sentiment difference speech vs interpelacje)
    dumbbell = agg.pivot(
        index="club", columns="source", values="sentiment_mean"
    ).reset_index()
    dumbbell.columns.name = None
    dumbbell["delta"] = dumbbell.get("stenogram", np.nan) - dumbbell.get(
        "interpelacja", np.nan
    )
    dumbbell.to_csv(OUTPUT_DIR / "sentiment_dumbbell.csv", index=False)

    # Mentions of key political figures (per 10k słów)
    name_rows = []
    for (club, source), df in corpus.groupby(["club", "source"]):
        combined = " ".join(df["text"].tolist()).lower()
        low = normalize_ascii(combined)
        tusk = len(re.findall(r"\btusk\w*", low))
        kacz = len(re.findall(r"\bkaczynski\w*", low))
        words = df["word_count"].sum()
        name_rows.append(
            {"club": club, "source": source, "tusk": tusk, "kaczynski": kacz, "words": words}
        )
    name_rows = pd.DataFrame(name_rows)
    name_rows["tusk_per10k"] = name_rows["tusk"] / name_rows["words"] * 10000
    name_rows["kaczynski_per10k"] = name_rows["kaczynski"] / name_rows["words"] * 10000
    name_rows.to_csv(OUTPUT_DIR / "name_counts.csv", index=False)

    # Government vs opposition comparison (log-odds)
    gov = {"KO", "PSL", "Polska2050", "Lewica", "Razem"}
    opp = {"PiS", "Konfederacja", "Republikanie"}
    comp_rows = []
    for source in corpus["source"].unique():
        gov_counts, opp_counts = Counter(), Counter()
        for _, row in corpus[corpus["source"] == source].iterrows():
            if row.club in gov:
                gov_counts.update(row.tokens)
            elif row.club in opp:
                opp_counts.update(row.tokens)
        gov_counts = prune_counter(gov_counts, 60)
        opp_counts = prune_counter(opp_counts, 60)
        if gov_counts and opp_counts:
            gov_odds = log_odds_ratio(gov_counts, opp_counts)
            opp_odds = log_odds_ratio(opp_counts, gov_counts)
            gov_odds["group"] = "rząd"
            opp_odds["group"] = "opozycja"
            gov_odds["source"] = source
            opp_odds["source"] = source
            comp_rows.append(gov_odds)
            comp_rows.append(opp_odds)
    if comp_rows:
        pd.concat(comp_rows, ignore_index=True).to_csv(
            OUTPUT_DIR / "comparison_log_odds.csv", index=False
        )


if __name__ == "__main__":
    main()
