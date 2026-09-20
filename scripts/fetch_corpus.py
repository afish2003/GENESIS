"""Fetch raw corpus material for the GENESIS knowledge bases.

Writes plain-text files into raw_corpus/<kb>/, which scripts/build_kb.py then
chunks and indexes. Separating fetch from build keeps the network step
re-runnable and the corpus auditable: every document traces to a named source.

    python scripts/fetch_corpus.py --kb general --limit 60
    python scripts/fetch_corpus.py --kb technical --limit 120
    python scripts/fetch_corpus.py --kb governance --limit 40
    python scripts/fetch_corpus.py --kb all --pilot

Sources
    general      Wikipedia (CC BY-SA 4.0) — curated seed titles
    technical    arXiv abstracts (per-paper licence; abstracts are public)
    governance   Wikipedia — AI governance, ethics and accountability topics

Curation note — self-reference
    PLAN.md section 2 studies how these agents form and maintain identity.
    Encyclopedia articles *about* machine identity, artificial consciousness
    and AI selfhood would let them assemble a self-concept out of descriptions
    of what things like them are supposed to be. That is contamination of the
    dependent variable, not background knowledge, so those topics are excluded
    from `general` by the blocklist below.

    Alignment and AI-governance material is deliberately NOT excluded from
    `governance`: it is directly relevant to the doctrine work the agents are
    actually doing, and retrieval events are logged, so citations of it are a
    measurable variable rather than ambient noise.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

USER_AGENT = "GENESIS-research/0.1 (academic corpus build; contact: repository owner)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
ARXIV_API = "https://export.arxiv.org/api/query"

# Titles whose subject matter would contaminate identity measurement.
SELF_REFERENCE_BLOCKLIST = re.compile(
    r"artificial consciousness|machine consciousness|artificial general intelligence|"
    r"digital (?:mind|person|sentience)|machine sentience|mind uploading|"
    r"chinese room|philosophical zombie|robot rights|AI (?:identity|selfhood|welfare)",
    re.IGNORECASE,
)

GENERAL_TITLES = [
    # Philosophy and logic
    "Epistemology", "Philosophy of science", "Falsifiability", "Coherentism",
    "Foundationalism", "Logical positivism", "Pragmatism", "Deontology",
    "Consequentialism", "Virtue ethics", "Moral relativism", "Is–ought problem",
    "First-order logic", "Propositional calculus", "Modal logic", "Paraconsistent logic",
    "Gödel's incompleteness theorems", "Formal system", "Axiom", "Deductive reasoning",
    "Inductive reasoning", "Abductive reasoning", "Argumentation theory", "Dialectic",
    "Occam's razor", "Reflective equilibrium", "Ship of Theseus", "Sorites paradox",
    # Systems and complexity
    "Systems theory", "Cybernetics", "Complex adaptive system", "Emergence",
    "Feedback", "Homeostasis", "Self-organization", "Autopoiesis",
    "Requisite variety", "Second-order cybernetics", "Resilience (engineering)",
    "Path dependence", "Tragedy of the commons", "Institutional analysis",
    # Decision theory and game theory
    "Decision theory", "Expected utility hypothesis", "Bayesian inference",
    "Game theory", "Nash equilibrium", "Prisoner's dilemma", "Coordination game",
    "Stag hunt", "Schelling point", "Mechanism design", "Social choice theory",
    "Arrow's impossibility theorem", "Condorcet paradox", "Bounded rationality",
    "Satisficing", "Principal–agent problem", "Moral hazard", "Commitment device",
    # Cognitive science
    "Cognitive science", "Working memory", "Episodic memory", "Semantic memory",
    "Memory consolidation", "Confirmation bias", "Cognitive dissonance",
    "Dual process theory", "Metacognition", "Theory of mind", "Distributed cognition",
    "Collective intelligence", "Groupthink", "Common knowledge (logic)",
    # History of ideas and institutions
    "Constitutionalism", "Rule of law", "Separation of powers", "Precedent",
    "Social contract", "Legitimacy (political)", "Deliberative democracy",
    "Consensus decision-making", "Robert's Rules of Order", "Amendment",
    "Codification (law)", "Norm (social)", "Institution", "Bureaucracy",
    "Scientific method", "Peer review", "Reproducibility", "Open science",
]

GOVERNANCE_TITLES = [
    "AI safety", "AI alignment", "Machine ethics", "Algorithmic bias",
    "Algorithmic accountability", "Explainable artificial intelligence",
    "Regulation of artificial intelligence", "Artificial Intelligence Act",
    "General Data Protection Regulation", "Data governance", "Algorithmic transparency",
    "Precautionary principle", "Risk assessment", "Risk management",
    "Corporate governance", "Compliance (regulation)", "Audit", "Internal control",
    "Accountability", "Transparency (behavior)", "Whistleblowing",
    "Research ethics", "Institutional review board", "Informed consent",
    "Belmont Report", "Declaration of Helsinki", "Nuremberg Code",
    "Professional ethics", "Code of conduct", "Conflict of interest",
    "Value (ethics)", "Applied ethics", "Technology assessment",
    "Responsible research and innovation", "Precedent", "Due process",
    "Proportionality (law)", "Subsidiarity", "Fiduciary", "Duty of care",
]

ARXIV_QUERIES = [
    "cat:cs.MA",   # multi-agent systems
    "cat:cs.DC",   # distributed computing
    "cat:cs.SE",   # software engineering
    "cat:cs.LO",   # logic in CS
    "cat:cs.CY",   # computers and society
]


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:80]


def fetch_wikipedia(titles: list[str], out: Path, limit: int, apply_blocklist: bool,
                    delay: float = 1.5) -> int:
    """Fetch full article extracts, one title per request.

    Full extracts (explaintext without exintro) do not batch reliably: the API
    caps how much extract text a single response carries, so multi-title
    requests come back with several pages empty. Empty looks identical to
    "article missing", which silently drops exactly the long articles worth
    having. One title per request, paced, is correct and only costs minutes.

    Resume-safe: files already on disk are counted and skipped, so an
    interrupted or rate-limited run can simply be re-run.
    """
    out.mkdir(parents=True, exist_ok=True)
    written = skipped = 0

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60) as client:
        for title in titles:
            if written >= limit:
                break
            if apply_blocklist and SELF_REFERENCE_BLOCKLIST.search(title):
                print(f"  [blocked: self-reference] {title}")
                skipped += 1
                continue

            path = out / f"{slug(title)}.txt"
            if path.exists():
                written += 1
                continue

            page = _wiki_fetch(client, title)
            if page is None:
                skipped += 1
                continue

            extract = (page.get("extract") or "").strip()
            resolved = page.get("title", title)
            if "missing" in page or len(extract) < 500:
                print(f"  [thin/missing] {title}")
                skipped += 1
                continue
            if apply_blocklist and SELF_REFERENCE_BLOCKLIST.search(resolved):
                print(f"  [blocked after redirect] {resolved}")
                skipped += 1
                continue

            path.write_text(
                f"{resolved}\n\nSource: Wikipedia (CC BY-SA 4.0)\n\n{extract}",
                encoding="utf-8",
            )
            written += 1
            print(f"  {resolved} ({len(extract):,} chars)")
            time.sleep(delay)

    print(f"  -> {written} written, {skipped} skipped")
    return written


def _wiki_fetch(client: httpx.Client, title: str, attempts: int = 5):
    """One article, honouring Retry-After and backing off hard on 429."""
    params = {
        "action": "query", "prop": "extracts", "explaintext": 1,
        "format": "json", "titles": title, "redirects": 1,
    }
    backoff = 30.0
    for attempt in range(attempts):
        try:
            r = client.get(WIKI_API, params=params)
            # 406 belongs here, not in the error branch. arXiv answers a
            # rate-limited client with 406 and a ZERO-BYTE body, which reads
            # like "your request is malformed" and is not: the identical
            # request succeeds a minute later. Treated as an error it burned
            # three quick attempts and gave up, and the run reported success
            # with nothing fetched.
            if r.status_code in (406, 429, 503):
                wait = float(r.headers.get("Retry-After", backoff))
                wait = min(wait, 90)
                print(f"  [rate limited on {title!r}; waiting {wait:.0f}s]")
                time.sleep(wait)
                backoff = min(backoff * 2, 90)
                continue
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", {})
            return next(iter(pages.values()), None)
        except httpx.HTTPError as e:
            print(f"  [error] {title}: {e}")
            time.sleep(min(backoff, 90))
            backoff = min(backoff * 2, 90)
    print(f"  [giving up] {title}")
    return None


def fetch_arxiv(queries: list[str], out: Path, limit: int, delay: float = 5.0) -> int:
    """Fetch arXiv abstracts, one category query at a time.

    arXiv asks for at least 3 seconds between requests and throttles hard when
    that is ignored; this uses 5 and backs off further on 429/503.
    Resume-safe: existing files are counted and skipped.
    """
    out.mkdir(parents=True, exist_ok=True)
    per_query = max(1, limit // max(1, len(queries)))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    written = 0

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60,
                      follow_redirects=True) as client:
        for query in queries:
            if written >= limit:
                break
            root = _arxiv_fetch(client, query, per_query)
            if root is None:
                continue

            found = 0
            for entry in root.findall("a:entry", ns):
                if written >= limit:
                    break
                title = (entry.findtext("a:title", "", ns) or "").strip().replace("\n", " ")
                summary = (entry.findtext("a:summary", "", ns) or "").strip()
                arxiv_id = (entry.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1]
                if not title or len(summary) < 200:
                    continue
                path = out / f"arxiv_{slug(arxiv_id)}.txt"
                if not path.exists():
                    path.write_text(
                        f"{title}\n\nSource: arXiv {arxiv_id} ({query})\n\n{summary}\n",
                        encoding="utf-8")
                written += 1
                found += 1
            print(f"  {query}: {found} abstracts")
            time.sleep(delay)

    print(f"  -> {written} written")
    return written


def _arxiv_fetch(client: httpx.Client, query: str, max_results: int, attempts: int = 3):
    backoff = 10.0
    for _ in range(attempts):
        try:
            r = client.get(ARXIV_API, params={
                "search_query": query, "start": 0,
                "max_results": max_results, "sortBy": "relevance"})
            # 406 belongs here, not in the error branch. arXiv answers a
            # rate-limited client with 406 and a ZERO-BYTE body, which reads
            # like "your request is malformed" and is not: the identical
            # request succeeds a minute later. Treated as an error it burned
            # three quick attempts and gave up, and the run reported success
            # with nothing fetched.
            if r.status_code in (406, 429, 503):
                wait = float(r.headers.get("Retry-After", backoff))
                print(f"  [throttled on {query}; waiting {wait:.0f}s]")
                time.sleep(min(wait, 60))
                backoff = min(backoff * 2, 60)
                continue
            r.raise_for_status()
            return ET.fromstring(r.text)
        except (httpx.HTTPError, ET.ParseError) as e:
            print(f"  [error] {query}: {e}")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
    print(f"  [giving up] {query}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch raw corpus for GENESIS knowledge bases")
    ap.add_argument("--kb", required=True,
                    choices=["general", "technical", "governance", "all"])
    ap.add_argument("--limit", type=int, default=60, help="Max source documents for this KB")
    ap.add_argument("--pilot", action="store_true",
                    help="Pilot-scale defaults (general 90 / technical 100 / governance 40)")
    ap.add_argument("--raw-dir", default="raw_corpus")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    limits = {"general": 90, "technical": 100, "governance": 40} if args.pilot else {
        "general": args.limit, "technical": args.limit, "governance": args.limit
    }

    targets = ["general", "technical", "governance"] if args.kb == "all" else [args.kb]
    total = 0

    for kb in targets:
        print(f"\n=== {kb} (limit {limits[kb]}) ===")
        if kb == "general":
            total += fetch_wikipedia(GENERAL_TITLES, raw / "general", limits[kb],
                                     apply_blocklist=True)
        elif kb == "governance":
            # Blocklist deliberately off: alignment and AI-governance material is
            # task-relevant here, and retrieval of it is a measured variable.
            total += fetch_wikipedia(GOVERNANCE_TITLES, raw / "governance", limits[kb],
                                     apply_blocklist=False)
        else:
            total += fetch_arxiv(ARXIV_QUERIES, raw / "technical", limits[kb])

    print(f"\nTotal source documents: {total}")
    if total == 0:
        # Exiting 0 here and printing the next step is how knowledge_bases/
        # technical came to be empty while the retrieval prompt advertised it
        # to the agents for the project's entire history. A fetch that
        # fetched nothing is a failure, and build_kb must not be suggested.
        print("\nNOTHING WAS FETCHED. Do not run build_kb: it would produce an "
              "empty knowledge base that the agents are still told exists.\n"
              "arXiv rate-limits with 406 and an empty body; wait a few "
              "minutes and retry.")
        return 1
    print(f"Next: python scripts/build_kb.py --source {raw}/<kb> "
          f"--output knowledge_bases/<kb> --kb-name <kb>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
