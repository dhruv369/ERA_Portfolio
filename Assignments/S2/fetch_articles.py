"""Download Wikipedia's India articles for the Session 2 tokenizer experiment."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


LANGUAGES = {
    "en": ("English", "enwiki"),
    "hi": ("Hindi", "hiwiki"),
    "te": ("Telugu", "tewiki"),
    "ta": ("Tamil", "tawiki"),
}
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API = "https://{language}.wikipedia.org/w/api.php"
USER_AGENT = "ERA-Portfolio-Session2/1.0 (multilingual BPE learning project)"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "articles.json"


def fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_article_titles() -> dict[str, str]:
    params = urlencode(
        {
            "action": "wbgetentities",
            "ids": "Q668",
            "props": "sitelinks",
            "sitefilter": "|".join(site for _, site in LANGUAGES.values()),
            "format": "json",
        }
    )
    data = fetch_json(f"{WIKIDATA_API}?{params}")
    sitelinks = data.get("entities", {}).get("Q668", {}).get("sitelinks", {})

    titles = {
        code: sitelinks[site]["title"]
        for code, (_, site) in LANGUAGES.items()
        if site in sitelinks
    }
    missing = set(LANGUAGES) - set(titles)
    if missing:
        raise RuntimeError(
            "Wikidata did not return article links for: "
            + ", ".join(sorted(missing))
        )
    return titles


def fetch_article(language: str, title: str) -> dict[str, Any]:
    params = urlencode(
        {
            "action": "query",
            "prop": "extracts|info",
            "explaintext": "1",
            "titles": title,
            "inprop": "url",
            "format": "json",
            "formatversion": "2",
        }
    )
    url = f"{WIKIPEDIA_API.format(language=language)}?{params}"
    data = fetch_json(url)
    pages = data.get("query", {}).get("pages", [])

    if not pages or "missing" in pages[0]:
        raise RuntimeError(f"Wikipedia article not found: {language}:{title}")

    page = pages[0]
    text = page.get("extract")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"Wikipedia returned no article text: {language}:{title}")

    return {
        "language": language,
        "language_name": LANGUAGES[language][0],
        "title": page["title"],
        "page_url": page["fullurl"],
        "revision_id": page["lastrevid"],
        "text": text,
    }


def main() -> None:
    titles = fetch_article_titles()
    articles = [
        fetch_article(language, titles[language])
        for language in LANGUAGES
    ]

    output = {
        "subject": "India",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "license": "Wikipedia text; see https://creativecommons.org/licenses/by-sa/4.0/",
        "articles": articles,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {len(articles)} articles to {OUTPUT_PATH}")
    for article in articles:
        print(
            f"{article['language_name']}: {article['title']} "
            f"(revision {article['revision_id']})"
        )


if __name__ == "__main__":
    main()
