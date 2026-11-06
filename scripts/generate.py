"""Daily news briefing powered by Google Gemini."""

import os
import json
import hashlib
import logging
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

import requests
import feedparser
import trafilatura
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TODAY = dt.datetime.now(ZoneInfo("Asia/Seoul")).date()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_NEWS = min(int(os.getenv("MAX_PER_CATEGORY", "5")), 5)

HEADERS = {
    "User-Agent": "Mozilla/5.0 DailyNewsReader/1.0"
}

FEEDS = {
    "경영·경제·금융": [
        "https://www.theguardian.com/business/rss",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
    ],
    "국내 정치·사회·정책": [
        "https://www.hani.co.kr/rss/",
        "https://www.koreatimes.co.kr/www/rss/nation.xml",
    ],
    "국제 정세·외교·안보": [
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.theguardian.com/world/rss",
    ],
    "과학·기술·환경": [
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
        "https://www.theguardian.com/environment/rss",
        "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    ],
}

logging.basicConfig(level=logging.INFO)


def ask_gemini(client, instruction, data, limit=2500):
    response = client.models.generate_content(
        model=MODEL,
        contents=json.dumps(data, ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            response_mime_type="application/json",
            max_output_tokens=limit,
            temperature=0.2,
            http_options=types.HttpOptions(timeout=60000),
        ),
    )

    if not response.text:
        raise ValueError("Empty Gemini response")

    return json.loads(response.text)


def collect_news(category):
    articles = []
    seen = set()

    for feed_url in FEEDS[category]:
        try:
            response = requests.get(
                feed_url,
                headers=HEADERS,
                timeout=12,
            )
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            for entry in feed.entries[:20]:
                url = entry.get("link", "")
                published = (
                    entry.get("published_parsed")
                    or entry.get("updated_parsed")
                )

                if not url.startswith("http"):
                    continue

                if url in seen or not published:
                    continue

                date = dt.date(*published[:3])
                age = (TODAY - date).days

                if age < -1 or age > 3:
                    continue

                seen.add(url)

                summary = BeautifulSoup(
                    entry.get("summary", ""),
                    "html.parser",
                ).get_text(" ", strip=True)

                articles.append({
                    "url": url,
                    "title": entry.get("title", ""),
                    "published": date.isoformat(),
                    "source": urlparse(url).hostname or "",
                    "snippet": summary[:300],
                })

        except Exception as error:
            logging.warning(
                "RSS error: %s",
                error,
            )

    return articles[:35]


def fetch_article(url):
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            "text/html",
        )

        if "text/html" not in content_type:
            return ""

        text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )

        return text or ""

    except Exception as error:
        logging.warning(
            "Article fetch error: %s",
            error,
        )
        return ""


def load_index():
    try:
        return json.loads(
            (DATA / "index.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError):
        return {"dates": []}


def main():
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing"
        )

    client = genai.Client(api_key=API_KEY)
    DATA.mkdir(exist_ok=True)

    index = load_index()
    previous_titles = []

    for date in index.get("dates", [])[:5]:
        try:
            file = DATA / f"{date}.json"
            old = json.loads(
                file.read_text(encoding="utf-8")
            )
            previous_titles.extend(
                article["title"]
                for article in old.get("articles", [])
            )
        except (OSError, ValueError, KeyError):
            pass

    results = []
    used_urls = set()

    for category in FEEDS:
        logging.info("Collecting: %s", category)

        candidates = collect_news(category)

        if not candidates:
            continue

        try:
            selected = ask_gemini(
                client,
                """
당신은 한국어 뉴스 편집자입니다.

제공된 뉴스 후보만 평가하세요.
새로운 기사를 만들어내지 마세요.

중요하고 최근에 발생한 사건을
우선적으로 선정하세요.

연예, 스포츠, 가십, 광고,
중복 기사는 제외하세요.

국내 분야는 대한민국 관련
뉴스만 선정하세요.

이전 기사와 같은 사건이라면
의미 있는 새로운 변화가
있을 때만 선정하세요.

최대 7개를 선택하세요.

반드시 JSON 형식으로 반환하세요.
{"indices":[0,1,2]}
""",
                {
                    "category": category,
                    "previous_titles": previous_titles[-60:],
                    "candidates": [
                        {
                            "i": i,
                            "title": item["title"],
                            "snippet": item["snippet"],
                        }
                        for i, item in enumerate(candidates)
                    ],
                },
                limit=500,
            ).get("indices", [])

        except Exception as error:
            logging.warning(
                "Selection error: %s",
                error,
            )
            continue

        count = 0

        for index_number in selected[:7]:
            if count >= MAX_NEWS:
                break

            if not isinstance(index_number, int):
                continue

            if not 0 <= index_number < len(candidates):
                continue

            item = candidates[index_number]

            if item["url"] in used_urls:
                continue

            article_text = fetch_article(item["url"])

            if len(article_text) < 650:
                continue

            try:
                summary = ask_gemini(
                    client,
                    """
당신은 정확성을 중시하는
한국어 뉴스 해설자입니다.

제공된 기사 원문에 근거하여
내용을 자세하게 정리하세요.

원문에 없는 사건, 수치,
인물의 발언을 만들어내지 마세요.

summary:
기사의 주요 내용을 5~9문장으로
자세히 요약하세요.
중요한 수치, 사건, 발언,
정책 내용을 포함하세요.

background:
사건의 배경과 관련 개념을
3~5문장으로 설명하세요.

impact:
경제적, 사회적 영향과
향후 전망을 4~6문장으로
분석하세요.
추론과 확인된 사실을 구분하세요.

terms:
어려운 용어를 쉽게 설명하세요.

반드시 다음 JSON 형식으로
반환하세요.

{
"title":"",
"summary":"",
"background":"",
"impact":"",
"terms":""
}
""",
                    {
                        "category": category,
                        "title": item["title"],
                        "published": item["published"],
                        "article": article_text[:14000],
                    },
                    limit=3000,
                )

                required = (
                    "title",
                    "summary",
                    "background",
                    "impact",
                )

                if not all(
                    len(str(summary.get(key, "")).strip()) >= 25
                    for key in required
                ):
                    continue

                article_id = hashlib.sha256(
                    item["url"].encode()
                ).hexdigest()[:16]

                results.append({
                    "id": article_id,
                    "category": category,
                    "url": item["url"],
                    "source": item["source"],
                    "published": item["published"],
                    "title": str(summary.get("title", "")),
                    "summary": str(summary.get("summary", "")),
                    "background": str(summary.get("background", "")),
                    "impact": str(summary.get("impact", "")),
                    "terms": str(summary.get("terms", "")),
                })

                used_urls.add(item["url"])
                count += 1

                logging.info(
                    "Added: %s",
                    item["title"],
                )

            except Exception as error:
                logging.warning(
                    "Summary error: %s",
                    error,
                )

    if not results:
        logging.warning(
            "No verified articles generated"
        )
        return

    date = TODAY.isoformat()

    document = {
        "date": date,
        "generated_at": dt.datetime.now(
            ZoneInfo("Asia/Seoul")
        ).strftime("%Y-%m-%d %H:%M KST"),
        "articles": results,
    }

    output_file = DATA / f"{date}.json"

    output_file.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    index["dates"] = sorted(
        set(index.get("dates", []) + [date]),
        reverse=True,
    )[:90]

    (DATA / "index.json").write_text(
        json.dumps(
            index,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logging.info(
        "Published %d articles",
        len(results),
    )


if __name__ == "__main__":
    main()
