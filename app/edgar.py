"""Download the latest 10-K for a ticker from SEC EDGAR and convert it to plain text."""
from __future__ import annotations

import re

import requests
from bs4 import BeautifulSoup

from . import config

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


def _get(url: str) -> requests.Response:
    if not config.SEC_USER_AGENT:
        raise RuntimeError(
            "Set SEC_USER_AGENT in your .env file, e.g. 'Your Name your@email.com'. "
            "The SEC requires it for automated requests."
        )
    response = requests.get(
        url,
        headers={"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"},
        timeout=60,
    )
    response.raise_for_status()
    return response


def ticker_to_cik(ticker: str) -> tuple[int, str]:
    """Return (CIK, company name) for a stock ticker."""
    for row in _get(TICKERS_URL).json().values():
        if row["ticker"].upper() == ticker.upper():
            return int(row["cik_str"]), row["title"]
    raise ValueError(f"Ticker '{ticker}' was not found in SEC EDGAR.")


def latest_10k(cik: int) -> dict:
    """Return metadata and the document URL for the company's most recent 10-K."""
    recent = _get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            accession = recent["accessionNumber"][i]
            document = recent["primaryDocument"][i]
            return {
                "form": form,
                "filing_date": recent["filingDate"][i],
                "accession": accession,
                "url": ARCHIVE_URL.format(cik=cik, accession=accession.replace("-", ""), document=document),
            }
    raise ValueError(f"No recent 10-K found for CIK {cik}.")


def html_to_text(html: str) -> str:
    """Strip markup, hidden XBRL blocks and blank lines from a filing's HTML."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        tag.decompose()
    text = soup.get_text("\n").replace("\xa0", " ")
    lines = (re.sub(r"\s+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def fetch_filing_text(url: str) -> str:
    return html_to_text(_get(url).text)
