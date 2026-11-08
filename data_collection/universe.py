"""The one list of stocks Stage 2 monitors.

Edit ``MONITORED_STOCKS`` to change the universe. Do not copy the symbols
into other modules.

Source: Finhacker, "Largest 500 U.S. Companies by Market Cap", data updated
November 6, 2026, 4:00 PM ET.
https://www.finhacker.cz/en/largest-us-companies-by-market-cap/

Finhacker prints Berkshire Hathaway class B as BRK-B. This list stores
BRK.B because Alpaca's market-data API uses the dot form and rejects the
hyphen form. It is the same share class, not a different company.

companiesmarketcap.com and marketcapwatch.com, checked the same week,
list these same companies but print Alphabet as GOOG and place Berkshire
ahead of Eli Lilly. This file follows Finhacker only.
"""

from __future__ import annotations

SOURCE_NAME = "Finhacker"
SOURCE_TITLE = "Largest 500 U.S. Companies by Market Cap"
SOURCE_URL = "https://www.finhacker.cz/en/largest-us-companies-by-market-cap/"
SOURCE_AS_OF = "2026-11-06T16:00:00-05:00"

# Rank is list order, largest market cap first. Each item is (symbol, company name).
MONITORED_STOCKS: tuple[tuple[str, str], ...] = (
    ("AAPL", "Apple"),
    ("NVDA", "NVIDIA"),
    ("MSFT", "Microsoft"),
    ("GOOGL", "Alphabet"),
    ("AMZN", "Amazon"),
    ("SPCX", "SpaceX"),
    ("META", "Meta Platforms"),
    ("TSLA", "Tesla"),
    ("AVGO", "Broadcom"),
    ("LLY", "Eli Lilly"),
    ("BRK.B", "Berkshire Hathaway"),
    ("WMT", "Walmart"),
    ("JPM", "JPMorgan Chase"),
    ("V", "Visa"),
    ("MU", "Micron Technology"),
    ("AMD", "Advanced Micro Devices"),
    ("XOM", "Exxon Mobil"),
    ("JNJ", "Johnson & Johnson"),
    ("MA", "Mastercard"),
    ("ABBV", "AbbVie"),
    ("COST", "Costco Wholesale"),
    ("CVX", "Chevron"),
    ("KO", "Coca-Cola"),
    ("PLTR", "Palantir Technologies"),
    ("PG", "Procter & Gamble"),
)
