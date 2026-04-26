"""Price-lookup backends for Phase 4.

Only the Groq vision backend is implemented; eBay scrape + Browse API are
outlined as stubs for later. All three return dicts so the caller can swap
implementations without code changes.
"""

import base64
import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TIMEOUT_SEC = 15.0
GROQ_RETRY_BACKOFF_SEC = 1.0

_client = None


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=GROQ_TIMEOUT_SEC)
    return _client

PROMPT = """You are a resale pricing expert for eBay / Facebook Marketplace / thrift flipping.

Task: from the image(s), (1) identify the item as specifically as possible - brand, model, generation if visible - (2) judge its visible condition, (3) estimate a realistic used resale price range in USD.

YOLO's automated guess was: "{yolo_hint}". YOLO is FREQUENTLY WRONG for items outside its 80-class training set (watches, controllers, trading cards, jewelry, tools, etc.). Trust your own read of the image over YOLO's label.

{multi_image_note}

Respond with ONLY valid JSON matching this schema, no prose, no markdown fences:
{{
  "item": "specific name, e.g. 'Xbox Series X Wireless Controller, Carbon Black'",
  "brand": "e.g. Microsoft, or null if unknown",
  "model": "e.g. Series X, or null",
  "condition": "one of: new | used-mint | used-good | used-fair | used-heavy",
  "confidence": 0.0 to 1.0 (your confidence in the item ID),
  "low": integer USD,
  "high": integer USD,
  "notes": "one sentence citing visible condition cues or market reasoning"
}}

If you genuinely cannot identify the item, set "item": "unknown", "confidence": 0, "low": 0, "high": 0."""


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_json(raw: str) -> dict:
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"item": "unknown", "confidence": 0, "low": 0, "high": 0, "notes": "parse_failed", "raw": raw}


def estimate_value_groq(image_path: str, yolo_hint: str | None = None, full_frame_path: str | None = None) -> dict:
    """Identify item + estimate resale price.

    image_path:       required, cropped region of interest.
    yolo_hint:        optional, YOLO's top-class guess (passed to the prompt so the model knows what to override).
    full_frame_path:  optional, unmodified full webcam frame for background context.
    """
    from groq import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    client = _get_client()

    multi_image_note = (
        "Image 1 is the full frame for context. Image 2 is the cropped region of interest — focus identification on Image 2."
        if full_frame_path
        else "The image shows the item to identify."
    )
    prompt = PROMPT.format(yolo_hint=yolo_hint or "none", multi_image_note=multi_image_note)

    content = [{"type": "text", "text": prompt}]
    if full_frame_path:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(full_frame_path)}"}})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(image_path)}"}})

    retryable = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

    def _call(msg_content):
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": msg_content}],
                    max_tokens=400,
                    temperature=0,
                )
                return response.choices[0].message.content or ""
            except retryable:
                if attempt == 0:
                    time.sleep(GROQ_RETRY_BACKOFF_SEC)
                    continue
                raise

    try:
        raw = _call(content)
    except Exception:
        if full_frame_path:
            # Graceful fallback: some providers reject multi-image payloads.
            fallback_content = [
                {"type": "text", "text": PROMPT.format(yolo_hint=yolo_hint or "none", multi_image_note="The image shows the item to identify.")},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(image_path)}"}},
            ]
            raw = _call(fallback_content)
        else:
            raise

    return _parse_json(raw)


def estimate_value_ebay_scrape(query: str) -> dict:
    """Median price from eBay sold-listings search (NOT IMPLEMENTED).

    Outline:
        url = f"https://www.ebay.com/sch/i.html?_nkw={quote_plus(query)}&LH_Sold=1&LH_Complete=1"
        html = requests.get(url, headers={"User-Agent": "..."}).text
        soup = BeautifulSoup(html, "html.parser")
        prices = []
        for span in soup.select(".s-item__price"):
            text = span.get_text().replace("$", "").replace(",", "").split()[0]
            try:
                prices.append(float(text))
            except ValueError:
                continue
        prices.sort()
        return {"query": query, "median": prices[len(prices)//2], "count": len(prices), "samples": prices[:10]}

    Caveats: fragile (HTML changes), technically against ToS to redistribute,
    requires pip install beautifulsoup4.
    """
    raise NotImplementedError("eBay scrape not implemented - see docstring for outline")


def estimate_value_ebay_api(query: str) -> dict:
    """Median price from eBay Browse API (NOT IMPLEMENTED).

    Outline:
        1. Create eBay developer account + app at developer.ebay.com
        2. Get client_id / client_secret, stash in .env as EBAY_CLIENT_ID / EBAY_CLIENT_SECRET
        3. OAuth client-credentials flow:
             POST https://api.ebay.com/identity/v1/oauth2/token
             body: grant_type=client_credentials&scope=https://api.ebay.com/oauth/api_scope
             auth: HTTP Basic (client_id:client_secret)
        4. GET https://api.ebay.com/buy/browse/v1/item_summary/search?q={query}&filter=conditions:{USED}
           with Authorization: Bearer {token}
        5. Parse item_summaries[*].price.value, return median + samples.

    Caveats: the Browse API returns active listings, not sold. For true sold-price data
    you need the (gated) Marketplace Insights API.
    """
    raise NotImplementedError("eBay Browse API not implemented - see docstring for outline")
