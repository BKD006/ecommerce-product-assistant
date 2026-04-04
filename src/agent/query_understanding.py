from typing import Dict, Any, Optional
import re
import json
from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import ProductAssistantException
from src.agent.prompts import intent_classifier

class QueryUnderstanding:

    def __init__(self, llm, known_categories=None, known_brands=None):
        self.llm = llm
        self.known_categories = known_categories or []
        self.known_brands = known_brands or []

    # =====================================================
    # MAIN ENTRY
    # =====================================================

    def parse(self, query: str) -> Dict[str, Any]:

        try:
            if not query or not query.strip():
                raise ProductAssistantException("Empty query")

            q = query.lower()

            # LLM-based parsing
            parsed = self._parse_with_llm(query)

            # -----------------------------------
            # OPTIONAL FALLBACK (SAFEGUARD)
            # -----------------------------------
            if parsed["price_min"] is None and parsed["price_max"] is None:
                fallback_min, fallback_max = self._extract_price(q)
                parsed["price_min"] = fallback_min
                parsed["price_max"] = fallback_max

            # -----------------------------------
            # DISABLE FILTERS FOR AGGREGATION
            # -----------------------------------
            if parsed["intent"] == "aggregation":
                parsed["category"] = None
                parsed["brand"] = None
                parsed["price_min"] = None
                parsed["price_max"] = None

            # -----------------------------------
            # CLEAN QUERY
            # -----------------------------------
            clean_query = self._clean_query(q)

            result = {
                **parsed,
                "raw_query": query,
                "clean_query": clean_query,
            }

            log.info("Query parsed", parsed=result)

            return result

        except Exception as e:
            log.error("Query parsing failed", exc_info=True)
            raise ProductAssistantException("Query understanding failed", e)

    # =====================================================
    # LLM PARSER (INTENT + PRICE + STRUCTURE)
    # =====================================================

    def _parse_with_llm(self, query: str) -> Dict[str, Any]:
        prompt = intent_classifier.format(user_query=query)

        try:
            response = self.llm.invoke(prompt, temperature=0).content
            parsed = json.loads(response)

            return {
                "intent": parsed.get("intent", "search"),
                "category": parsed.get("category"),
                "brand": parsed.get("brand"),
                "price_min": parsed.get("price_min"),
                "price_max": parsed.get("price_max"),
            }

        except Exception:
            return {
                "intent": "search",
                "category": None,
                "brand": None,
                "price_min": None,
                "price_max": None,
            }

    # =====================================================
    # OPTIONAL: CATEGORY DETECTION (KEEP OR REMOVE)
    # =====================================================

    def _detect_category(self, q: str) -> Optional[str]:
        for cat in self.known_categories:
            if cat.lower() in q:
                return cat
        return None

    # =====================================================
    # OPTIONAL: BRAND DETECTION (KEEP OR REMOVE)
    # =====================================================

    def _detect_brand(self, q: str) -> Optional[str]:
        for brand in self.known_brands:
            if brand.lower() in q:
                return brand
        return None

    # =====================================================
    # FALLBACK PRICE (OPTIONAL BUT SAFE)
    # =====================================================

    def _extract_price(self, q: str):

        q = re.sub(r"(\d+)\s*k", lambda m: str(int(m.group(1)) * 1000), q)

        numbers = re.findall(r"\d+", q)

        if not numbers:
            return None, None

        if "under" in q or "below" in q:
            return None, int(numbers[0])

        if "above" in q:
            return int(numbers[0]), None

        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])

        return None, None

    # =====================================================
    # CLEAN QUERY
    # =====================================================

    def _clean_query(self, q: str) -> str:

        q = re.sub(r"\b(under|below|above|less than)\b", "", q)
        q = re.sub(r"\d+", "", q)

        return q.strip()

