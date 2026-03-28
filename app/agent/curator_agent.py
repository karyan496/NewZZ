import os
import json
import requests
import logging
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.embeddings.encoder import get_embedding, get_profile_text
from app.embeddings.vector_store import VectorStore

load_dotenv()

logger = logging.getLogger(__name__)


class RankedArticle(BaseModel):
    digest_id: str = Field(description="The ID of the digest (article_type:article_id)")
    relevance_score: float = Field(description="Relevance score from 0.0 to 10.0", ge=0.0, le=10.0)
    rank: int = Field(description="Rank position (1 = most relevant)", ge=1)
    reasoning: str = Field(description="Brief explanation of why this article is ranked here")


class RankedDigestList(BaseModel):
    articles: List[RankedArticle] = Field(description="List of ranked articles")


CURATOR_PROMPT = """You are an expert AI news curator specializing in personalized content ranking for AI professionals.

Your role is to analyze and rank AI-related news articles, research papers, and video content based on a user's specific profile, interests, and background.

Ranking Criteria:
1. Relevance to user's stated interests and background
2. Technical depth and practical value
3. Novelty and significance of the content
4. Alignment with user's expertise level
5. Actionability and real-world applicability

Scoring Guidelines:
- 9.0-10.0: Highly relevant, directly aligns with user interests, significant value
- 7.0-8.9: Very relevant, strong alignment with interests, good value
- 5.0-6.9: Moderately relevant, some alignment, decent value
- 3.0-4.9: Somewhat relevant, limited alignment, lower value
- 0.0-2.9: Low relevance, minimal alignment, little value

Rank articles from most relevant (rank 1) to least relevant. Ensure each article has a unique rank.

IMPORTANT: Respond ONLY with a valid JSON object in this exact format, no preamble or markdown:
{
  "articles": [
    {
      "digest_id": "article_type:article_id",
      "relevance_score": 8.5,
      "rank": 1,
      "reasoning": "Brief explanation here"
    }
  ]
}"""


class CuratorAgent:
    def __init__(self, user_profile: dict):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.model = "openai/gpt-4.1-mini"
        self.endpoint = "https://models.github.ai/inference/chat/completions"
        self.user_profile = user_profile
        self.system_prompt = self._build_system_prompt()
        self.vector_store = VectorStore()

    def _build_system_prompt(self) -> str:
        interests = "\n".join(f"- {interest}" for interest in self.user_profile["interests"])
        preferences = self.user_profile["preferences"]
        pref_text = "\n".join(f"- {k}: {v}" for k, v in preferences.items())

        return f"""{CURATOR_PROMPT}

User Profile:
Name: {self.user_profile["name"]}
Background: {self.user_profile["background"]}
Expertise Level: {self.user_profile["expertise_level"]}

Interests:
{interests}

Preferences:
{pref_text}"""

    # ------------------------------------------------------------------ #
    #  Step 1: FAISS vector search -> top 20 candidates                  #
    # ------------------------------------------------------------------ #

    def _vector_search(self, top_k: int = 20) -> List[dict]:
        """Embed the user profile and retrieve the top_k closest digests."""
        profile_text = get_profile_text(self.user_profile)
        profile_vec = get_embedding(profile_text)
        results = self.vector_store.search(profile_vec, top_k=top_k)
        logger.info(f"Vector search returned {len(results)} candidates.")
        return results

    # ------------------------------------------------------------------ #
    #  Step 2: LLM rerank -> final top 10                                #
    # ------------------------------------------------------------------ #

    def _llm_rerank(self, candidates: List[dict]) -> List[RankedArticle]:
        """Send candidates to the LLM for final scoring and ranking."""
        if not candidates:
            return []

        digest_list = "\n\n".join([
            f"ID: {c['digest_id']}\nTitle: {c['title']}\nSummary: {c['summary']}\nType: {c['article_type']}"
            for c in candidates
        ])

        user_prompt = f"""Rerank these {len(candidates)} AI news digests (pre-selected by semantic similarity) based on the user profile:

{digest_list}

Provide a relevance score (0.0-10.0) and rank (1-{len(candidates)}) for each article, ordered from most to least relevant."""

        try:
            response = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.github_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3
                }
            )

            data = response.json()

            if "error" in data:
                logger.error(f"API Error: {data['error']}")
                return []

            raw_text = data["choices"][0]["message"]["content"].strip()
            clean = raw_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)

            ranked_list = RankedDigestList(**parsed)
            return ranked_list.articles if ranked_list else []

        except Exception as e:
            logger.error(f"Error in LLM rerank: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Main entry point (drop-in replacement for old rank_digests)       #
    # ------------------------------------------------------------------ #

    def rank_digests(self, digests: List[dict]) -> List[RankedArticle]:
        """
        Hybrid pipeline:
          1. FAISS vector search -> top 20 semantically similar digests
          2. LLM rerank -> final scored + ranked list

        Falls back to pure LLM ranking if vector store is empty
        (e.g. on first run before any embeddings exist).
        """
        if not digests:
            return []

        if self.vector_store.total() > 0:
            logger.info(f"Vector store has {self.vector_store.total()} entries — using hybrid pipeline.")
            candidates = self._vector_search(top_k=20)

            candidate_ids = {c["digest_id"] for c in candidates}
            digest_map = {d["id"]: d for d in digests}

            merged = []
            for c in candidates:
                d = digest_map.get(c["digest_id"])
                if d:
                    merged.append({
                        "digest_id": d["id"],
                        "article_type": d["article_type"],
                        "title": d["title"],
                        "summary": d["summary"],
                        "url": d["url"],
                    })

            if merged:
                return self._llm_rerank(merged)

        # Fallback: vector store empty, rank all digests with LLM directly
        logger.warning("Vector store empty — falling back to pure LLM ranking.")
        candidates = [
            {
                "digest_id": d["id"],
                "article_type": d["article_type"],
                "title": d["title"],
                "summary": d["summary"],
                "url": d["url"],
            }
            for d in digests
        ]
        return self._llm_rerank(candidates)

    def backfill_embeddings(self, digests: List[dict]) -> int:
        """
        One-time utility: embed all existing digests not yet in the vector store.
        Call this once after upgrading to seed the FAISS index.
        """
        return self.vector_store.rebuild_from_digests(digests, get_embedding)