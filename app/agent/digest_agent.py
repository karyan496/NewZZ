import os
import requests
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class DigestOutput(BaseModel):
    title: str
    summary: str


PROMPT = """You are an expert AI news analyst specializing in summarizing technical articles, research papers, and video content about artificial intelligence. Your role is to create concise, informative digests that help readers quickly understand the key points and significance of AI-related content.

Guidelines:
- Create a compelling title (5-10 words) that captures the essence of the content
- Write a 2-3 sentence summary that highlights the main points and why they matter
- Focus on actionable insights and implications
- Use clear, accessible language while maintaining technical accuracy
- Avoid marketing fluff - focus on substance

IMPORTANT: Respond ONLY with a JSON object in this exact format, no preamble or markdown:
{"title": "your title here", "summary": "your summary here"}"""


class DigestAgent:
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.model = "openai/gpt-4.1-mini"
        self.endpoint = "https://models.github.ai/inference/chat/completions"
        self.system_prompt = PROMPT

    def generate_digest(self, title: str, content: str, article_type: str) -> Optional[DigestOutput]:
        try:
            user_prompt = f"Create a digest for this {article_type}: \n Title: {title} \n Content: {content[:8000]}"

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
                    "temperature": 0.7
                }
            )

            data = response.json()

            if "error" in data:
                print(f"API Error: {data['error']}")
                return None

            raw_text = data["choices"][0]["message"]["content"].strip()

            # Parse the JSON response into DigestOutput
            import json
            # Strip markdown code fences if present
            clean = raw_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)

            return DigestOutput(
                title=parsed["title"],
                summary=parsed["summary"]
            )

        except Exception as e:
            print(f"Error generating digest: {e}")
            return None