"""
Run this ONCE after upgrading to seed the FAISS index with all
existing digests already in Postgres.

    uv run python services/backfill_embeddings.py
"""
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.agent.curator_agent import CuratorAgent
from app.database.repository import Repository
from app.profiles.user_profile import USER_PROFILE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    repo = Repository()
    curator = CuratorAgent(USER_PROFILE)

    # Fetch ALL digests (no hours limit)
    all_digests = repo.get_recent_digests(hours=99999)
    logger.info(f"Found {len(all_digests)} total digests in Postgres.")

    added = curator.backfill_embeddings(all_digests)
    print(f"\nBackfill complete: {added} new vectors added to FAISS index.")
    print(f"Total vectors in store: {curator.vector_store.total()}")