"""Hybrid library search service.

A single entry point — `HybridSearchService.search()` — handles both Rule and CE
queries (and any future asset type) through the same pipeline. Routes call this
service; they don't write SQL.

SOLID intent:
  * Single responsibility — this module owns the math + SQL of hybrid retrieval.
    Routes own HTTP I/O and validation. Hydration (joining categories/CEs) lives
    elsewhere.
  * Open/closed — adding a new searchable asset type means adding one entry to
    ASSET_REGISTRY. The service code, fusion math, and routes stay unchanged.
  * Dependency inversion — the service receives a callable embedder, so it can
    be unit-tested without loading SentenceTransformer.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from utils.PostgreSQL import execute_query_dict


# Reciprocal Rank Fusion constant. 60 from Cormack et al. 2009 — robust default.
_RRF_K = 60
# Weight on the trigram name-match bonus. Empirically calibrated so a verbatim
# name hit ranks above the strongest pure-semantic top-1 (1/(60+1) ≈ 0.0164)
# without drowning out keyword fusion.
_NAME_BOOST = 0.5


@dataclass(frozen=True)
class AssetSpec:
    """Tells `HybridSearchService` how to query one searchable asset type.

    Anything with (id, name, body, embedding, search_vector, categories) and an
    optional bookmark join table can be plugged in by adding an entry to
    `ASSET_REGISTRY` — no service or route changes required.
    """
    asset_type: str               # tag emitted in result rows ("rule", "ce", ...)
    table: str                    # main table name
    id_col: str                   # primary key column name
    content_col: str              # body column surfaced as `content` in results
    bookmark_table: Optional[str] = None  # for "search my bookmarks" — same id_col


# Open/closed extension point. To add a new searchable asset type, append here.
ASSET_REGISTRY: dict = {
    "rule": AssetSpec(
        asset_type="rule",
        table="rules",
        id_col="rule_id",
        content_col="predicate",
        bookmark_table="rule_bookmarks",
    ),
    "ce": AssetSpec(
        asset_type="ce",
        table="cognitive_elements",
        id_col="ce_id",
        content_col="definition",
        bookmark_table="ce_bookmarks",
    ),
}


class HybridSearchService:
    """RRF-fused hybrid search across one or more asset types.

    Three signals per asset type:
      1. Semantic — pgvector cosine via HNSW
      2. Keyword  — Postgres full-text (websearch_to_tsquery + ts_rank_cd)
      3. Name     — trigram-similarity bonus on verbatim name matches

    Each signal contributes its top-N candidates, ranked. RRF combines them via
    1 / (k + rank). The name signal is added as a multiplicative bonus to catch
    short / acronym queries that tsvector under-ranks.

    The service is stateless; it can be a long-lived singleton or constructed
    per request — both are fine.
    """

    def __init__(self, embedder: Callable[[str], List[float]]):
        # Inject the embedding function so tests can substitute a fake.
        self._embed = embedder

    # ------------------------------------------------------------------ public API

    def search(
        self,
        *,
        query_text: str,
        asset_types: Sequence[str],
        category_ids: Optional[Sequence[int]] = None,
        bookmark_user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[dict]:
        """Return ranked candidate rows. Each row carries id, asset_type, name,
        content, type, categories, final_score. Empty list on no hits or empty
        query."""
        query_text = (query_text or "").strip()
        if not query_text:
            return []

        specs = [ASSET_REGISTRY[t] for t in asset_types if t in ASSET_REGISTRY]
        if not specs:
            return []

        query_vector = self._embed(query_text)
        if not query_vector:
            return []

        sql, params = self._build_sql(
            specs=specs,
            query_vector=query_vector,
            query_text=query_text,
            category_ids=list(category_ids or []),
            bookmark_user_id=bookmark_user_id,
            limit=limit,
        )
        try:
            return execute_query_dict(sql, params) or []
        except Exception:
            # Routes decide how to surface this; we just don't swallow the trace.
            raise

    # ----------------------------------------------------------------- SQL builder

    def _build_sql(
        self,
        *,
        specs: Sequence[AssetSpec],
        query_vector: List[float],
        query_text: str,
        category_ids: List[int],
        bookmark_user_id: Optional[int],
        limit: int,
    ):
        # category_ids contain only ints already; safe to inline.
        cat_filter = ""
        if category_ids:
            cat_ids_sql = ",".join(str(int(cid)) for cid in category_ids)
            cat_filter = f"AND x.categories && ARRAY[{cat_ids_sql}]"

        # pgvector accepts the literal '[1,2,...]' form. Built once and reused so
        # the query text is identical across CTEs (helps Postgres' plan cache).
        qvec_lit = str(query_vector)

        ctes: List[str] = []
        union_parts: List[str] = []
        params: List = []

        for spec in specs:
            if bookmark_user_id is not None and spec.bookmark_table:
                join_clause = (
                    f"JOIN {spec.bookmark_table} bk ON x.{spec.id_col} = bk.{spec.id_col}"
                )
                user_filter = f"AND bk.user_id = {int(bookmark_user_id)}"
            else:
                join_clause = ""
                user_filter = ""

            tag = spec.asset_type

            # 1) Semantic — top-K nearest neighbors via HNSW.
            ctes.append(f"""
            sem_{tag} AS (
                SELECT x.{spec.id_col} AS id, '{tag}' AS asset_type, x.name,
                       x.{spec.content_col} AS content, x.type, x.categories,
                       x.is_local_draft, x.created_by_username, x.public_id,
                       RANK() OVER (ORDER BY x.embedding <=> '{qvec_lit}') AS rk
                FROM {spec.table} x
                {join_clause}
                WHERE x.embedding IS NOT NULL {cat_filter} {user_filter}
                ORDER BY x.embedding <=> '{qvec_lit}'
                LIMIT {limit}
            )""")

            # 2) Keyword — full-text rank on weighted tsvector (name=A, body=B).
            ctes.append(f"""
            kw_{tag} AS (
                SELECT x.{spec.id_col} AS id, '{tag}' AS asset_type, x.name,
                       x.{spec.content_col} AS content, x.type, x.categories,
                       x.is_local_draft, x.created_by_username, x.public_id,
                       RANK() OVER (
                           ORDER BY ts_rank_cd(x.search_vector, websearch_to_tsquery('english', %s)) DESC
                       ) AS rk
                FROM {spec.table} x
                {join_clause}
                WHERE x.search_vector @@ websearch_to_tsquery('english', %s)
                      {cat_filter} {user_filter}
                LIMIT {limit}
            )""")
            params.extend([query_text, query_text])

            # 3) Name match — trigram similarity on verbatim name. Catches
            # acronyms / short queries that tsvector misses. Uses gin_trgm_ops.
            ctes.append(f"""
            nm_{tag} AS (
                SELECT x.{spec.id_col} AS id, '{tag}' AS asset_type, x.name,
                       x.{spec.content_col} AS content, x.type, x.categories,
                       x.is_local_draft, x.created_by_username, x.public_id,
                       similarity(x.name, %s) AS sim
                FROM {spec.table} x
                {join_clause}
                WHERE x.name ILIKE %s {cat_filter} {user_filter}
                LIMIT {limit}
            )""")
            params.extend([query_text, f"%{query_text}%"])

            union_parts.append(
                f"SELECT id, asset_type, name, content, type, categories, is_local_draft, created_by_username, public_id, "
                f"1.0 / ({_RRF_K} + rk) AS contrib FROM sem_{tag}"
            )
            union_parts.append(
                f"SELECT id, asset_type, name, content, type, categories, is_local_draft, created_by_username, public_id, "
                f"1.0 / ({_RRF_K} + rk) AS contrib FROM kw_{tag}"
            )
            union_parts.append(
                f"SELECT id, asset_type, name, content, type, categories, is_local_draft, created_by_username, public_id, "
                f"sim * {_NAME_BOOST} AS contrib FROM nm_{tag}"
            )

        cte_sql = "WITH " + ",\n".join(ctes)
        union_sql = "\n        UNION ALL\n        ".join(union_parts)

        main_sql = f"""
        {cte_sql}
        SELECT id, asset_type, name, content, type, categories, is_local_draft,
               created_by_username, public_id,
               SUM(contrib) AS final_score
        FROM (
            {union_sql}
        ) unified
        GROUP BY id, asset_type, name, content, type, categories, is_local_draft,
                 created_by_username, public_id
        ORDER BY final_score DESC
        LIMIT {limit}
        """
        return main_sql, tuple(params)
