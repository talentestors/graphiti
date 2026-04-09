"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import math
from typing import Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.operations.search_ops import SearchOperations
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.driver.record_parsers import (
    community_node_from_record,
    entity_edge_from_record,
    entity_node_from_record,
    episodic_node_from_record,
)
from graphiti_core.edges import EntityEdge
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
from graphiti_core.models.nodes.node_db_queries import (
    COMMUNITY_NODE_RETURN,
    EPISODIC_NODE_RETURN,
    get_entity_node_return_query,
)
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 128


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_fts_query(
    query: str,
    group_ids: list[str] | None = None,
    max_query_length: int = MAX_QUERY_LENGTH,
) -> str:
    """Build a PostgreSQL ILIKE pattern for AGE full-text search.

    Since AGE does not support db.idx.fulltext.queryNodes(), we use
    Cypher's native string matching via ILIKE as a fallback.
    This provides basic substring/pattern matching over node properties.
    """
    # Sanitize: escape LIKE special characters
    sanitized = query.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
    if len(sanitized) > max_query_length:
        return ''
    return sanitized


class AGESearchOperations(SearchOperations):
    # --- Node search ---

    async def node_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityNode]:
        """Full-text search using Cypher ILIKE pattern matching.

        AGE does not support Neo4j's db.idx.fulltext.queryNodes().
        We use Cypher's native ILIKE on node properties as a fallback.
        For production use with large datasets, consider adding a
        PostgreSQL GIN trigram index on Entity.name and Entity.summary.
        """
        fts_pattern = _build_fts_query(query, group_ids)
        if fts_pattern == '':
            return []

        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filter, GraphProvider.AGE
        )

        if group_ids is not None:
            filter_queries.append('n.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' WHERE ' + (' AND '.join(filter_queries))

        # ILIKE pattern: match words anywhere in name or summary
        pattern = f'%{fts_pattern}%'
        filter_query += ' AND (n.name ILIKE $fts_pattern OR n.summary ILIKE $fts_pattern)'
        filter_params['fts_pattern'] = pattern

        cypher = (
            """
            MATCH (n:Entity)
            """
            + filter_query
            + """
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.AGE)
            + """
            LIMIT $limit
            """
        )
        filter_params['limit'] = limit

        records, _, _ = await executor.execute_query(cypher, **filter_params)
        return [entity_node_from_record(r) for r in records]

    async def node_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[EntityNode]:
        """Vector similarity search using Cypher + Python-side computation.

        AGE does not support vector.similarity.cosine() natively (unlike Neo4j).
        Strategy: fetch candidate nodes via Cypher, compute cosine similarity
        in Python, then filter by min_score.

        For better performance with large datasets, consider installing the
        age-extras extension which provides vector functions inside Cypher.
        """
        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filter, GraphProvider.AGE
        )

        if group_ids is not None:
            filter_queries.append('n.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' WHERE ' + (' AND '.join(filter_queries))

        # Fetch nodes with non-null embeddings
        cypher = (
            """
            MATCH (n:Entity)
            """
            + filter_query
            + """
            WHERE n.name_embedding IS NOT NULL
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.AGE)
        )
        filter_params['limit'] = limit

        records, _, _ = await executor.execute_query(cypher, **filter_params)

        # Compute cosine similarity in Python
        scored: list[tuple[EntityNode, float]] = []
        for r in records:
            try:
                node = entity_node_from_record(r)
                if node.name_embedding and len(node.name_embedding) == len(search_vector):
                    score = _cosine_similarity(node.name_embedding, search_vector)
                    if score >= min_score:
                        scored.append((node, score))
            except Exception:
                continue

        scored.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in scored[:limit]]

    async def node_bfs_search(
        self,
        executor: QueryExecutor,
        origin_uuids: list[str],
        search_filter: SearchFilters,
        max_depth: int,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityNode]:
        if not origin_uuids or max_depth < 1:
            return []

        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filter, GraphProvider.AGE
        )

        if group_ids is not None:
            filter_queries.append('n.group_id = origin.group_id')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' AND '.join(filter_queries)

        cypher = (
            f"""
            UNWIND $bfs_origin_node_uuids AS origin_uuid
            MATCH (origin {{uuid: origin_uuid}})-[:RELATES_TO|MENTIONS*1..{max_depth}]->(n:Entity)
            """
            + ('WHERE ' + filter_query if filter_query else '')
            + """
            RETURN
            """
            + get_entity_node_return_query(GraphProvider.AGE)
            + """
            LIMIT $limit
            """
        )
        filter_params['bfs_origin_node_uuids'] = origin_uuids
        filter_params['limit'] = limit

        records, _, _ = await executor.execute_query(cypher, **filter_params)
        return [entity_node_from_record(r) for r in records]

    # --- Edge search ---

    async def edge_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityEdge]:
        """Full-text search on edges using Cypher ILIKE."""
        fts_pattern = _build_fts_query(query, group_ids)
        if fts_pattern == '':
            return []

        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filter, GraphProvider.AGE
        )

        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' AND ' + (' AND '.join(filter_queries))

        pattern = f'%{fts_pattern}%'
        filter_query += ' AND (e.name ILIKE $fts_pattern OR e.fact ILIKE $fts_pattern)'
        filter_params['fts_pattern'] = pattern

        cypher = (
            """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
            """
            + ('WHERE ' + filter_query if filter_query else '')
            + """
            RETURN
            """
            + get_entity_edge_return_query(GraphProvider.AGE)
            + """
            LIMIT $limit
            """
        )
        filter_params['limit'] = limit

        records, _, _ = await executor.execute_query(cypher, **filter_params)
        return [entity_edge_from_record(r) for r in records]

    async def edge_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        source_node_uuid: str | None,
        target_node_uuid: str | None,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[EntityEdge]:
        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filter, GraphProvider.AGE
        )

        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

            if source_node_uuid is not None:
                filter_params['source_uuid'] = source_node_uuid
                filter_queries.append('n.uuid = $source_uuid')

            if target_node_uuid is not None:
                filter_params['target_uuid'] = target_node_uuid
                filter_queries.append('m.uuid = $target_uuid')

        filter_query = ''
        if filter_queries:
            filter_query = ' AND '.join(filter_queries)

        # Fetch edges with embeddings, compute similarity in Python
        cypher = (
            """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
            """
            + ('WHERE ' + filter_query if filter_query else '')
            + """
            WHERE e.fact_embedding IS NOT NULL
            RETURN
            """
            + get_entity_edge_return_query(GraphProvider.AGE)
        )

        records, _, _ = await executor.execute_query(cypher, **filter_params)

        scored: list[tuple[EntityEdge, float]] = []
        for r in records:
            try:
                edge = entity_edge_from_record(r)
                if edge.fact_embedding and len(edge.fact_embedding) == len(search_vector):
                    score = _cosine_similarity(edge.fact_embedding, search_vector)
                    if score >= min_score:
                        scored.append((edge, score))
            except Exception:
                continue

        scored.sort(key=lambda x: x[1], reverse=True)
        return [edge for edge, _ in scored[:limit]]

    async def edge_bfs_search(
        self,
        executor: QueryExecutor,
        origin_uuids: list[str],
        max_depth: int,
        search_filter: SearchFilters,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EntityEdge]:
        if not origin_uuids:
            return []

        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filter, GraphProvider.AGE
        )

        if group_ids is not None:
            filter_queries.append('e.group_id IN $group_ids')
            filter_params['group_ids'] = group_ids

        filter_query = ''
        if filter_queries:
            filter_query = ' AND '.join(filter_queries)

        cypher = (
            f"""
            UNWIND $bfs_origin_node_uuids AS origin_uuid
            MATCH path = (origin {{uuid: origin_uuid}})-[:RELATES_TO|MENTIONS*1..{max_depth}]->(:Entity)
            UNWIND relationships(path) AS rel
            MATCH (n:Entity)-[e:RELATES_TO {{uuid: rel.uuid}}]-(m:Entity)
            """
            + ('WHERE ' + filter_query if filter_query else '')
            + """
            RETURN DISTINCT
            """
            + get_entity_edge_return_query(GraphProvider.AGE)
            + """
            LIMIT $limit
            """
        )
        filter_params['bfs_origin_node_uuids'] = origin_uuids
        filter_params['limit'] = limit

        records, _, _ = await executor.execute_query(cypher, **filter_params)
        return [entity_edge_from_record(r) for r in records]

    # --- Episode search ---

    async def episode_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        search_filter: SearchFilters,  # noqa: ARG002
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[EpisodicNode]:
        fts_pattern = _build_fts_query(query, group_ids)
        if fts_pattern == '':
            return []

        filter_params: dict[str, Any] = {}
        group_filter_query = ''
        if group_ids is not None:
            group_filter_query = 'e.group_id IN $group_ids'
            filter_params['group_ids'] = group_ids

        pattern = f'%{fts_pattern}%'
        filter_params['fts_pattern'] = pattern

        # ILIKE on episode content, source, and source_description
        cypher = (
            """
            MATCH (e:Episodic)
            WHERE e.content ILIKE $fts_pattern
                OR e.source ILIKE $fts_pattern
                OR e.source_description ILIKE $fts_pattern
            """
            + (' AND ' + group_filter_query if group_filter_query else '')
            + """
            RETURN
            """
            + EPISODIC_NODE_RETURN
            + """
            LIMIT $limit
            """
        )
        filter_params['limit'] = limit

        records, _, _ = await executor.execute_query(cypher, **filter_params)
        return [episodic_node_from_record(r) for r in records]

    # --- Community search ---

    async def community_fulltext_search(
        self,
        executor: QueryExecutor,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[CommunityNode]:
        fts_pattern = _build_fts_query(query, group_ids)
        if fts_pattern == '':
            return []

        filter_params: dict[str, Any] = {}
        group_filter_query = ''
        if group_ids is not None:
            group_filter_query = 'WHERE c.group_id IN $group_ids'
            filter_params['group_ids'] = group_ids

        pattern = f'%{fts_pattern}%'
        filter_params['fts_pattern'] = pattern

        cypher = (
            """
            MATCH (c:Community)
            WHERE c.name ILIKE $fts_pattern
            """
            + (' ' + group_filter_query if group_filter_query else '')
            + """
            RETURN
            """
            + COMMUNITY_NODE_RETURN
            + """
            LIMIT $limit
            """
        )
        filter_params['limit'] = limit

        records, _, _ = await executor.execute_query(cypher, **filter_params)
        return [community_node_from_record(r) for r in records]

    async def community_similarity_search(
        self,
        executor: QueryExecutor,
        search_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.6,
    ) -> list[CommunityNode]:
        query_params: dict[str, Any] = {}

        group_filter_query = ''
        if group_ids is not None:
            group_filter_query += 'WHERE c.group_id IN $group_ids'
            query_params['group_ids'] = group_ids

        cypher = (
            'MATCH (c:Community)'
            + (' ' + group_filter_query if group_filter_query else '')
            + """
            WHERE c.name_embedding IS NOT NULL
            RETURN
            """
            + COMMUNITY_NODE_RETURN
        )

        records, _, _ = await executor.execute_query(cypher, **query_params)

        scored: list[tuple[CommunityNode, float]] = []
        for r in records:
            try:
                node = community_node_from_record(r)
                if node.name_embedding and len(node.name_embedding) == len(search_vector):
                    score = _cosine_similarity(node.name_embedding, search_vector)
                    if score >= min_score:
                        scored.append((node, score))
            except Exception:
                continue

        scored.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in scored[:limit]]

    # --- Rerankers ---

    async def node_distance_reranker(
        self,
        executor: QueryExecutor,
        node_uuids: list[str],
        center_node_uuid: str,
        min_score: float = 0,
    ) -> list[EntityNode]:
        filtered_uuids = [u for u in node_uuids if u != center_node_uuid]
        scores: dict[str, float] = {center_node_uuid: 0.0}

        cypher = """
        UNWIND $node_uuids AS node_uuid
        MATCH (center:Entity {uuid: $center_uuid})-[:RELATES_TO]-(n:Entity {uuid: node_uuid})
        RETURN 1 AS score, node_uuid AS uuid
        """

        results, _, _ = await executor.execute_query(
            cypher,
            node_uuids=filtered_uuids,
            center_uuid=center_node_uuid,
        )

        for result in results:
            scores[result['uuid']] = result['score']

        for uuid in filtered_uuids:
            if uuid not in scores:
                scores[uuid] = float('inf')

        filtered_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

        if center_node_uuid in node_uuids:
            scores[center_node_uuid] = 0.1
            filtered_uuids = [center_node_uuid] + filtered_uuids

        reranked_uuids = [u for u in filtered_uuids if (1 / scores[u]) >= min_score]

        if not reranked_uuids:
            return []

        # Fetch the actual EntityNode objects
        get_query = """
            MATCH (n:Entity)
            WHERE n.uuid IN $uuids
            RETURN
            """ + get_entity_node_return_query(GraphProvider.AGE)

        records, _, _ = await executor.execute_query(get_query, uuids=reranked_uuids)

        node_map = {r['uuid']: entity_node_from_record(r) for r in records}
        return [node_map[u] for u in reranked_uuids if u in node_map]

    async def episode_mentions_reranker(
        self,
        executor: QueryExecutor,
        node_uuids: list[str],
        min_score: float = 0,
    ) -> list[EntityNode]:
        if not node_uuids:
            return []

        scores: dict[str, float] = {}

        results, _, _ = await executor.execute_query(
            """
            UNWIND $node_uuids AS node_uuid
            MATCH (episode:Episodic)-[r:MENTIONS]->(n:Entity {uuid: node_uuid})
            RETURN count(*) AS score, n.uuid AS uuid
            """,
            node_uuids=node_uuids,
        )

        for result in results:
            scores[result['uuid']] = result['score']

        for uuid in node_uuids:
            if uuid not in scores:
                scores[uuid] = float('inf')

        sorted_uuids = list(node_uuids)
        sorted_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

        reranked_uuids = [u for u in sorted_uuids if scores[u] >= min_score]

        if not reranked_uuids:
            return []

        # Fetch the actual EntityNode objects
        get_query = """
            MATCH (n:Entity)
            WHERE n.uuid IN $uuids
            RETURN
            """ + get_entity_node_return_query(GraphProvider.AGE)

        records, _, _ = await executor.execute_query(get_query, uuids=reranked_uuids)

        node_map = {r['uuid']: entity_node_from_record(r) for r in records}
        return [node_map[u] for u in reranked_uuids if u in node_map]

    # --- Filter builders ---

    def build_node_search_filters(self, search_filters: SearchFilters) -> Any:
        filter_queries, filter_params = node_search_filter_query_constructor(
            search_filters, GraphProvider.AGE
        )
        return {'filter_queries': filter_queries, 'filter_params': filter_params}

    def build_edge_search_filters(self, search_filters: SearchFilters) -> Any:
        filter_queries, filter_params = edge_search_filter_query_constructor(
            search_filters, GraphProvider.AGE
        )
        return {'filter_queries': filter_queries, 'filter_params': filter_params}

    # --- Fulltext query builder ---

    def build_fulltext_query(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_query_length: int = 8000,
    ) -> str:
        return _build_fts_query(query, group_ids, max_query_length)
