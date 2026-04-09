"""
AGE operations package — Cypher queries are identical to Neo4j.
The only differences are:
- GraphProvider.AGE instead of GraphProvider.NEO4J
- CALL ... IN TRANSACTIONS is not supported in AGE; batched via Python UNWIND
"""

from graphiti_core.driver.age.operations.entity_node_ops import AGEEntityNodeOperations
from graphiti_core.driver.age.operations.entity_edge_ops import AGEEntityEdgeOperations
from graphiti_core.driver.age.operations.episode_node_ops import AGEEpisodeNodeOperations
from graphiti_core.driver.age.operations.community_node_ops import AGECommunityNodeOperations
from graphiti_core.driver.age.operations.community_edge_ops import AGECommunityEdgeOperations
from graphiti_core.driver.age.operations.episodic_edge_ops import AGEEpisodicEdgeOperations
from graphiti_core.driver.age.operations.saga_node_ops import AGESagaNodeOperations
from graphiti_core.driver.age.operations.has_episode_edge_ops import AGEHasEpisodeEdgeOperations
from graphiti_core.driver.age.operations.next_episode_edge_ops import AGENextEpisodeEdgeOperations
from graphiti_core.driver.age.operations.search_ops import AGESearchOperations
from graphiti_core.driver.age.operations.graph_ops import AGEGraphMaintenanceOperations

__all__ = [
    'AGEEntityNodeOperations',
    'AGEEpisodeNodeOperations',
    'AGECommunityNodeOperations',
    'AGESagaNodeOperations',
    'AGEEntityEdgeOperations',
    'AGEEpisodicEdgeOperations',
    'AGECommunityEdgeOperations',
    'AGEHasEpisodeEdgeOperations',
    'AGENextEpisodeEdgeOperations',
    'AGESearchOperations',
    'AGEGraphMaintenanceOperations',
]
