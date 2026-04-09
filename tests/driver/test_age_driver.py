"""
AGE driver integration tests.

These tests require a running Apache AGE / PostgreSQL instance.
Run with: pytest tests/driver/test_age_driver.py -v

Set environment variables before running:
    export AGE_HOST=localhost
    export AGE_PORT=5432
    export AGE_DATABASE=graphiti_test
    export AGE_USER=postgres
    export AGE_PASSWORD=secret

Or use a postgresql:// URI:
    export AGE_URI=postgresql://postgres:secret@localhost:5432/graphiti_test
"""

import uuid
from datetime import datetime, timezone

import pytest


@pytest.fixture
def age_driver():
    """Create an AGE driver connected to a test database.

    Requires AGE_HOST, AGE_PORT, AGE_DATABASE, AGE_USER, AGE_PASSWORD env vars
    or AGE_URI (postgresql://...).
    """
    import os

    from graphiti_core.driver.age_driver import AGEDriver

    uri = os.environ.get('AGE_URI', '')
    if uri:
        driver = AGEDriver.from_uri(uri, graph_name=f'test_{uuid.uuid4().hex[:8]}')
    else:
        driver = AGEDriver(
            host=os.environ.get('AGE_HOST', 'localhost'),
            port=int(os.environ.get('AGE_PORT', 5432)),
            database=os.environ.get('AGE_DATABASE', 'graphiti_test'),
            user=os.environ.get('AGE_USER', 'postgres'),
            password=os.environ.get('AGE_PASSWORD', 'postgres'),
            graph_name=f'test_{uuid.uuid4().hex[:8]}',
        )
    yield driver
    # Cleanup: clear all nodes
    import asyncio

    async def cleanup():
        try:
            await driver.execute_query('MATCH (n) DETACH DELETE n')
        except Exception:
            pass
        await driver.close()

    asyncio.run(cleanup())


class TestAGESession:
    async def test_session_run_with_return(self, age_driver):
        """Execute a Cypher query that returns data."""
        session = age_driver.session()
        async with session:
            result = await session.run('RETURN 1 AS num, "hello" AS text')
            assert len(result[0]) == 1
            assert result[0][0]['num'] == 1
            assert result[0][0]['text'] == 'hello'

    async def test_session_run_write(self, age_driver):
        """Execute a Cypher write query (no RETURN)."""
        session = age_driver.session()
        node_uuid = str(uuid.uuid4())
        async with session:
            await session.run(
                'CREATE (n:Entity {uuid: $uuid, name: "Test Node", group_id: "test"})',
                uuid=node_uuid,
            )

    async def test_transaction_commit(self, age_driver):
        """Verify transaction is committed on clean exit."""
        node_uuid = str(uuid.uuid4())
        async with age_driver.transaction() as tx:
            await tx.run(
                'CREATE (n:Entity {uuid: $uuid, name: "TX Node", group_id: "tx_test"})',
                uuid=node_uuid,
            )
        # Verify committed: read it back
        session = age_driver.session()
        async with session:
            result = await session.run(
                'MATCH (n:Entity {uuid: $uuid}) RETURN n.name AS name',
                uuid=node_uuid,
            )
            assert len(result[0]) == 1

    async def test_transaction_rollback(self, age_driver):
        """Verify transaction is rolled back on exception."""
        node_uuid = str(uuid.uuid4())
        with pytest.raises(ZeroDivisionError):
            async with age_driver.transaction() as tx:
                await tx.run(
                    'CREATE (n:Entity {uuid: $uuid, name: "Rollback Test", group_id: "rb_test"})',
                    uuid=node_uuid,
                )
                raise ZeroDivisionError('intentional rollback')

        # Verify node was NOT created
        session = age_driver.session()
        async with session:
            result = await session.run(
                'MATCH (n:Entity {uuid: $uuid}) RETURN count(n) AS cnt',
                uuid=node_uuid,
            )
            assert result[0][0]['cnt'] == 0


class TestAGENodeOperations:
    async def test_save_and_retrieve_entity_node(self, age_driver):
        """Create and retrieve an EntityNode via AGEDriver."""
        from graphiti_core.nodes import EntityNode

        node = EntityNode(
            uuid=str(uuid.uuid4()),
            name='AGE Test Entity',
            group_id='age_ops_test',
            summary='A test entity stored in Apache AGE',
            created_at=datetime.now(timezone.utc),
        )
        await age_driver.entity_node_ops.save(age_driver, node)

        # Retrieve it
        retrieved = await age_driver.entity_node_ops.get_by_uuid(age_driver, node.uuid)
        assert retrieved.uuid == node.uuid
        assert retrieved.name == node.name
        assert retrieved.group_id == node.group_id

    async def test_save_bulk_nodes(self, age_driver):
        """Bulk save multiple nodes."""
        from graphiti_core.nodes import EntityNode

        nodes = [
            EntityNode(
                uuid=str(uuid.uuid4()),
                name=f'Bulk Node {i}',
                group_id='age_bulk_test',
                created_at=datetime.now(timezone.utc),
            )
            for i in range(5)
        ]
        await age_driver.entity_node_ops.save_bulk(age_driver, nodes)

        uuids = [n.uuid for n in nodes]
        retrieved = await age_driver.entity_node_ops.get_by_uuids(age_driver, uuids)
        assert len(retrieved) == 5

    async def test_delete_node(self, age_driver):
        """Delete a node."""
        from graphiti_core.nodes import EntityNode

        node = EntityNode(
            uuid=str(uuid.uuid4()),
            name='To Delete',
            group_id='age_delete_test',
            created_at=datetime.now(timezone.utc),
        )
        await age_driver.entity_node_ops.save(age_driver, node)
        await age_driver.entity_node_ops.delete(age_driver, node)

        from graphiti_core.errors import NodeNotFoundError

        with pytest.raises(NodeNotFoundError):
            await age_driver.entity_node_ops.get_by_uuid(age_driver, node.uuid)


class TestAGEEdgeOperations:
    async def test_save_and_retrieve_edge(self, age_driver):
        """Create nodes and a RELATES_TO edge between them."""
        from graphiti_core.edges import EntityEdge
        from graphiti_core.nodes import EntityNode

        source = EntityNode(
            uuid=str(uuid.uuid4()),
            name='Source Node',
            group_id='age_edge_test',
            created_at=datetime.now(timezone.utc),
        )
        target = EntityNode(
            uuid=str(uuid.uuid4()),
            name='Target Node',
            group_id='age_edge_test',
            created_at=datetime.now(timezone.utc),
        )
        await age_driver.entity_node_ops.save(age_driver, source)
        await age_driver.entity_node_ops.save(age_driver, target)

        edge = EntityEdge(
            uuid=str(uuid.uuid4()),
            source_node_uuid=source.uuid,
            target_node_uuid=target.uuid,
            name='test_rel',
            fact='is related to',
            group_id='age_edge_test',
            created_at=datetime.now(timezone.utc),
        )
        await age_driver.entity_edge_ops.save(age_driver, edge)

        retrieved = await age_driver.entity_edge_ops.get_by_uuid(age_driver, edge.uuid)
        assert retrieved.uuid == edge.uuid
        assert retrieved.source_node_uuid == source.uuid
        assert retrieved.target_node_uuid == target.uuid


class TestAGESearchOperations:
    async def test_node_fulltext_search(self, age_driver):
        """Full-text search via ILIKE."""
        from graphiti_core.nodes import EntityNode
        from graphiti_core.search.search_filters import SearchFilters

        # Create test nodes
        for name in ['Apple fruit', 'Banana split', 'Apple computer']:
            node = EntityNode(
                uuid=str(uuid.uuid4()),
                name=name,
                group_id='age_fts_test',
                created_at=datetime.now(timezone.utc),
            )
            await age_driver.entity_node_ops.save(age_driver, node)

        # Search for "Apple"
        results = await age_driver.search_ops.node_fulltext_search(
            age_driver,
            query='Apple',
            search_filter=SearchFilters(),
            group_ids=['age_fts_test'],
            limit=10,
        )
        names = {r.name for r in results}
        assert 'Apple fruit' in names
        assert 'Apple computer' in names
        assert 'Banana split' not in names

    async def test_node_bfs_search(self, age_driver):
        """BFS traversal search."""
        from graphiti_core.nodes import EntityNode
        from graphiti_core.search.search_filters import SearchFilters

        # Create a chain: A -> B -> C
        a = EntityNode(
            uuid='a', name='A', group_id='age_bfs_test', created_at=datetime.now(timezone.utc)
        )
        b = EntityNode(
            uuid='b', name='B', group_id='age_bfs_test', created_at=datetime.now(timezone.utc)
        )
        c = EntityNode(
            uuid='c', name='C', group_id='age_bfs_test', created_at=datetime.now(timezone.utc)
        )
        for node in [a, b, c]:
            await age_driver.entity_node_ops.save(age_driver, node)

        # Create edges A->B and B->C
        from graphiti_core.edges import EntityEdge

        ab = EntityEdge(
            uuid='ab',
            source_node_uuid='a',
            target_node_uuid='b',
            name='a_to_b',
            fact='',
            group_id='age_bfs_test',
            created_at=datetime.now(timezone.utc),
        )
        bc = EntityEdge(
            uuid='bc',
            source_node_uuid='b',
            target_node_uuid='c',
            name='b_to_c',
            fact='',
            group_id='age_bfs_test',
            created_at=datetime.now(timezone.utc),
        )
        await age_driver.entity_edge_ops.save(age_driver, ab)
        await age_driver.entity_edge_ops.save(age_driver, bc)

        # BFS from A, depth 2 -> should find B and C
        results = await age_driver.search_ops.node_bfs_search(
            age_driver,
            origin_uuids=['a'],
            search_filter=SearchFilters(),
            max_depth=2,
            group_ids=['age_bfs_test'],
            limit=10,
        )
        found_names = {r.name for r in results}
        assert 'B' in found_names
        assert 'C' in found_names
        assert 'A' not in found_names  # origin is not included in results


class TestAGEMiscellaneous:
    async def test_health_check(self, age_driver):
        """Health check should succeed with a running AGE instance."""
        await age_driver.health_check()  # Should not raise

    async def test_provider_enum(self, age_driver):
        """Verify the provider is set correctly."""
        assert age_driver.provider.value == 'age'

    async def test_from_uri_factory(self):
        """Test AGEDriver.from_uri() correctly parses postgresql:// URIs."""
        from graphiti_core.driver.age_driver import AGEDriver
        from graphiti_core.driver.driver import GraphProvider

        driver = AGEDriver.from_uri(
            'postgresql://myuser:mypass@db.example.com:5432/mydb',
            graph_name='uri_test_graph',
        )
        assert driver.provider == GraphProvider.AGE
        assert driver._graph_name == 'uri_test_graph'
        await driver.close()
