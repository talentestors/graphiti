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

from __future__ import annotations

"""
Apache AGE (PostgreSQL graph extension) driver for Graphiti.

AGE executes Cypher queries via the SQL cypher() function:

    SELECT * FROM cypher('graph_name', $$
        MATCH (n:Entity) RETURN n.uuid, n.name
    $$) AS (uuid agtype, name agtype);

Key differences from Neo4j:
- psycopg2 is synchronous; async is bridged via asyncio.to_thread()
- agtype (AGE's JSON-like type) requires parsing to Python dicts
- Fulltext/vector search uses PostgreSQL FTS + pg_vector (not Neo4j APIs)
"""

import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import psycopg2
    import psycopg2.extensions

from graphiti_core.driver.driver import (
    GraphDriver,
    GraphDriverSession,
    GraphProvider,
)
from graphiti_core.driver.query_executor import Transaction

logger = logging.getLogger(__name__)

# Default graph name used when none is specified
DEFAULT_GRAPH_NAME = 'graphiti'


class AGEDriverSession(GraphDriverSession):
    """AGE session that wraps psycopg2 with async semantics."""

    provider = GraphProvider.AGE

    def __init__(self, conn: psycopg2.extensions.connection, graph_name: str):
        self._conn = conn
        self._graph_name = graph_name
        self._closed = False

    async def __aenter__(self) -> AGEDriverSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def run(self, query: str, **kwargs: Any) -> Any:
        """Execute a Cypher query wrapped in the AGE cypher() SQL function.

        For queries that return data, we wrap in:
            SELECT * FROM cypher('graph', $$ <query> $$) AS (col1 agtype, ...)

        For write-only queries (no RETURN clause), we execute directly via _do_write.
        """
        params = kwargs.pop('params', None) or {}
        params = self._convert_params(params)

        if self._has_return_clause(query):
            wrapped = self._wrap_cypher_read(query, params)
            return await asyncio.to_thread(self._do_read, wrapped)
        else:
            wrapped = self._wrap_cypher_write(query, params)
            return await asyncio.to_thread(self._do_write, wrapped)

    async def close(self) -> None:
        if not self._closed:
            self._closed = True

    async def execute_write(self, func, *args: Any, **kwargs: Any) -> Any:
        # AGE inherits PostgreSQL ACID; execute in autocommit=False transaction
        return await func(self, *args, **kwargs)

    # --- Private helpers ---

    def _has_return_clause(self, query: str) -> bool:
        """Check if the Cypher query has a RETURN clause.

        Strips comments first to avoid false positives.
        """
        stripped = re.sub(r'//.*|/\*[\s\S]*?\*/', '', query).strip()
        # Look for RETURN keyword as a top-level clause (not inside subquery braces)
        return bool(re.search(r'\bRETURN\b', stripped, re.IGNORECASE))

    def _wrap_cypher_read(self, query: str, params: dict[str, Any]) -> str:
        """Wrap a Cypher read query in the cypher() SQL function.

        Extracts RETURN column names from the query and emits:
            SELECT * FROM cypher('graph', $$ <query> $$) AS (col1 agtype, col2 agtype, ...)
        """
        col_names = self._extract_return_columns(query)
        if not col_names:
            col_names = ['result']
        as_clause = ', '.join(f'{name} agtype' for name in col_names)
        return (
            f"SELECT * FROM cypher('{self._graph_name}', $$ "
            f"{query}"
            f" $$) AS ({as_clause})"
        )

    def _wrap_cypher_write(self, query: str, params: dict[str, Any]) -> str:
        """Wrap a Cypher write query in the cypher() SQL function.

        Write-only queries don't return rows, so no AS clause is needed.
        """
        return f"SELECT * FROM cypher('{self._graph_name}', $$ {query} $$)"

    def _extract_return_columns(self, query: str) -> list[str]:
        """Extract column names from a RETURN clause.

        Handles:
            RETURN n.uuid, n.name
            RETURN n.uuid AS id, n.name AS name
            RETURN DISTINCT n.uuid, n.name
            RETURN collect(n.uuid) AS uuids
        """
        stripped = re.sub(r'//.*|/\*[\s\S]*?\*/', '', query).strip()
        match = re.search(r'\bRETURN\s+(.+)', stripped, re.IGNORECASE | re.DOTALL)
        if not match:
            return []
        return_clause = match.group(1)
        # Strip DISTINCT
        return_clause = re.sub(r'^\s*DISTINCT\s+', '', return_clause, flags=re.IGNORECASE).strip()
        # Split on commas that are not inside parentheses
        parts: list[str] = []
        depth = 0
        current = ''
        for ch in return_clause:
            if ch == '(':
                depth += 1
                current += ch
            elif ch == ')':
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                parts.append(current.strip())
                current = ''
            else:
                current += ch
        if current.strip():
            parts.append(current.strip())

        columns: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Handle "AS alias"
            as_match = re.search(r'\bAS\s+(\w+)\s*$', part, re.IGNORECASE)
            if as_match:
                columns.append(as_match.group(1))
            else:
                # Take the last dotted component (e.g. "n.uuid" -> "uuid")
                segments = part.split('.')
                last = segments[-1].strip()
                # Strip aggregate function wrappers like collect(...), count(*), etc.
                last = re.sub(r'^\w+\(', '', last)
                last = re.sub(r'\)\s*$', '', last)
                columns.append(last)
        return columns

    def _convert_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Convert Python values to PostgreSQL-compatible types.

        - datetime objects -> ISO strings
        - UUID objects -> strings
        - lists -> PostgreSQL ARRAY strings (age accepts them as JSON arrays)
        """
        result: dict[str, Any] = {}
        for k, v in params.items():
            if v is None:
                result[k] = None
            elif isinstance(v, uuid.UUID):
                result[k] = str(v)
            elif isinstance(v, list):
                # Serialize list as JSON array; AGE agtype accepts JSON arrays
                result[k] = json.dumps(v)
            elif hasattr(v, 'isoformat'):
                # datetime, date, etc.
                result[k] = v.isoformat()
            else:
                result[k] = v
        return result

    def _do_read(self, sql: str) -> tuple[list[dict[str, Any]], None, None]:
        """Synchronous read execution via psycopg2."""
        cur = self._conn.cursor()
        try:
            cur.execute(sql)
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description] if cur.description else []
            records = [self._parse_agtype_row(dict(zip(col_names, row))) for row in rows]
            return records, None, None
        finally:
            cur.close()

    def _do_write(self, sql: str) -> tuple[list[dict[str, Any]], None, None]:
        """Synchronous write execution via psycopg2 (autocommit-aware)."""
        cur = self._conn.cursor()
        try:
            cur.execute(sql)
            self._conn.commit()
            return [], None, None
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def _parse_agtype_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Parse agtype values in a row to Python dicts.

        agtype returns JSON strings for complex objects, and primitives for scalars.
        """
        result: dict[str, Any] = {}
        for k, v in row.items():
            result[k] = self._parse_agtype_value(v)
        return result

    def _parse_agtype_value(self, v: Any) -> Any:
        """Parse a single agtype value.

        agtype returns:
        - JSON strings for objects/arrays: '{"uuid": "..."}' or '[...]'
        - Python primitives for scalars (str, int, float, bool)
        - None for null
        """
        if v is None:
            return None
        if isinstance(v, str):
            v_stripped = v.strip()
            if v_stripped.startswith('{') or v_stripped.startswith('['):
                try:
                    return json.loads(v_stripped)
                except (json.JSONDecodeError, TypeError):
                    return v_stripped
        return v


class AGEDriver(GraphDriver):
    """Apache AGE driver for Graphiti.

    AGE is a PostgreSQL extension that implements a Cypher query layer.
    This driver wraps the SQL cypher() function to provide a Cypher-native
    interface consistent with the other Graphiti drivers.

    Usage::

        from graphiti_core.driver.age_driver import AGEDriver

        driver = AGEDriver(
            host='localhost',
            port=5432,
            database='graphiti_db',
            user='postgres',
            password='secret',
            graph_name='my_graph',
        )

        async with driver.transaction() as tx:
            await driver.entity_node_ops.save(driver, node, tx=tx)

    Connection parameters follow psycopg2 conventions. The driver can also
    be instantiated via a URI::

        driver = AGEDriver.from_uri('postgresql://postgres:secret@localhost:5432/graphiti_db')
    """

    provider = GraphProvider.AGE

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5432,
        database: str = 'postgres',
        user: str | None = None,
        password: str | None = None,
        graph_name: str = DEFAULT_GRAPH_NAME,
        *,
        # Accept existing psycopg2 connection (for DI / testing)
        _conn: psycopg2.extensions.connection | None = None,
    ):
        """
        Initialize the AGE driver.

        Parameters
        ----------
        host : str
            PostgreSQL host. Defaults to 'localhost'.
        port : int
            PostgreSQL port. Defaults to 5432.
        database : str
            PostgreSQL database name.
        user : str, optional
            PostgreSQL username.
        password : str, optional
            PostgreSQL password.
        graph_name : str
            Name of the AGE graph to use. Created automatically if it doesn't exist.
            Defaults to 'graphiti'.
        _conn : psycopg2.connection, optional
            Existing psycopg2 connection. If provided, host/port/user/password are ignored.
            Intended for dependency injection and testing.
        """
        super().__init__()
        self._host = host
        self._port = port
        self._database = database
        self._user = user or ''
        self._password = password or ''
        self._graph_name = graph_name
        self._database_name = database  # matches Neo4j's database concept

        if _conn is not None:
            self._conn = _conn
        else:
            import psycopg2

            self._conn = psycopg2.connect(
                host=self._host,
                port=self._port,
                dbname=self._database,
                user=self._user,
                password=self._password,
                autocommit=False,
            )

        # Initialize the AGE graph on first use
        self._init_graph_if_not_exists()

        # Instantiate AGE operations (Cypher strings are the same as Neo4j)
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

        self._entity_node_ops = AGEEntityNodeOperations()
        self._episode_node_ops = AGEEpisodeNodeOperations()
        self._community_node_ops = AGECommunityNodeOperations()
        self._saga_node_ops = AGESagaNodeOperations()
        self._entity_edge_ops = AGEEntityEdgeOperations()
        self._episodic_edge_ops = AGEEpisodicEdgeOperations()
        self._community_edge_ops = AGECommunityEdgeOperations()
        self._has_episode_edge_ops = AGEHasEpisodeEdgeOperations()
        self._next_episode_edge_ops = AGENextEpisodeEdgeOperations()
        self._search_ops = AGESearchOperations()
        self._graph_ops = AGEGraphMaintenanceOperations()

    def _init_graph_if_not_exists(self) -> None:
        """Initialize the AGE graph and load the age extension.

        Called once on driver construction. Safe to call multiple times.
        """
        cur = self._conn.cursor()
        try:
            # Enable the age extension
            cur.execute("LOAD 'age'")
            # Set search path so cypher() is resolvable without schema prefix
            cur.execute("SET search_path = ag_catalog, public")
            self._conn.commit()

            # Create graph if it doesn't exist
            cur.execute(
                f"SELECT * FROM cypher('{self._graph_name}', $$ RETURN 1 $$) AS (result agtype) LIMIT 1"
            )
        except Exception as e:
            self._conn.rollback()
            # Graph may not exist yet; try to create it
            try:
                cur.execute(f"SELECT create_graph('{self._graph_name}')")
                self._conn.commit()
                logger.info(f'Created AGE graph: {self._graph_name}')
            except Exception:
                self._conn.rollback()
                logger.debug(f'Graph {self._graph_name} may already exist: {e}')
        finally:
            cur.close()

    @classmethod
    def from_uri(cls, uri: str, graph_name: str = DEFAULT_GRAPH_NAME) -> AGEDriver:
        """Construct an AGEDriver from a postgresql:// URI.

        Example::

            driver = AGEDriver.from_uri('postgresql://postgres:secret@localhost:5432/mydb')
        """
        # Parse postgresql://... URI
        import re as _re

        m = _re.match(
            r'postgresql://(?:(?P<user>[^:@]+)(?::(?P<password>[^@]*))?@)?'
            r'(?P<host>[^:]+)?(?::(?P<port>\d+))?/(?P<database>.+)?',
            uri,
        )
        if not m:
            raise ValueError(f'Invalid postgresql URI: {uri}')
        return cls(
            host=m.group('host') or 'localhost',
            port=int(m.group('port') or 5432),
            database=m.group('database') or 'postgres',
            user=m.group('user'),
            password=m.group('password'),
            graph_name=graph_name,
        )

    # --- GraphDriver interface ---

    @property
    def entity_node_ops(self):
        from graphiti_core.driver.operations.entity_node_ops import EntityNodeOperations

        return self._entity_node_ops

    @property
    def episode_node_ops(self):
        from graphiti_core.driver.operations.episode_node_ops import EpisodeNodeOperations

        return self._episode_node_ops

    @property
    def community_node_ops(self):
        from graphiti_core.driver.operations.community_node_ops import CommunityNodeOperations

        return self._community_node_ops

    @property
    def saga_node_ops(self):
        from graphiti_core.driver.operations.saga_node_ops import SagaNodeOperations

        return self._saga_node_ops

    @property
    def entity_edge_ops(self):
        from graphiti_core.driver.operations.entity_edge_ops import EntityEdgeOperations

        return self._entity_edge_ops

    @property
    def episodic_edge_ops(self):
        from graphiti_core.driver.operations.episodic_edge_ops import EpisodicEdgeOperations

        return self._episodic_edge_ops

    @property
    def community_edge_ops(self):
        from graphiti_core.driver.operations.community_edge_ops import CommunityEdgeOperations

        return self._community_edge_ops

    @property
    def has_episode_edge_ops(self):
        from graphiti_core.driver.operations.has_episode_edge_ops import HasEpisodeEdgeOperations

        return self._has_episode_edge_ops

    @property
    def next_episode_edge_ops(self):
        from graphiti_core.driver.operations.next_episode_edge_ops import NextEpisodeEdgeOperations

        return self._next_episode_edge_ops

    @property
    def search_ops(self):
        from graphiti_core.driver.operations.search_ops import SearchOperations

        return self._search_ops

    @property
    def graph_ops(self):
        from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations

        return self._graph_ops

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Transaction]:
        """AGE transaction with PostgreSQL ACID semantics."""
        cur = self._conn.cursor()
        try:
            yield _AGETansaction(cur)
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    async def execute_query(self, cypher_query_: str, **kwargs: Any) -> Any:
        session = AGEDriverSession(self._conn, self._graph_name)
        try:
            return await session.run(cypher_query_, **kwargs)
        finally:
            await session.close()

    def session(self, database: str | None = None) -> GraphDriverSession:
        _ = database  # AGE uses graph_name, not database
        return AGEDriverSession(self._conn, self._graph_name)

    async def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    async def delete_all_indexes(self) -> None:
        """Drop all AGE graph indexes.

        In AGE, indexes are managed via PostgreSQL CREATE/DROP INDEX commands.
        We enumerate and drop indexes matching the graph's label patterns.
        """
        cur = self._conn.cursor()
        try:
            # AGE stores graph data in PostgreSQL tables prefixed with the graph name.
            # List indexes on those tables and drop them.
            cur.execute("""
                SELECT indexname
                FROM pg_indexes
                WHERE tablename LIKE '%%' AND schemaname = 'ag_catalog'
            """)
            for (idx_name,) in cur.fetchall():
                if idx_name:
                    try:
                        cur.execute(f'DROP INDEX IF EXISTS ag_catalog."{idx_name}" CASCADE')
                    except Exception:
                        pass  # Ignore errors dropping individual indexes
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            logger.warning(f'Error deleting indexes: {e}')
        finally:
            cur.close()

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        """Build graph indices and constraints via Cypher.

        Delegates to the AGEGraphMaintenanceOperations which uses
        the provider-specific index queries from graph_queries.py.
        """
        if delete_existing:
            await self.delete_all_indexes()

        from graphiti_core.graph_queries import get_range_indices

        range_indices = get_range_indices(self.provider)
        for query in range_indices:
            try:
                await self.execute_query(query)
            except Exception as e:
                logger.debug(f'Index may already exist: {e}')

    async def health_check(self) -> None:
        """Check AGE connectivity by running a trivial Cypher query."""
        try:
            await self.execute_query('RETURN 1 AS health_check')
        except Exception as e:
            logger.error(f'AGE health check failed: {e}')
            raise


class _AGETansaction(Transaction):
    """Wraps a psycopg2 cursor for the Transaction ABC."""

    def __init__(self, cur: psycopg2.extensions.cursor):
        self._cur = cur
        self._graph_name = DEFAULT_GRAPH_NAME

    async def run(self, query: str, **kwargs: Any) -> Any:
        # Synchronous execution within the transaction
        params = kwargs.pop('params', None) or {}
        # Convert Python values
        for k, v in params.items():
            if hasattr(v, 'isoformat'):
                params[k] = v.isoformat()
            elif isinstance(v, uuid.UUID):
                params[k] = str(v)
            elif isinstance(v, list):
                params[k] = json.dumps(v)

        has_return = bool(re.search(r'\bRETURN\b', query, re.IGNORECASE))
        if has_return:
            col_names: list[str] = []
            parts = re.split(r'\bRETURN\s+', query, flags=re.IGNORECASE)
            if len(parts) == 2:
                ret_part = re.split(r'\s+WHERE\s+', parts[1].strip(), flags=re.IGNORECASE)[0] if re.search(r'\bWHERE\b', parts[1], re.IGNORECASE) else parts[1].strip()
                # rough column extraction
                col_names = [s.strip().split()[-1].split('.')[-1].rstrip(',;') for s in re.split(r',\s*', ret_part) if s.strip()]
            if not col_names:
                col_names = ['result']
            as_clause = ', '.join(f'{n} agtype' for n in col_names)
            wrapped = (
                f"SELECT * FROM cypher('{self._graph_name}', $$ {query} $$) AS ({as_clause})"
            )
        else:
            wrapped = f"SELECT * FROM cypher('{self._graph_name}', $$ {query} $$)"

        self._cur.execute(wrapped, params)
        if has_return:
            rows = self._cur.fetchall()
            col_names_result = [d[0] for d in self._cur.description] if self._cur.description else []
            return [dict(zip(col_names_result, row)) for row in rows], None, None
        return [], None, None
