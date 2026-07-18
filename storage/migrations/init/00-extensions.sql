-- Postgres init: run once on first container boot.
--
-- The Apache AGE image already loads the AGE extension code; we still
-- need CREATE EXTENSION inside the target database and a couple of
-- companions (uuid-ossp for UUID generation, pgcrypto for PII at rest).
--
-- pg_search is the BM25 extension from ParadeDB. We `IF NOT EXISTS` it
-- so the file is a no-op when the image doesn't have pg_search baked
-- in (the app falls back to plain tsvector ranking in that case).

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, "$user", public;
SELECT create_graph('kg');

-- Optional BM25. If the extension isn't installed we just keep tsvector.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'pg_search') THEN
    CREATE EXTENSION IF NOT EXISTS pg_search;
  END IF;
END$$;
