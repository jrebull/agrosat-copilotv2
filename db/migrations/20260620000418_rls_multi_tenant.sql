-- migrate:up
-- US-051: multi-tenant Row-Level Security (RLS) by session_id.
--
-- Real ownership graph (verified against the 4 existing migrations):
--   chat_sessions (id UUID PK)            -> the session IS the row; owner = id
--     |- aois.session_id     UUID NOT NULL FK->chat_sessions ON DELETE CASCADE  [direct owner]
--     |- parcels.session_id  UUID NULL     FK->chat_sessions ON DELETE CASCADE  [direct owner, nullable]
--     |    |- features_parcels.parcel_id BIGINT FK->parcels  ON DELETE CASCADE  [indirect owner; NO session_id]
--     |- rag_documents       -- global PASTIS-R corpus, NOT multi-tenant -> NO RLS (justified out-of-scope)
--
-- Tenant key: current_setting('app.current_session', true)::uuid
--   missing_ok=true (the `true` arg) -> when no session is set the call returns NULL
--   instead of raising; `col = NULL` is NULL -> 0 rows (fail-closed), never an error.
--   This matches the listener contract emitted by ml/agent/db.py and the backend pool:
--     SELECT set_config('app.current_session', $1, true)   -- SET LOCAL semantics.
--
-- Role strategy: a NON-superuser, NOBYPASSRLS application role `agrosat_app` is created
--   here so RLS is observable. The migration role `agrosat` (compose superuser) keeps its
--   implicit BYPASSRLS, so it remains the bypassing migration role (no separate migration
--   role needed). FORCE ROW LEVEL SECURITY makes the policy apply even to the table owner,
--   but a SUPERUSER always bypasses RLS -- hence the isolation test must connect as
--   agrosat_app. In staging/prod the app MUST connect as agrosat_app (or a cloud role
--   without BYPASSRLS); the dev-only password below is acceptable for local compose and is
--   replaced by Secret Manager in prod.

-- Application role (idempotent: CREATE ROLE has no IF NOT EXISTS, guard with a DO block).
-- Dev-only password for local docker-compose; overridden by Secret Manager / Key Vault in prod.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agrosat_app') THEN
        CREATE ROLE agrosat_app
            NOSUPERUSER
            NOBYPASSRLS
            NOCREATEDB
            NOCREATEROLE
            LOGIN
            PASSWORD 'agrosat_app';
    END IF;
END
$$;

-- DML grants on the 4 multi-tenant tables; read-only on the global rag_documents corpus.
GRANT SELECT, INSERT, UPDATE, DELETE ON chat_sessions, aois, parcels, features_parcels TO agrosat_app;
GRANT SELECT ON rag_documents TO agrosat_app;
-- BIGSERIAL backing sequences (aois_id_seq, parcels_id_seq, features_parcels_id_seq) so
-- INSERT can advance the identity; chat_sessions uses gen_random_uuid() (no sequence).
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO agrosat_app;

-- chat_sessions: the session row only ever sees itself (owner = its own id).
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON chat_sessions
    FOR ALL
    USING (id = current_setting('app.current_session', true)::uuid)
    WITH CHECK (id = current_setting('app.current_session', true)::uuid);

-- aois: direct owner via session_id (NOT NULL).
ALTER TABLE aois ENABLE ROW LEVEL SECURITY;
ALTER TABLE aois FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON aois
    FOR ALL
    USING (session_id = current_setting('app.current_session', true)::uuid)
    WITH CHECK (session_id = current_setting('app.current_session', true)::uuid);

-- parcels: direct owner via session_id (nullable). Rows with session_id IS NULL are
-- invisible to every tenant (also fail-closed); acceptable for MVP since demo parcels are
-- seeded with a session. Debt: backfill + NOT NULL in a future rollforward migration.
ALTER TABLE parcels ENABLE ROW LEVEL SECURITY;
ALTER TABLE parcels FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON parcels
    FOR ALL
    USING (session_id = current_setting('app.current_session', true)::uuid)
    WITH CHECK (session_id = current_setting('app.current_session', true)::uuid);

-- features_parcels: NO session_id column -> isolate indirectly via the parent parcel.
-- EXISTS subquery against parcels (whose own RLS policy also applies inside the subquery,
-- giving defense in depth). Cost is per-row; debt: a denormalized session_id column +
-- trigger in a future US. WITH CHECK uses the same predicate so INSERT/UPDATE cannot
-- attach a feature row to a parcel owned by another session.
ALTER TABLE features_parcels ENABLE ROW LEVEL SECURITY;
ALTER TABLE features_parcels FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON features_parcels
    FOR ALL
    USING (
        EXISTS (
            SELECT 1
            FROM parcels p
            WHERE p.id = features_parcels.parcel_id
              AND p.session_id = current_setting('app.current_session', true)::uuid
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1
            FROM parcels p
            WHERE p.id = features_parcels.parcel_id
              AND p.session_id = current_setting('app.current_session', true)::uuid
        )
    );

-- rag_documents: intentionally NO RLS. It is the shared global PASTIS-R corpus (not
-- per-session); read access is public to all tenants. Documented as out-of-scope of RLS.

-- migrate:down
-- Reverse order: drop policies, disable RLS, revoke grants, then drop the role (a role
-- cannot be dropped while it still owns privileges on existing objects).
DROP POLICY IF EXISTS tenant_isolation ON features_parcels;
DROP POLICY IF EXISTS tenant_isolation ON parcels;
DROP POLICY IF EXISTS tenant_isolation ON aois;
DROP POLICY IF EXISTS tenant_isolation ON chat_sessions;

ALTER TABLE features_parcels NO FORCE ROW LEVEL SECURITY;
ALTER TABLE features_parcels DISABLE ROW LEVEL SECURITY;
ALTER TABLE parcels NO FORCE ROW LEVEL SECURITY;
ALTER TABLE parcels DISABLE ROW LEVEL SECURITY;
ALTER TABLE aois NO FORCE ROW LEVEL SECURITY;
ALTER TABLE aois DISABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions NO FORCE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions DISABLE ROW LEVEL SECURITY;

REVOKE USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public FROM agrosat_app;
REVOKE SELECT ON rag_documents FROM agrosat_app;
REVOKE SELECT, INSERT, UPDATE, DELETE ON chat_sessions, aois, parcels, features_parcels FROM agrosat_app;

-- Later migrations (chat_messages, list_chat_sessions) grant more privileges to the
-- role; DROP OWNED BY revokes every privilege it still holds in this database (the
-- role owns no objects) so the role can be dropped regardless of what came after.
DROP OWNED BY agrosat_app;
DROP ROLE IF EXISTS agrosat_app;
