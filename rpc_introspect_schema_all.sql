-- Drops and recreates rpc_introspect_schema with updated definition
DROP FUNCTION IF EXISTS rpc_introspect_schema();

CREATE FUNCTION rpc_introspect_schema()
RETURNS TABLE (
    schema_name text,
    table_name text,
    column_name text,
    data_type text
) AS $$
BEGIN
    RETURN QUERY
    SELECT c.table_schema, c.table_name, c.column_name, c.data_type
    FROM information_schema.columns c
    WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
    ORDER BY c.table_schema, c.table_name, c.ordinal_position;
END;
$$ LANGUAGE plpgsql;
