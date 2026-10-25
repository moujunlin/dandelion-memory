create or replace function public.aml_add_request(
  p_request_id text,
  p_user_id text,
  p_session_id text,
  p_payload_hash text,
  p_items jsonb,
  p_archive_ids uuid[] default '{}',
  p_storage_token text default null
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  inserted_count integer := 0;
  existing aml.add_requests%rowtype;
begin
  perform aml.require_storage_token(p_storage_token);
  if jsonb_typeof(p_items) <> 'array' or jsonb_array_length(p_items) = 0 then
    raise exception using errcode = '22023', message = 'p_items must be a non-empty array';
  end if;

  insert into aml.add_requests(request_id, user_id, session_id, payload_hash, archive_ids)
  values (p_request_id, p_user_id, p_session_id, p_payload_hash, coalesce(p_archive_ids, '{}'))
  on conflict (request_id) do nothing;
  get diagnostics inserted_count = row_count;

  if inserted_count = 0 then
    select * into existing from aml.add_requests where request_id = p_request_id;
    if existing.user_id is distinct from p_user_id
       or existing.session_id is distinct from p_session_id
       or existing.payload_hash is distinct from p_payload_hash then
      raise exception using
        errcode = '23505',
        message = 'AML_IDEMPOTENCY_CONFLICT: request_id reused with a different payload';
    end if;
    return jsonb_build_object('inserted', false, 'request_id', p_request_id);
  end if;

  update aml.memories
     set archived_at = now(), archived_by_request_id = p_request_id
   where user_id = p_user_id
     and archived_at is null
     and id = any(coalesce(p_archive_ids, '{}'));

  insert into aml.memories(
    id, request_id, user_id, session_id, message_index, chunk_index,
    role, source_timestamp, content, embedding, archived_at, archived_by_request_id
  )
  select
    (item->>'id')::uuid, p_request_id, p_user_id, p_session_id,
    (item->>'message_index')::integer, (item->>'chunk_index')::integer,
    item->>'role', nullif(item->>'source_timestamp', '')::timestamptz,
    item->>'content', (item->>'embedding')::extensions.vector(1024),
    case when coalesce((item->>'searchable')::boolean, true) then null else now() end,
    case when coalesce((item->>'searchable')::boolean, true) then null else p_request_id end
  from jsonb_array_elements(p_items) as item;

  return jsonb_build_object(
    'inserted', true, 'request_id', p_request_id,
    'memory_count', jsonb_array_length(p_items),
    'archived_count', coalesce(array_length(p_archive_ids, 1), 0)
      + (select count(*) from jsonb_array_elements(p_items) item
           where not coalesce((item->>'searchable')::boolean, true))
  );
end;
$$;
