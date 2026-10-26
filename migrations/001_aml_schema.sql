create schema if not exists aml;

create table if not exists aml.add_requests (
  request_id text primary key,
  user_id text not null,
  session_id text not null,
  payload_hash text not null check (length(payload_hash) = 64),
  archive_ids uuid[] not null default '{}',
  created_at timestamptz not null default now()
);

create table if not exists aml.config (
  key text primary key,
  value text not null,
  updated_at timestamptz not null default now()
);

create table if not exists aml.memories (
  id uuid primary key,
  request_id text not null references aml.add_requests(request_id) on delete cascade,
  user_id text not null,
  session_id text not null,
  message_index integer not null check (message_index >= 0),
  chunk_index integer not null check (chunk_index >= 0),
  role text not null check (role in ('user', 'assistant')),
  source_timestamp timestamptz,
  content text not null check (length(btrim(content)) > 0),
  embedding extensions.vector(1024) not null,
  archived_at timestamptz,
  archived_by_request_id text references aml.add_requests(request_id),
  created_at timestamptz not null default now(),
  unique (request_id, message_index, chunk_index)
);

create index if not exists aml_memories_user_active_idx
  on aml.memories (user_id, session_id, source_timestamp, created_at)
  where archived_at is null;

create index if not exists aml_memories_embedding_hnsw_idx
  on aml.memories using hnsw (embedding extensions.vector_cosine_ops)
  where archived_at is null;

create index if not exists aml_memories_content_pgroonga_idx
  on aml.memories using pgroonga (content)
  where archived_at is null;

alter table aml.add_requests enable row level security;
alter table aml.config enable row level security;
alter table aml.memories enable row level security;
revoke all on all tables in schema aml from anon, authenticated;
grant usage on schema aml to anon, authenticated, service_role;

create or replace function aml.require_storage_token(p_storage_token text)
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
  expected_hash text;
  supplied_hash text;
begin
  select value into expected_hash from aml.config where key = 'storage_token_sha256';
  supplied_hash := pg_catalog.encode(extensions.digest(coalesce(p_storage_token, ''), 'sha256'), 'hex');
  if expected_hash is null or supplied_hash is distinct from expected_hash then
    raise exception using errcode = '42501', message = 'invalid storage token';
  end if;
end;
$$;

revoke all on function aml.require_storage_token(text) from public, anon, authenticated;

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
    (item->>'id')::uuid,
    p_request_id,
    p_user_id,
    p_session_id,
    (item->>'message_index')::integer,
    (item->>'chunk_index')::integer,
    item->>'role',
    nullif(item->>'source_timestamp', '')::timestamptz,
    item->>'content',
    (item->>'embedding')::extensions.vector(1024),
    case when coalesce((item->>'searchable')::boolean, true) then null else now() end,
    case when coalesce((item->>'searchable')::boolean, true) then null else p_request_id end
  from jsonb_array_elements(p_items) as item;

  return jsonb_build_object(
    'inserted', true,
    'request_id', p_request_id,
    'memory_count', jsonb_array_length(p_items),
    'archived_count', coalesce(array_length(p_archive_ids, 1), 0)
  );
end;
$$;

create or replace function public.aml_search_candidates(
  p_user_id text,
  p_query_text text,
  p_query_embedding extensions.vector(1024),
  p_semantic_limit integer default 200,
  p_lexical_limit integer default 200,
  p_result_limit integer default 200,
  p_storage_token text default null
)
returns table (
  id uuid,
  content text,
  score double precision,
  created_at timestamptz,
  session_id text,
  source_timestamp timestamptz,
  message_index integer,
  chunk_index integer
)
language plpgsql
stable
security definer
set search_path = ''
as $$
begin
  perform aml.require_storage_token(p_storage_token);
  return query
  with semantic as (
    select m.id,
           row_number() over (order by m.embedding operator(extensions.<=>) p_query_embedding) as rank
      from aml.memories m
     where m.user_id = p_user_id and m.archived_at is null
     order by m.embedding operator(extensions.<=>) p_query_embedding
     limit greatest(1, least(p_semantic_limit, 1000))
  ), lexical as (
    select m.id,
           row_number() over (order by extensions.pgroonga_score(m.tableoid, m.ctid) desc) as rank
      from aml.memories m
     where m.user_id = p_user_id
       and m.archived_at is null
       and m.content operator(extensions.&@) p_query_text
     order by extensions.pgroonga_score(m.tableoid, m.ctid) desc
     limit greatest(1, least(p_lexical_limit, 1000))
  ), fused as (
    select candidate.id,
           sum(candidate.rrf)::double precision as score
      from (
        select semantic.id, 1.0 / (60.0 + semantic.rank) as rrf from semantic
        union all
        select lexical.id, 1.0 / (60.0 + lexical.rank) as rrf from lexical
      ) candidate
     group by candidate.id
  )
  select m.id, m.content, fused.score, m.created_at, m.session_id,
         m.source_timestamp, m.message_index, m.chunk_index
    from fused
    join aml.memories m on m.id = fused.id
   order by fused.score desc, coalesce(m.source_timestamp, m.created_at) desc, m.id
   limit greatest(1, least(p_result_limit, 1000));
end;
$$;

create or replace function public.aml_neighbor_memories(
  p_user_id text,
  p_anchor_ids uuid[],
  p_radius integer default 1,
  p_storage_token text default null
)
returns table (
  id uuid,
  content text,
  score double precision,
  created_at timestamptz,
  session_id text,
  source_timestamp timestamptz,
  message_index integer,
  chunk_index integer,
  anchor_id uuid,
  anchor_rank integer,
  distance integer
)
language plpgsql
stable
security definer
set search_path = ''
as $$
begin
  perform aml.require_storage_token(p_storage_token);
  return query
  with anchors as (
    select value as id, ordinality::integer as anchor_rank
      from unnest(p_anchor_ids) with ordinality as a(value, ordinality)
  ), ordered as (
    select m.*,
           row_number() over (
             partition by m.session_id
             order by coalesce(m.source_timestamp, m.created_at), m.created_at,
                      m.message_index, m.chunk_index, m.id
           )::integer as position
      from aml.memories m
     where m.user_id = p_user_id and m.archived_at is null
  ), anchor_positions as (
    select a.id, a.anchor_rank, o.session_id, o.position
      from anchors a join ordered o on o.id = a.id
  ), matches as (
    select n.id, n.content, n.created_at, n.session_id, n.source_timestamp,
           n.message_index, n.chunk_index, a.id as anchor_id, a.anchor_rank,
           n.position - a.position as distance,
           row_number() over (
             partition by n.id
             order by a.anchor_rank, abs(n.position - a.position)
           ) as duplicate_rank
      from anchor_positions a
      join ordered n on n.session_id = a.session_id
                    and n.position between a.position - greatest(0, least(p_radius, 3))
                                       and a.position + greatest(0, least(p_radius, 3))
     where n.id <> a.id
  )
  select matches.id, matches.content, 0.0::double precision, matches.created_at,
         matches.session_id, matches.source_timestamp, matches.message_index,
         matches.chunk_index, matches.anchor_id, matches.anchor_rank, matches.distance
   from matches
   where matches.duplicate_rank = 1
   order by matches.anchor_rank, matches.distance;
end;
$$;

revoke all on function public.aml_add_request(text, text, text, text, jsonb, uuid[], text) from public;
revoke all on function public.aml_search_candidates(text, text, extensions.vector, integer, integer, integer, text) from public;
revoke all on function public.aml_neighbor_memories(text, uuid[], integer, text) from public;
grant execute on function public.aml_add_request(text, text, text, text, jsonb, uuid[], text) to anon, authenticated, service_role;
grant execute on function public.aml_search_candidates(text, text, extensions.vector, integer, integer, integer, text) to anon, authenticated, service_role;
grant execute on function public.aml_neighbor_memories(text, uuid[], integer, text) to anon, authenticated, service_role;
