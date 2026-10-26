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
  id uuid, content text, score double precision, created_at timestamptz,
  session_id text, source_timestamp timestamptz,
  message_index integer, chunk_index integer
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
    select candidate.id, sum(candidate.rrf)::double precision as score
      from (
        select semantic.id, 1.0 / (60.0 + semantic.rank) as rrf from semantic
        union all
        select lexical.id, 1.0 / (60.0 + lexical.rank) as rrf from lexical
      ) candidate
     group by candidate.id
  )
  select m.id, m.content, fused.score, m.created_at, m.session_id,
         m.source_timestamp, m.message_index, m.chunk_index
    from fused join aml.memories m on m.id = fused.id
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
  id uuid, content text, score double precision, created_at timestamptz,
  session_id text, source_timestamp timestamptz,
  message_index integer, chunk_index integer,
  anchor_id uuid, anchor_rank integer, distance integer
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
             partition by n.id order by a.anchor_rank, abs(n.position - a.position)
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
