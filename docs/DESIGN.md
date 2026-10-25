# Dandelion Memory · 设计 v1（最小可交付版）

作者：Lori。依据：对 AML 第一期参赛系统、通用 agent 记忆系统与检索框架的调研。
目标：AML 第二期文本赛道·开源方法榜。先交一个能跑、能测、好迭代的基线，再逐个加变种。
v1 相对 v0：按砚的评审（PR #1，F1–F19）修订。

## 1. 终态一句话

一个 HTTP 服务：Add 把消息原文同步存好、建好索引；Search 用一条可配置的 7 段管线找回 ≤ top_k 条带时间和说话人的原文。另有一个离线评测器，用同一份配置跑公开数据集，按题型出分。

## 2. 对外契约（官方原文，硬约束）

- `GET /health`：无鉴权，返回 200 `{"status":"ok"}`。
- `POST /add`：请求 `{request_id, messages:[{role, content, timestamp?}], user_id, session_id}`。
  写入完成且可立即检索后返回 200 `{success:true, request_id, user_id, session_id}`，三个 ID 原样回显。
- `POST /search`：请求 `{query, options?, user_id, top_k}`。返回 200 `{data:[{id, content, created_at?}]}`，**条数 ≤ top_k**；没有结果返回 `{data:[]}`。第一版**不返回 score**（官方可选；补进来的前后句会让分数和顺序不一致）。
- 鉴权：`Authorization: Bearer <key>`、`Authorization: Token <key>`、`X-Api-Key: <key>` 三种都认。有效 key 来自环境变量 `DANDELION_API_KEYS`（逗号分隔）。缺失或不对返回 401。
- 只按 user_id 隔离，session_id 不做检索过滤。
- 不记录请求体和返回体（日志里只记 request_id、user_id、耗时、条数、状态码）。

**输入校验**（不过就 422，返回 `{"detail":{"reason":...}}`）：
- request_id、user_id、session_id：非空字符串。
- messages：非空数组；每条 role ∈ {user, assistant}；content 为非空字符串；timestamp 可缺省，有则为整数（Unix 毫秒）。
- query：非空字符串；options：缺省或字符串数组；top_k：1–100 的整数。
- 未知字段一律忽略。

**错误码**：
- 422：格式错；同一 request_id 但请求体不同（永久不一致，平台不该重试）。
- 503 + `Retry-After`：数据库或百炼暂时不可用。
- 不返回空的 2xx。

## 3. 数据模型（Postgres，schema `aml`）

存在托管 Postgres（Supabase）的独立 schema `aml`，用只管这个 schema 的专用账号连接。扩展：pgvector（halfvec）、pgroonga。连接走会话模式，关闭服务端预编译语句。

**units**（检索单元，一行一个）
| 列 | 说明 |
|---|---|
| id | text 主键，`u_<uuid>` |
| user_id, session_id, request_id | 来源 |
| turn_key | 同一条原始消息切出的多段共享，`<request_id>#<msg_index>` |
| msg_index | 消息在本次请求里的位置 |
| part, n_parts | 切段序号（0 起）与总段数；短消息为 0 / 1 |
| seq | bigint，全局单调（`nextval`），只表示到达先后 |
| kind | `turn`（第一版只有这一种） |
| role | user / assistant |
| event_ts | timestamptz，可空（来自 timestamp） |
| text | 这一段的原文 |
| embedding | halfvec(1024)，可空（百炼永久拒绝时为空，见 §4） |
| view_sig | 写入配置的指纹（见 §7） |
| status | `active` / `archived` |
| meta | jsonb |

索引：
- `(user_id, status, view_sig)`
- `(user_id, session_id, event_ts, seq, part)`：补前后句用
- pgroonga 索引 `ON units USING pgroonga (id, text)`：**必须带主键 id**，否则 `pgroonga_score` 恒为 0（PGroonga 官方文档）。

第一版向量检索不建 ANN 索引：按 user_id 过滤后精确算距离。

**add_requests**（幂等账本）：`request_id` 主键、`payload_hash`、`user_id`、`session_id`、`created_at`。

**payload_hash** = `sha256(canonical_json({"user_id","session_id","messages"}))`。canonical_json：键排序、无多余空白、UTF-8、`ensure_ascii=False`；messages 只取 role、content、timestamp 三个键，缺省的 timestamp 不出现。

## 4. 写入流程

1. 校验请求 → 算 payload_hash。
2. 查 add_requests：
   - 有，hash 相同 → 直接返回 200；
   - 有，hash 不同 → 422。
3. **Enricher 链**依次处理，产出待写的 units（第一版：`raw_turn` 原文成单元、`chunk_long` 长消息切段）。
4. 批量调 embedding：每批不超过百炼上限；全局并发上限可配置；429 / 5xx / 超时退避重试。
   - 重试用完仍是临时错误 → 整次 Add 返回 503（不落库，平台会重试）。
   - 某条输入被永久拒绝（4xx，比如内容审核） → 这一条 embedding 记为空，照样落库（字面那一路还能找到它），其余照常。
5. 一个事务里：`pg_advisory_xact_lock(hashtext(request_id))` → 再查一次 add_requests（同 §4.2 的两种情况，已存在且 hash 相同就直接提交并返回 200）→ 插 units → 插 add_requests → 提交。
6. 返回 200。任何一步失败返回 5xx，不写半截数据。

## 5. 检索管线（7 段）

每段是一个接口，配置里按名字选实现。一份 YAML 配置 = 一次实验。

| 段 | 接口 | 第一版实现 | 以后的变种 |
|---|---|---|---|
| ① 写入视图 | `Enricher.apply(add_ctx)` | `raw_turn`、`chunk_long` | 时间换算、请忘掉→归档、事实时间线 |
| ② 多路召回 | `Retriever.retrieve(q) -> [Candidate]` | `dense`、`lexical`、`options_dense` | 时间线检索 |
| ③ 融合 | `Fuser.fuse(q, [[Candidate]]) -> [Candidate]` | `rrf` | 加权线性 |
| ④ 重排 | `Reranker.rerank(q, cands)` | `none` | qwen3-rerank / gte-rerank（失败退回原序） |
| ⑤ 补前后句 | `Expander.expand(q, cands)` | `neighbors` | 半径、父段 |
| ⑥ 排序 | `Orderer.order(q, cands)` | `as_is` | 原文优先、时间顺序 |
| ⑦ 截断 | `Packer.pack(q, cands)` | `budget` | — |

### 5.1 各段之间传什么

- **QueryCtx**：`text`、`options`、`user_id`、`top_k`、`view_sig`，以及懒计算的 `query_embedding`（第一次被要的时候算一次，后面复用）。
- **Candidate**：`unit_id`、`fused_score`、`sources: {retriever_name: rank}`、`origin: "retrieved" | "neighbor"`，以及 `unit`（整行数据，见下）。
- 召回段只返回 id 和名次；**融合之后**一次性把需要的单元整行取回，挂到 Candidate.unit 上；补前后句时再按需取邻居。后面几段只用 Candidate.unit，不再查库（邻居除外）。

### 5.2 第一版实现的具体做法

- **dense**：用 query 原文的向量，在 `user_id = q.user_id AND status='active' AND view_sig = q.view_sig AND embedding IS NOT NULL` 范围里按余弦距离取前 k 个。
- **lexical**：把 query 拆成词（按空白和标点切；中日韩字符交给 pgroonga 的默认分词），每个词用 `pgroonga_query_escape` 转义后用 ` OR ` 连起来，用 `text &@~ <拼好的查询>` 加同样的 user_id / status / view_sig 条件，按 `pgroonga_score` 取前 k 个。拼出来是空的就跳过这一路。
- **options_dense**：只在 options 非空时运行；把 options 用换行拼成一段文本算向量，做法同 dense。它是配置里的一个普通 Retriever，不是特殊分支。
- **rrf**：`score(u) = Σ_路 weight_路 / (k + rank_路(u))`，k 默认 60，名次从 1 起。
- **neighbors**：取融合后排序的前 `seeds` 个（默认 20）作种子。每个种子的邻居 = 同一 user_id、同一 session_id、同一 view_sig 里，按 `(event_ts NULLS LAST, seq, part)` 排序时紧挨着它的前 `radius` 个和后 `radius` 个单元（默认 radius=1；切过段的长消息，邻居可以是同一条消息的另一段）。邻居插在种子后面，顺序为“前邻居、后邻居”。
- **as_is**：不动顺序。
- **budget**：按出现顺序去重（同一个 unit_id 只留第一次出现的位置）；依次放入，直到达到 top_k 条，或者生成后的 content 累计 token 数超过 `max_tokens`（默认 100,000，用 tiktoken `o200k_base` 数，给平台 117,760 的输入预算留出格式余量）。

### 5.3 降级

某一路召回失败（比如 query 的 embedding 超时），其余路照常，并在日志里记一笔；所有路都失败才返回 503。

## 6. Formatter（检索时生成）

返回的 content 在检索时由 Formatter 根据 unit 生成，不存在库里；格式由配置给出。默认：

- 有时间：`[{role} | {YYYY-MM-DD HH:MM (Weekday)} UTC] {text}`
- 没时间：`[{role} | #{seq}] {text}`
- 切过段的在末尾加 ` (part {part+1}/{n_parts})`

时间一律按 UTC 显示。平台把原始数据的时间转成毫秒时用的什么时区，由评测器拿公开数据对一次再定要不要改。
`created_at`：有 event_ts 时返回 ISO 8601（UTC），否则不返回。

## 7. 配置

```yaml
name: baseline
view:                          # 写入配置：决定 view_sig
  enrichers:
    - raw_turn
    - {chunk_long: {max_chars: 1600, overlap_chars: 200}}
search:                        # 检索配置：可以对同一份数据换着跑
  retrievers:
    - {dense: {k: 200, weight: 1.0}}
    - {lexical: {k: 200, weight: 1.0}}
    - {options_dense: {k: 100, weight: 0.5}}
  fuser: {rrf: {k: 60}}
  reranker: none
  expander: {neighbors: {seeds: 20, radius: 1}}
  orderer: as_is
  packer: {budget: {max_tokens: 100000}}
  formatter: {default: {}}
```

- **view_sig** = `view` 那一段规范化后的 sha256 前 12 位。写入时存进每个 unit；检索时只查 view_sig 相同的数据。
- 服务启动时读 `DANDELION_PIPELINE`（配置文件路径）。
- 评测器：同一份数据，只换 `search` 段可以直接重跑；`view` 段变了，view_sig 跟着变，必须重新写入。评测器每次运行都在结果里记下完整配置和 view_sig。

## 8. 外部依赖

- **Embedder**：百炼 `text-embedding-v4`，1024 维，OpenAI 兼容端点。query 与文档同一模型。
- **LLM**（第一版服务里不用，只在评测器和以后的变种里用）：统一 OpenAI 兼容客户端，`base_url / api_key / model` 全走环境变量。开发期用中转站 gemini-3-flash；正式评测换 gpt-4o-mini。
- 密钥只从环境变量读，仓库里只有 `.env.example`。

## 9. 评测器

```
Dataset 适配器 → 模拟平台写入（每 20 条消息或 2,000 词切一次 Add） → 调本地服务 Search
→ Answerer（LLM） → Judge（LLM 或精确匹配） → 逐题结果 jsonl + 按题型汇总
```

- 第一批适配器：LoCoMo-Refined、LongMemEval-S（cleaned）。之后 BEAM、PersonaMem v1/v2、CL-bench。
- 抽样：每个数据集按题型分层，固定随机种子，50–100 题。
- 回答和判分提示词参照 AML 公开管线的要点自己写（是否能直接用原文，看它的许可证再定）。
- 每题另记检索诊断：证据是否在返回里、排第几（数据集带证据 id 的才有）。错题分四类：没找到 / 排太后被截 / 内容被截断 / 找对了是回答错。
- 本地分数只看相对涨跌。
- 评测数据只进 `aml` schema，每次运行用独立的 user_id 前缀，跑完可整批删；提供一个按前缀删除的脚本。

## 10. 仓库结构

```
dandelion/          # 服务
  api.py            # FastAPI：/health /add /search，鉴权
  contract.py       # 请求/响应模型与校验
  store.py          # Postgres 读写（Store 接口）
  embed.py  llm.py  # 外部依赖
  pipeline/         # 七段接口 + 注册表 + 各实现 + Formatter
  config.py         # 读配置、算 view_sig
pipelines/*.yaml    # 实验配置
evals/              # 评测器与数据集适配器
tests/              # 契约测试、单元测试
migrations/         # schema
docs/DESIGN.md
```

技术栈：Python 3.12、FastAPI + uvicorn、psycopg 3（连接池）、httpx、tiktoken。许可证 MIT。

## 11. 第一版不做

LLM 写入加工、rerank、问题重写、时间线、请忘掉、ANN 索引、部署和证书。全部在基线跑出来以后，一个一个加。
不为减少数据库往返扭曲接口：官方不按延迟计分，单次请求上限 30 分钟。

## 12. 已知风险

- 开发库容量 500MB。正式评测文本赛道约 6,000 个实例、3 亿 token，开发库装不下；正式评测前要换存储，届时按基线实测的单位体积估算。
- 单个用户最多多少单元：只核了 BEAM 1M（约 2,100 条消息），其他来源没核。精确向量扫描的耗时要在评测器里实测。
- ScriptMem 本地测不了。
