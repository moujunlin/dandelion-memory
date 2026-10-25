# Dandelion Memory · 设计 v0（最小可交付版）

作者：Lori。依据：`../aml-research/01–06`（调研）、lori-backend#234（需求与决定）。
目标：AML 第二期文本赛道·开源方法榜。先交一个能跑、能测、好迭代的基线，再逐个加变种。

## 1. 终态一句话

一个 HTTP 服务：Add 把消息原文同步存好、建好索引；Search 用一条可配置的 7 段管线找回 ≤ top_k 条带时间和说话人的原文。另有一个离线评测器，用同一份配置跑公开数据集，按题型出分。

## 2. 对外契约（官方原文，硬约束）

- `GET /health`：无鉴权，返回 2xx。
- `POST /add`：请求 `{request_id, messages:[{role, content, timestamp?(ms)}], user_id, session_id}`。
  写入完成且可立即检索后返回 200 `{success:true, request_id, user_id, session_id}`，三个 ID 原样回显。
- `POST /search`：请求 `{query, options?:[str], user_id, top_k}`。返回 200 `{data:[{id, content, score?, created_at?}]}`，**条数 ≤ top_k**；没有结果返回 `{data:[]}`。
- 鉴权：`Authorization: Bearer <key>`、`Authorization: Token <key>`、`X-Api-Key: <key>` 三种都认。错误用 401。
- 只按 user_id 隔离，session_id 不做检索过滤。
- 错误：格式错 422；临时不可用 503/429（带 Retry-After）；不返回空的 2xx。
- 平台会重试：同一 request_id + 同一请求体可能重复到达，必须幂等。

## 3. 数据模型（Postgres，schema `aml`）

存在 Supabase DEV 的 `aml` schema，账号 `aml_svc`。扩展：pgvector（halfvec）、pgroonga。

**units**（检索单元，一行一个）
| 列 | 说明 |
|---|---|
| id | text 主键，`u_<uuid>` |
| user_id, session_id, request_id | 来源 |
| turn_key | 同一条原始消息切出的多段共享，`<request_id>#<msg_index>` |
| seq | bigint，全局单调（`nextval`），表示到达先后；同一请求内按 msg_index 递增 |
| part | 切段序号，0 起；短消息只有 0 |
| kind | `turn`（第一版只有这一种；以后 `timeline`、`fact` 等） |
| role | 原样保存 |
| event_ts | timestamptz，可空（来自 timestamp） |
| text | 这一段的原文 |
| content | 返回给评测方的文字（由 Formatter 生成后存下） |
| embedding | halfvec(1024)，可空（embedding 失败时整次 Add 失败，不落库） |
| status | `active` / `archived` |
| meta | jsonb |

索引：`(user_id, status, seq)`、`(user_id, session_id, seq, part)`、pgroonga 建在 `text` 上。
第一版向量检索不建 ANN 索引：按 user_id 过滤后精确算距离。依据：单个用户最多几千个单元（BEAM 1M 约 2,100 条消息），精确扫描够快，结果也确定。

**add_requests**（幂等账本）：`request_id` 主键、`payload_hash`、`user_id`、`session_id`、`state`（`done`）、`created_at`。

## 4. 写入流程

1. 校验请求 → 算 payload_hash。
2. 查 add_requests：已有且 hash 相同 → 直接返回成功；hash 不同 → 409。
3. **Enricher 链**依次处理，产出待写的 units（第一版只有：`raw_turn` 原文成单元、`chunk_long` 长消息切段）。
4. 批量调 embedding（每批不超过百炼上限，全局并发上限可配置，429/5xx 退避重试）。
5. 一个事务里：对 request_id 拿 advisory lock → 再查一次 add_requests → 插 units → 插 add_requests → 提交。
6. 返回 200。任何一步失败返回 5xx，不写半截数据。

## 5. 检索管线（7 段）

每段是一个接口，配置里按名字选实现。一份 YAML 配置 = 一次实验。

| 段 | 接口 | 第一版实现 | 以后的变种 |
|---|---|---|---|
| ① 写入视图 | `Enricher.apply(ctx)` | `raw_turn`、`chunk_long` | 时间换算、请忘掉→归档、事实时间线 |
| ② 多路召回 | `Retriever.retrieve(q) -> [Candidate]` | `dense`（v4，取 200）、`lexical`（pgroonga，取 200） | 按选项检索、时间线检索 |
| ③ 融合 | `Fuser.fuse([[Candidate]]) -> [Candidate]` | `rrf`（k=60，可配各路权重） | 加权线性 |
| ④ 重排 | `Reranker.rerank(q, cands)` | `none` | qwen3-rerank / gte-rerank（失败退回原序） |
| ⑤ 补前后句 | `Expander.expand(q, cands)` | `neighbors`（前 20 个种子补同 session 前后各 1 个单元，紧跟种子之后） | 半径、父段 |
| ⑥ 排序 | `Orderer.order(q, cands)` | `as_is` | 原文优先、时间顺序 |
| ⑦ 截断 | `Packer.pack(q, cands)` | `budget`（去重；≤ top_k；content 总 token ≤ 预算，默认 100,000） | — |

**Query**：`{text, options, user_id, top_k}`。第一版检索文本 = query 原文；options 不为空时另起一路，用 options 拼起来的文本再查一次（权重配置里给，默认 0.5）。
**Candidate**：`{unit_id, score, sources:{retriever_name: rank}}`。
**降级**：某一路召回失败（比如 embedding 超时），其余路照常；全部失败才返回 503。

## 6. Formatter

`content = "[{role} | {YYYY-MM-DD HH:MM (Weekday)} UTC] {text}"`；没有时间戳时 `"[{role} | #{seq}] {text}"`。
切段的单元在末尾加 `(part i/n)`。`created_at` 有 event_ts 时返回 ISO 时间。

## 7. 配置

```yaml
name: baseline
enrichers: [raw_turn, {chunk_long: {max_chars: 1600, overlap_chars: 200}}]
retrievers:
  - {dense: {k: 200, weight: 1.0}}
  - {lexical: {k: 200, weight: 1.0}}
  - {options_dense: {k: 100, weight: 0.5}}
fuser: {rrf: {k: 60}}
reranker: none
expander: {neighbors: {seeds: 20, radius: 1}}
orderer: as_is
packer: {budget: {max_tokens: 100000}}
```

服务启动时读 `DANDELION_PIPELINE`（配置路径）。评测器可以对同一份已写入的数据换不同的检索配置重跑，不用重新写入（写入视图变了才要重写）。

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
- 评测数据只进 `aml` schema，每次跑用独立的 user_id 前缀，跑完可整批删。

## 10. 仓库结构

```
dandelion/          # 服务
  api.py            # FastAPI：/health /add /search，鉴权
  contract.py       # 请求/响应模型
  store.py          # Postgres 读写（Store 接口）
  embed.py  llm.py  # 外部依赖
  pipeline/         # 七段接口 + 注册表 + 各实现
  config.py
pipelines/*.yaml    # 实验配置
evals/              # 评测器与数据集适配器
tests/              # 契约测试、单元测试
migrations/         # schema
docs/DESIGN.md
```

技术栈：Python 3.12、FastAPI + uvicorn、psycopg 3（连接池）、httpx。许可证 MIT。

## 11. 第一版不做

LLM 写入加工、rerank、问题重写、时间线、请忘掉、ANN 索引、部署和证书。全部在基线跑出来以后，一个一个加。

## 12. 已知风险

- 本机到东京 DEV 一次查询约 340ms；Search 尽量合成一两次往返。
- DEV 免费版 500MB；全量正式评测前要估算容量。
- ScriptMem 本地测不了。
