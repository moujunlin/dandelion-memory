# Dandelion Memory v2 设计文档（v2.2 统一流水线定稿版）

> **关联 Issue / PR**：`moujunlin/lori-backend#234` · [`dandelion-memory`](file:///home/maomao/dandelion-memory) (`feat/dandelion-v2` · PR #2)  
> **状态**：v2.2（经真题诊断、砚代码边界挑刺、Codex 对抗性评审及整体架构收敛后的统一流水线版）

---

## 1. 核心架构收敛原则（告别补丁式同步）

v2.1 及初版草案最大的复杂性根源，在于将同一条记忆同时存为 5 份可变副本（`raw_slice`、`fact_item`、可涂改的 `parent_messages.content`、可追加/删行的 `parent_messages.fact_notes`、`tag_timelines`），并在检索时过早将整条父消息与邻居糊成一团送入精排。

v2.2 确立**「单一权威检索单元 + 内存按序状态机 + 检索末段按需组装」**三条铁律：
1. **父消息只读，检索单元（`search_chunks`）为唯一开关载体**：
   - `parent_messages.content` 原文永不涂改，**废弃 `parent_messages.fact_notes` 冗余字符串维护**；
   - 是否被遗忘（`tombstoned`）、属于哪种事实层级（`source_tier`: `'user_direct'` | `'user_draft_fiction'` | `'other_person'`）全部收敛在 `search_chunks`（`raw_slice` 与 `fact_item`）及 `tag_timelines` 上。
2. **写入侧「外部慢请求前置 $\rightarrow$ 内存按 `seq` 串行重放 $\rightarrow$ 短事务原子落库」**：
   - 调 LLM 与调 Embedding API 期间**严禁持有 SQLite 写事务锁**；
   - 所有窗口并发抽取完成后，先对不可变文本（`raw_slice`、`fact_item`、遗忘目标词）一次性批量计算向量，再在内存中按 `seq = 1..N` 严格单线程重放写入与 `seq <= trigger_seq` 遗忘，最后仅对定型的 `timeline_summary` 算一次向量，并在单次毫秒级短事务中落库。
3. **检索侧「小块精排 $\rightarrow$ 门槛排座 $\rightarrow$ 最终 Top-K 现场按存活切片组装父消息」**：
   - 送 `qwen3-rerank` 时只送同类型的浓缩子切片 + 存活 `fact_item`（$\le 600$ 字），不拼邻居、不送整篇长文；
   - 选出最终 Top-K 后，才现场读取父消息下存活且同类型的 `raw_slice` 与 `fact_item` 拼出上下文（若有切片被墓碑化则在该切片位置渲染 `[已擦除]`，若混有小说切片则自动跳过）。

---

## 2. 对外契约与容错边界

1. **`Message.role` 与 `system` 过滤**：
   - [`models.py`](file:///home/maomao/dandelion-memory/src/dandelion_memory/models.py) 中 `Message.role` 为 `Literal["user", "assistant", "system"]`。
   - `/add` 忽略 `role == "system"` 的消息（防 `PersonaMem v2` 首条 3.7k 初始画像泄密）；若过滤后有效消息为空，仍记录 `add_requests` 幂等账本并返回 HTTP 200。
2. **用户级写串行化与幂等前置**：
   - 按 `user_id` 粒度加异步协程锁（`asyncio.Lock`），防止同一用户并发 `/add` 争抢 `last_seq`；
   - 入口第一步校验 `add_requests`：同 `request_id` 且同 `payload_hash` 直接返回 HTTP 200（0 次外部调用）；同 `request_id` 但负载不同返回 HTTP 409。
3. **滑窗并发 LLM 调用的强降级契约**：
   - 单窗超时 `8.0s`，最多重试 1 次，`Semaphore(4)` 并发控制；
   - 任一窗口若超时或 JSON 非法，自动降级为确定性规则（[`text.py`](file:///home/maomao/dandelion-memory/src/dandelion_memory/text.py) 的 `extract_forget_target` + `temporal_hints` + 启发式草稿/第三人识别），绝不抛出 500。

---

## 3. 单栈数据模型设计（Schema v2.2）

### 3.1 `add_requests`（幂等账本）
- `request_id` (`TEXT PRIMARY KEY`), `user_id` (`TEXT NOT NULL`), `session_id` (`TEXT NOT NULL`), `payload_hash` (`TEXT NOT NULL`), `created_at` (`TEXT NOT NULL`)

### 3.2 `user_state`（用户级单调序号与跨批遗忘确认锁）
- `user_id` (`TEXT PRIMARY KEY`)
- `last_seq` (`INTEGER NOT NULL DEFAULT 0`)
- `pending_forget_confirm` (`INTEGER NOT NULL DEFAULT 0`)

### 3.3 `parent_messages`（只读原始消息表）
- `msg_id` (`TEXT PRIMARY KEY`)
- `user_id` (`TEXT NOT NULL`), `session_id` (`TEXT NOT NULL`), `request_id` (`TEXT NOT NULL`)
- `seq` (`INTEGER NOT NULL`)：配合 `UNIQUE(user_id, seq)` 唯一索引
- `role` (`TEXT NOT NULL`)：`user` | `assistant`
- `speaker` (`TEXT NOT NULL`)
- `content` (`TEXT NOT NULL`)：原始未修改正文
- `session_ts` (`INTEGER`) / `session_time_str` (`TEXT`)
- `fact_notes` (`TEXT DEFAULT ''`)：保留列以兼容旧表结构，v2.2 不再依赖其作为可变状态源（检索展示时直接由存活的 `fact_item` 动态汇总）
- `tombstoned` (`INTEGER NOT NULL DEFAULT 0`)：仅当整条消息的所有子切片均被墓碑化（或本身为遗忘指令/AI 确认句）时置为 `1`

### 3.4 `search_chunks`（统一权威检索单元表）
- `chunk_id` (`TEXT PRIMARY KEY`)：`raw_slice` 采用 `{msg_id}:raw:{slice_idx:04d}` 保序编码
- `user_id` (`TEXT NOT NULL`), `session_id` (`TEXT NOT NULL`), `parent_msg_id` (`TEXT`)
- `seq` (`INTEGER NOT NULL`), `session_ts` (`INTEGER`)
- `kind` (`TEXT NOT NULL`)：`raw_slice` | `fact_item` | `timeline_summary`
- `subject` (`TEXT NOT NULL DEFAULT 'user'`)
- `source_tier` (`TEXT NOT NULL DEFAULT 'user_direct'`) **[v2.2 新增]**：
  - `'user_direct'`：用户直接陈述的现实事实、经历、状态或偏好；
  - `'user_draft_fiction'`：用户提交的小说/故事/报告草稿，或 AI 顺着草稿润色的回复；
  - `'other_person'`：归属于朋友、家人、同事等第三人的事实。
- `tag` (`TEXT`)：二级主题标签
- `context_prefix` (`TEXT DEFAULT ''`) **[v2.2 新增]**：轻量语境前缀（如 `[#seq | speaker | time | tag]`；若 `source_tier == 'user_draft_fiction'` 且有 `style_note` 则附加 `[文稿折射风格: ...]`）
- `display_text` (`TEXT NOT NULL`)：检索单元正文
- `search_text` (`TEXT NOT NULL`)：送 `text-embedding-v4` 及 `qwen3-rerank` 的完整检索文本
- `bm25_tokens` (`TEXT NOT NULL`)：含英文词干还原（Stemming）后的分词词元串
- `event_time_start` / `event_time_end` (`INTEGER`)
- `embedding` (`BLOB`)：`np.float32` 1024 维向量（内存 `_vector_cache` 在 `upsert_chunk` 时强制覆盖更新，并在 `tombstoned=1` 时同步剔除缓存项）
- `tombstoned` (`INTEGER NOT NULL DEFAULT 0`)

### 3.5 `tag_timelines`（角色与主题状态时间线表）
- `user_id`, `subject`, `tag` (`PRIMARY KEY`)
- `latest_seq` (`INTEGER NOT NULL`), `latest_session_ts` (`INTEGER`)
- `current_state` (`TEXT NOT NULL`)
- `history_chain` (`TEXT NOT NULL`)：JSON 数组 `[{"seq": 12, "fact_chunk_id": "...", "session_ts": ..., "time_str": "...", "state": "...", "reason": "..."}]`，按 `(session_ts, seq)` 升序排列
- `rendered_summary` (`TEXT NOT NULL`)
- `tombstoned` (`INTEGER NOT NULL DEFAULT 0`)

---

## 4. 写入链路（`/add`）三阶段单向流水线

### 阶段 1：预处理、切分与并发滑窗抽取（无数据库写锁）
1. **读取基准状态**：查询 `add_requests` 幂等状态、`user_state(last_seq, pending_forget_confirm)` 及当前活跃 `(subject, tag)` 列表。
2. **切分 `raw_slice` 与英文词干归一（[`text.py`](file:///home/maomao/dandelion-memory/src/dandelion_memory/text.py)）**：
   - 过滤 `role == "system"`，为有效消息分配 `seq = last_seq + 1 .. last_seq + M`；
   - **顺带更新独立成片**：普通消息（$\le 3000$ 字）按句切分（每 2 句或满 300 字一片）；若分句内出现顺带更新引导词（`by the way`、`btw`、`oh also`、`incidentally`、`speaking of which`、`on another note`、`顺便说`、`对了`、`另外提一嘴`），强制在该分句前断开独立成片；
   - **英文轻量词干还原**：`tokenize_bm25` 对英文词元同时输出原词与词干（`biking -> bike`、`baking -> bake`、`running -> run`、`-ies -> -y`、`-ed`、`-es/-s`）。
3. **12 轮滑窗并发抽取**：
   - 若有效消息数 $\le 16$，单次调用 `qwen-flash`；若 $> 16$，按 **12 条可写消息一窗（$S_{\text{win}}$）+ 前序 2 条只读上下文（$S_{\text{ro}}$）** 切分，`Semaphore(4)` 并发调用 `qwen-flash`；
   - **只读上下文硬隔离**：`sanitize_extraction_result` 仅接受 $S_{\text{win}}$ 内的序号，`seqs = [s for s in raw_seqs if s in S_win]` 为空时直接丢弃该 `fact`（严禁回退绑到窗口末条）；`forget_action` 要求 `trigger_seq in S_win` 且原文含显式遗忘词根。

### 阶段 2：CPU 预对齐 + 不可变文本批量算向量 + 内存按 `seq` 串行状态机重放
1. **CPU 预对齐与不可变文本一次性 Batch Embedding**：
   - 先在内存中完成零开销的**切片级 `source_tier` 对齐**（含 `by the way` 等顺带更新词的 `raw_slice` 强制锁为 `'user_direct'`；其余按词干重叠度对齐；纯草稿轮次后的 Assistant 润色回复连坐为 `'user_draft_fiction'`）及 **`其他/*` 字面包含/二元重叠合流（$\ge 0.75$）**，生成最终的 `context_prefix`、`search_text` 与 `bm25_tokens`；
   - 将本批所有 `raw_slice`、候选 `fact_item` 以及 `forget_action.target_description` 合并为一次批量 `embedder.embed()` 请求，结果存入内存映射，重放中途**0 次网络请求**。
2. **内存按 `seq` 从小到大单线程串行重放**：
   - 加载该用户已入库的活跃 `search_chunks` 与 `tag_timelines` 到内存工作集；
   - 维护 `pending_confirm_seq: int | None`（若初始 `pending_forget_confirm == 1` 且首条为 `assistant`，则 `pending_confirm_seq = first_seq`）；
   - 按 `seq` 升序遍历本批每条消息：
     - **A. 检查 AI 遗忘确认句**：若 `seq == pending_confirm_seq` 且 `role == "assistant"`，将该消息及其下属 `raw_slice` 直接标为 `tombstoned=1`，清零 `pending_confirm_seq`，跳过后续处理；
     - **B. 激活当前 `seq` 的 `raw_slice` 与 `fact_item`**：
       - **固定 18 类词表 + `其他/*` 向量余弦兜底合流**：若 `fact.tag` 为 `其他/<名词>` 且字面未命中，直接用刚算好的 `fact_item` 向量与工作集中同 `subject` 的已有 `其他/*` 向量算余弦（$> 0.85$ 则合流到已有 `tag` 并刷新 `bm25_tokens`）；
       - 将当前 `seq` 的 `raw_slice`、`fact_item` 加入内存存活集；**仅当 `fact.source_tier != 'user_draft_fiction'` 时**，将其追加到内存 `tag_timelines[subject, tag].history_chain`（**严禁 `user_draft_fiction` 进入 `tag_timelines`**，防草稿绕过门槛置顶）。
     - **C. 执行当前 `seq` 触发的遗忘指令（`trigger_seq == seq`）**：
       - **严格因果边界**：仅在内存存活集中筛选 **`chunk.seq <= trigger_seq`** 的记录；
       - 匹配条件：① `tag in target_existing_tags`；② 与 `target_keywords` 词干重叠命中（**短分母保护**：若目标词干数 $\le 2$ 则要求全部命中；若 $\ge 3$ 则要求至少命中 2 个不同词干且重叠率 $\ge 0.35$）；③ 与 `target_description` 向量余弦 $\ge 0.58$；
       - 将命中的 `raw_slice` 与 `fact_item` 标为 `tombstoned=1`；
       - 从受影响的 `tag_timelines.history_chain` 中移除 `seq <= trigger_seq` 的命中节点（若链变空则标 `tombstoned=1`，否则用剩余最新节点重算 `current_state` 与 `rendered_summary`）；
       - 将 `#trigger_seq` 自身标 `tombstoned=1`，并设 `pending_confirm_seq = trigger_seq + 1`（若 `trigger_seq` 是整批末条消息，则记 `final_pending_forget_confirm = 1`）。
3. **最终定型时间线 Embedding**：
   - 重放结束后，仅收集本批发生新增或被遗忘修改且 `tombstoned == 0` 的 `tag_timelines`，对其最终定型的 `rendered_summary` 批量调用 1 次 `embedder.embed()`。

### 阶段 3：单次毫秒级短事务原子落库
- 开启 `with self.store.transaction():`，一次性写入 `parent_messages`、`search_chunks`、`tag_timelines`、`user_state` 与 `add_requests`，并同步刷新内存 `_vector_cache`。

---

## 5. 检索链路（`/search`）四步流水线

### 步骤 1：四路混合粗排（仅限 `tombstoned == 0`）
- **路 1（Dense 向量，$w_1 = 1.5$）**：Query 向量（若含选择题 `options` 则按 `0.7 * vec_q + 0.3 * vec_opt` 合成）余弦 Top 150；
- **路 2（词干归一 BM25，$w_2 = 1.0$）**：含英文词干还原的 BM25 Top 150；
- **路 3（时间 Range 通道，$w_3 = 1.2$）**：时间区间交集 Top 100；
- 加权 RRF 融合得出每个存活子块的 `rrf_norm`。

### 步骤 2：按 `(parent_msg_id, source_tier)` 同质分组与小块送审 Rerank（Small-to-Big）
1. **同质证据组聚合（防止小说与真事串味）**：
   - 非时间线子块按 `(parent_msg_id, source_tier)` 聚合为**同质证据组**（组唯一 ID 设为 `pmid if source_tier == "user_direct" else f"{pmid}:{source_tier}"`；超长文档 `> 3000` 字按 `(chunk_id, source_tier)` 独立成组）；
   - 这样即使同一条父消息前半段是小说（`user_draft_fiction`）、后半段 `by the way` 是真事（`user_direct`），也会被拆成两个互不干扰的候选组，绝不会让小说借真事的壳混进 `D_direct`。
2. **浓缩文本送审 `qwen3-rerank`**：
   - 每个同质证据组的送审文本（`rerank_input`）仅取**该组内得分最高的 1~2 个 `raw_slice.search_text` + 该父消息下同 `source_tier` 且 `tombstoned == 0` 的 `fact_item.display_text`**（总长 $\le 600$ 字，不拼上下游邻居，不送长父消息全文）；
   - 取前 50 个候选组（含 `timeline_summary`）送入 `qwen3-rerank`，计算综合分：
     $$\text{base\_score} = 0.80 \times \text{rerank\_score} + 0.20 \times \text{rrf\_norm} + \mathbb{I}(\text{is\_latest\_in\_tag}) \times 0.05$$
   - 若 `source_tier == 'other_person'` 且 Query 未提及该第三人称呼，则 $\text{final\_score} = \text{base\_score} \times 0.35$；否则 $\text{final\_score} = \text{base\_score}$。

### 步骤 3：相关性门控排座与 `DRAFT_FALLBACK_THRESHOLD` 草稿垫底回填
1. **时间线相关性门控（防无关时间线霸榜）**：
   - 仅当 `timeline_summary` 满足 `is_global_constraint == True` 或精排后 `final_score >= 0.40`（无 Rerank 降级时 `rrf_norm >= 0.55`）时，才允许进入置顶队列 `pinned_tl`（至多 `max_tl = min(5, max(1, request.top_k // 5))` 条）；未过线的时间线不予置顶。
2. **直接事实与草稿分桶排座（可调参数 `DRAFT_FALLBACK_THRESHOLD = 0.55`）**：
   - 将剩余候选分为**真实直接队列 `D_direct`**（`source_tier != 'user_draft_fiction'`，按 `final_score` 降序）与**及格草稿队列 `D_draft`**（`source_tier == 'user_draft_fiction'` 且 `final_score >= 0.30`，按 `final_score` 降序）；
   - 设剩余槽位数 `rem = request.top_k - len(pinned_tl)`，统计 `D_direct` 中 `final_score >= DRAFT_FALLBACK_THRESHOLD` 的条数 `n_high`：
     - 若 `n_high >= rem`（高置信真实记忆已够填满）：全部取 `D_direct[:rem]`；
     - 若 `n_high < rem` 且 `D_draft` 非空：先保留 `direct_keep = max(n_high, rem - 2)` 条 `selected_direct = D_direct[:direct_keep]`，再取 `selected_draft = D_draft[: min(2, rem - len(selected_direct))]` 放入底部；**若放入草稿后总条数仍不满 `rem`，用剩余的 `D_direct[direct_keep:]` 继续回填补满**。

### 步骤 4：最终入选 Top-K 的按需父消息组装
- 仅对最终入选 `request.top_k` 的对话证据组 `(parent_msg_id, tier)` 执行上下文展开：
  1. **组装父消息正文**：查询该 `parent_msg_id` 下所有 `kind == 'raw_slice'`（按 `chunk_id` 序号升序）：
     - 若所有子切片均满足 `tombstoned == 0` 且 `source_tier == tier`，直接使用 `parent.content` 原文；
     - 若存在被遗忘（`tombstoned == 1`）或异类 `source_tier` 的子切片，则按顺序拼接：对 `tombstoned == 1` 的切片输出 `[已擦除]`，对 `source_tier != tier` 的切片直接跳过，对匹配的存活切片输出其 `display_text`；
  2. **组装底部 `[提炼事实]`**：直接读取该 `parent_msg_id` 下所有 `kind == 'fact_item' AND tombstoned == 0 AND source_tier == tier` 的 `display_text` 拼接在末尾；
  3. **内联上下游邻居**：仅当当前组 `tier == 'user_direct'` 时，内联同 `session_id` 下未墓碑化且非纯草稿的 `#(seq-1)` 与 `#(seq+1)`（若该邻居自身有局部墓碑切片，同样按存活切片渲染）。
