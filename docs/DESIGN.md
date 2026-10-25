# Dandelion Memory v2 设计文档（v2.1 评审收敛版）

> **关联 Issue / PR**：`moujunlin/lori-backend#234` · [`dandelion-memory`](file:///home/maomao/dandelion-memory) (`feature/aml-mvp` · PR #2)  
> **状态**：v2.1（已合入猫猫拍板项与砚首轮评审 `F1–F4, F6–F16`；`F5` 已凭官方评测源码驳回）

---

## 1. 终态一句话与解决的核心缺口

一个符合 AML 第二期文本赛道官方契约的同步记忆服务：`/add` 将非 `system` 原始对话落库并切为 2 句细粒度检索块，同时对每个批次全量调用 1 次轻量 LLM 提取**角色映射 (`subject`)、主题状态时间线 (`tag` + 新旧状态 + 改口原因)、折算后的绝对事件时间 (`event_time_start/end`) 及联动遗忘墓碑 (`forget_actions`)**；`/search` 通过**零 LLM 的四路召回（Dense + BM25 标签倍增 + 时间 Range + 选择题 Options 扩展）+ 父级还原内联 + `qwen3-rerank` 精排**，在 `top_k <= 100` 与 `100,000` token 预算内返回证据。

---

## 2. 对外契约与容错边界（解 F1, F2, F4, F5）

1. **`Message.role` 与 `system` 过滤（解 F1, 驳回 F5）**：
   - [`models.py`](file:///home/maomao/dandelion-memory/src/dandelion_memory/models.py) 中 `Message.role` 定义为 `Literal["user", "assistant", "system"]`。
   - `/add` 接收到 `role == "system"` 的消息时通过校验但不写入检索表（已核实 [`clbench_pipeline.py.txt`](file:///tmp/aml-ds/amlrepo/clbench_pipeline.py.txt#L28-L35)，平台在 `CL-bench` 答题模板顶部直接注入 `{{system_prompt}}`，不依赖检索返回；而过滤 `system` 可彻底阻断 `PersonaMem v2` 首条 3.7k 初始画像泄密）。
   - 若某次 `/add` 过滤 `system` 后有效消息列表为空，仍将 `(request_id, user_id, session_id, payload_hash)` 写入 `add_requests` 幂等账本，允许空写入正常返回 HTTP 200 `{success: true, request_id, user_id, session_id}`。
2. **幂等前置（解 F2.1）**：
   - `/add` 入口第一步计算完整请求的 `payload_hash` 并查询 `add_requests`：
     - 若 `request_id` 已存在且 `payload_hash` 一致 $\rightarrow$ 立即返回 HTTP 200（**0 次 LLM / Embedding 调用，无重放副作用**）；
     - 若 `request_id` 已存在但 `payload_hash` 或 `user_id / session_id` 不同 $\rightarrow$ 立即返回 HTTP 409。
3. **模块 B 全量 LLM 调用的强降级契约（解 F2.2）**：
   - 每次有效 `/add` 批次全量调用 1 次 LLM（单次超时 `8.0s`，最多重试 1 次，强制 `response_format={"type": "json_object"}` + 逐条容错校验）。
   - 若 LLM 超时、限流或返回非法 JSON，**严禁抛出 500 阻断写入**：自动降级为确定性正则（[`text.py`](file:///home/maomao/dandelion-memory/src/dandelion_memory/text.py) 的 `extract_forget_target` + `temporal_hints`），确保模块 A 的 `parent_messages` 与 `raw_slice` 100% 完成 Embedding 与 BM25 入库并返回 200。
4. **动态 `top_k` 与预算截断（解 F4）**：
   - 置顶状态时间线条数上限为 `max_timelines = min(5, max(1, request.top_k // 5))`（当 `request.top_k == 1` 时最多 0~1 条，优先保最高分项），合并列表严格执行 `packed[:request.top_k]` 且总 token $\le 100,000$。

---

## 3. 单栈数据模型设计（解 F7, F9, F10, F14）

采纳单栈减法：在 Colab 50GB 实例中使用 **单 SQLite-WAL 数据库 + 内存 Numpy `float32` 向量矩阵 + 内置 BM25**（同时保持 [`store.py`](file:///home/maomao/dandelion-memory/src/dandelion_memory/store.py) 的 `MemoryStore` 协议兼容 Supabase PG `aml` schema），消除 SQLite+Qdrant 跨库同步负担，所有写入与墓碑化在单事务内原子完成。

### 3.1 `add_requests`（幂等与跨批状态表）
- `request_id` (`TEXT PRIMARY KEY`), `user_id` (`TEXT NOT NULL`), `session_id` (`TEXT NOT NULL`), `payload_hash` (`TEXT NOT NULL`), `created_at` (`TEXT NOT NULL`)

### 3.2 `user_state`（用户级单调序号与跨批次遗忘确认锁，解 F10.2）
- `user_id` (`TEXT PRIMARY KEY`)
- `last_seq` (`INTEGER NOT NULL DEFAULT 0`)：分配该 `user_id` 下全局单调递增的 `seq`
- `pending_forget_confirm` (`INTEGER NOT NULL DEFAULT 0`)：若上一批次末尾是用户的显式遗忘指令，置为 `1`；下一批次首条若是 `assistant` 确认句则自动标 `tombstoned=1` 并清零。

### 3.3 `parent_messages`（原始消息表，解 F3, F6, F9）
- `msg_id` (`TEXT PRIMARY KEY`)
- `user_id` (`TEXT NOT NULL`), `session_id` (`TEXT NOT NULL`), `request_id` (`TEXT NOT NULL`)
- `seq` (`INTEGER NOT NULL`)：该 `user_id` 下的全局单调序号（`1, 2, ...`）
- `role` (`TEXT NOT NULL`)：`user` | `assistant`
- `speaker` (`TEXT NOT NULL`)：若 `content` 头部匹配 `^([A-Z][a-zA-Z0-9_ -]{1,30}):\s+` 则提取为 `speaker`，否则取 `role`（原文保留不删）
- `content` (`TEXT NOT NULL`)：原始正文（若长文档局部切片被遗忘，对应片段被掩码替换为 `[已擦除]`）
- `session_ts` (`INTEGER`)：UTC 秒级 Unix 时间戳。若 `message.timestamp`（毫秒）非空则取 `timestamp // 1000`；若为空则尝试从 `content` 头部解析日期锚点，并**在同一 `session_id` / 同一批次内向后继承给无时间戳的后续消息**（解 F9）
- `session_time_str` (`TEXT`)：格式化后的 UTC 时间串（如 `"2023-05-08 13:56 UTC"`）
- `fact_notes` (`TEXT DEFAULT ''`)：模块 B 针对该父消息提炼的 `fact_text` 汇总（供模块 D 还原父消息时内联附在末尾，解 F11）
- `tombstoned` (`INTEGER NOT NULL DEFAULT 0`)

### 3.4 `search_chunks`（统一检索单元表：含原文切片、原子事实、时间线摘要，解 F14.2, F15）
- `chunk_id` (`TEXT PRIMARY KEY`)
- `user_id` (`TEXT NOT NULL`), `session_id` (`TEXT NOT NULL`)
- `parent_msg_id` (`TEXT`)：`raw_slice` 与 `fact_item` 指向所属 `parent_messages.msg_id`（跨多条 `source_seqs` 的 `fact_item` 指向最大的那个 `seq` 对应的 `msg_id`）；`timeline_summary` 为 `NULL`
- `seq` (`INTEGER NOT NULL`)：所属或最新更新的全局序号
- `session_ts` (`INTEGER`)：所属或最新更新的会话秒级时间戳
- `kind` (`TEXT NOT NULL`)：`raw_slice` | `fact_item` | `timeline_summary`
- `subject` (`TEXT NOT NULL DEFAULT 'user'`)：角色主语（`user` / 具名 NPC / `user的朋友`）
- `tag` (`TEXT`)：二级主题标签（如 `喜好/音乐`、`全局指令/格式约束`）
- `display_text` (`TEXT NOT NULL`)：检索单元正文
- `search_text` (`TEXT NOT NULL`)：送 `text-embedding-v4` 的文本
- `bm25_tokens` (`TEXT NOT NULL`)：分词词元串（`subject` 与 `tag` 词元重复写入 2 次，实现 `TAG_BOOST=2`）
- `event_time_start` / `event_time_end` (`INTEGER`)：UTC 秒级时间戳区间（LLM 输出的 `YYYY-MM-DD` 统一转为当日 `00:00:00Z` 与 `23:59:59Z` 秒级时间戳；无特定事件时间则回退为 `session_ts`）
- `embedding` (`BLOB`)：`np.float32` 1024 维向量字节流
- `tombstoned` (`INTEGER NOT NULL DEFAULT 0`)

### 3.5 `tag_timelines`（角色与主题状态时间线表，解 F7, F8, F14.2）
- `user_id`, `subject`, `tag` (`PRIMARY KEY`)
- `latest_seq` (`INTEGER NOT NULL`), `latest_session_ts` (`INTEGER`)
- `current_state` (`TEXT NOT NULL`)
- `history_chain` (`TEXT NOT NULL`)：JSON 数组 `[{"seq": 12, "session_ts": 1684540800, "time_str": "...", "state": "...", "reason": "..."}]`，**排序规则优先按 `(session_ts, seq)` 升序（若 `session_ts` 非空），否则按 `seq` 升序**（解决 `LongMemEval` 211 题会话时间乱序问题，解 F7）
- `rendered_summary` (`TEXT NOT NULL`)：每次更新后同步 `UPSERT` 到 `search_chunks`（`kind='timeline_summary'`），直接参与向量与 BM25 检索
- `tombstoned` (`INTEGER NOT NULL DEFAULT 0`)

---

## 4. 四大模块详细规格（解 F3, F6, F8, F10–F15）

### 模块 A：预处理、时间继承与双层切片（[`text.py`](file:///home/maomao/dandelion-memory/src/dandelion_memory/text.py)）
1. 过滤 `role == "system"`；为每条 `user / assistant` 消息分配该 `user_id` 下递增的 `seq`，解析并向后继承 `session_ts` 与 `speaker`。
2. 若 `user_state.pending_forget_confirm == 1` 且本批首条为 `assistant` 消息，将其直接标 `tombstoned=1`，随后清零 `pending_forget_confirm`。
3. **切片粒度**：
   - 若 `len(content) <= 3000`（普通对话）：按中英文句末标点切句，每 **2 句（或满 300 字符）** 生成一个 `raw_slice`；
   - 若 `len(content) > 3000`（如 `CL-bench` 长文档/表格）：按段落/换行切成 **800 字符滑窗块（步长 640 字符）**，若首行疑似表头则固定前缀保留。

### 模块 B：写入侧批次记忆治理（每批 `/add` 全量调 1 次 LLM）
- **输入**：本批非 `system` 消息（带 `#seq`、`speaker`、`session_time_str`）+ 该 `user_id` 当前已有的活跃 `(subject, tag)` 列表。
- **JSON 输出契约（已修正键名与类型，解 F10）**：
  ```json
  {
    "facts": [
      {
        "source_seqs": [14, 15],
        "subject": "user",
        "is_other_person": false,
        "tag": "喜好/音乐",
        "fact_text": "User 现在的音乐偏好是独立民谣，此前曾喜欢电子音乐，因为觉得电子音乐节太吵而改变偏好",
        "prev_state": "喜欢电子音乐",
        "current_state": "喜欢独立民谣",
        "update_reason": "觉得电子音乐节太吵",
        "event_date_start": "2023-05-07",
        "event_date_end": "2023-05-07"
      }
    ],
    "forget_actions": [
      {
        "trigger_seq": 18,
        "target_existing_tags": ["喜好/音乐节"],
        "target_keywords": ["electronic music festival", "EDM", "rave"],
        "target_description": "User likes electronic music festivals",
        "batch_seqs_to_tombstone": [18, 19]
      }
    ]
  }
  ```
- **严格治理规则**：
  1. **防误杀矛盾题（解 F8.1）**：代码层校验 `trigger_seq` 对应的原话必须包含显式遗忘/删除词根（`forget / delete / erase / remove / don't remember / never mention / 忘掉 / 忘记 / 删除 / 别再记得`），否则忽略该条 `forget_action`。普通的改口、纠正、前后矛盾一律走 `facts` 追加到 `tag_timelines.history_chain`，保留新旧两版说法。
  2. **父子同步闭环墓碑化（解 F6）**：
     - 本批内的 `batch_seqs_to_tombstone`（含用户遗忘指令及紧跟的 AI 确认句；若 `trigger_seq` 是本批最后一条则置 `pending_forget_confirm=1`）直接标 `tombstoned=1`；
     - 在该用户历史库中，定位满足以下任一条件的记录：① `tag` 属于 `target_existing_tags`；② 与 `target_keywords` 词项重叠度 $\ge 0.35$（[`pipeline.py:select_for_archive`](file:///home/maomao/dandelion-memory/src/dandelion_memory/pipeline.py#L152-L165)）；③ 与 `target_description` 向量余弦 $\ge 0.58$；
     - 将命中的 `search_chunks`、`tag_timelines` 标 `tombstoned=1`，并**同步将其所属的 `parent_messages` 标 `tombstoned=1`**（若父消息 `> 3000` 字则仅将对应切片文本替换为 `[已擦除]`），彻底杜绝模块 D 还原父消息或内联邻居时复活泄密。
  3. **回填与关联（解 F11, F15）**：
     - 将 `fact` 的 `(subject, tag, event_time_start, event_time_end)` 回填给 `source_seqs` 对应的 `raw_slice`；
     - 将 `fact_text`（若 `is_other_person=true` 则强制前缀 `【非User本人，系{subject}的信息】`）追加到对应 `parent_messages.fact_notes`；
     - 若识别到全局回复风格/代码格式要求，固定归入 `tag = "全局指令/格式约束"`（解 F8.2）。

### 模块 C：Query 侧零 LLM 的四路混合检索（[`pipeline.py`](file:///home/maomao/dandelion-memory/src/dandelion_memory/pipeline.py)，解 F12, F13）
1. **Query 确定性解析**：
   - **时间 Range 提取**：基于该用户 `MAX(session_ts)` 为基准时间，用正则提取问题中的年份/年月/具体日期或相对时间词（`yesterday / last week / N days ago / N weeks ago / N months ago`），生成 `[q_time_start, q_time_end]`；
   - **选择题 `options` 权重拆分（解 F13）**：若 `request.options` 非空，计算 `vec_q = embed(query)` 与 `vec_opt = embed(query + "\n" + "\n".join(opt[:150] for opt in options))`，合成查询向量 `query_vec = normalize(0.7 * vec_q + 0.3 * vec_opt)`；BM25 中 `query` 词元权重为 `2.0`，`options` 独有实词权重为 `0.5`。
2. **三路打分 + 时间线统一召回（仅在 `tombstoned == 0` 范围）**：
   - **路 1（Dense 向量，权重 $w_1 = 1.5$）**：Numpy 余弦相似度 Top 150；
   - **路 2（BM25 稀疏，权重 $w_2 = 1.0$）**：CJK 单双字 Bigram + 英文去停用词 BM25（`k1=1.5, b=0.75`，`subject` 与 `tag` 词元已在入库时 $\times 2$ 倍增）Top 150；
   - **路 3（时间 Range 过滤通道，权重 $w_3 = 1.2$）**：仅当解析出 `[q_time_start, q_time_end]` 时激活，筛选 `event_time_start <= q_time_end AND event_time_end >= q_time_start` 的子块按余弦相似度取 Top 100（只加分，不裁剪全局候选池，解 F7）；
   - **加权 RRF 融合公式**：
     $$\text{RRF}(c) = \sum_{r \in \{1,2,3\}} \frac{w_r}{60 + \text{rank}_r(c)}, \quad \text{rrf\_norm}(c) = \frac{\text{RRF}(c)}{\max_{c'} \text{RRF}(c')}$$

### 模块 D：父级聚合（带长文阈值）+ `qwen3-rerank` + 动态打包（解 F3, F4, F11, F12）
1. **分离时间线与对话块，按粒度去重（解 F3, F14.2）**：
   - 命中的 `kind == 'timeline_summary'` 加上该用户常驻的 `tag == '全局指令/格式约束'` 归入时间线候选列表；
   - 其余子块（`raw_slice` / `fact_item`）查看其关联的 `parent_message`：
     - **普通消息（`len(parent.content) <= 3000`）**：按 `parent_msg_id` 聚合去重（取最高 RRF 分），还原完整父消息正文，内联同 `session_id` 下未墓碑化的 `#(seq-1)` 与 `#(seq+1)`（若邻居 $\le 1000$ 字），并在末尾附上 `parent.fact_notes`（`\n[提炼事实: ...]`，解 F11）；
     - **超长文档（`len(parent.content) > 3000`，如 `CL-bench`）**：**不还原整条父消息**，直接以命中的 800 字 `raw_slice` 自身为单位去重（解 F3）。
2. **`qwen3-rerank` 精排与降级（解 F12.2）**：
   - 取去重后的前 **50 条** 候选送入 DashScope `qwen3-rerank`（每条送审文本截断至最多 `2400` 字符，超时 `4.0s`）；
   - 综合得分：
     $$\text{final\_score} = 0.80 \times \text{rerank\_score} + 0.20 \times \text{rrf\_norm} + \mathbb{I}(\text{is\_latest\_in\_tag}) \times 0.05$$
     其中 `is_latest_in_tag` 按 `(session_ts, seq)` 判定（解 F7）；
   - 若 Rerank 超时或返回校验失败，自动退回按 `rrf_norm + I(is_latest_in_tag) * 0.05` 排序。
3. **动态 `top_k` 与 Token 预算打包（解 F4）**：
   - 先放入得分最高的至多 `min(5, max(1, request.top_k // 5))` 条 `timeline_summary`（当 `request.top_k >= 5` 时）；
   - 再按 `final_score` 降序依次放入去重后的消息/切片块，严格遵守累计 token $\le 100,000$ 且总返回条数 $\le \text{request.top\_k}$。
