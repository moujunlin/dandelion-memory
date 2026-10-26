# Dandelion Memory v2.3 设计文档

> **关联 Issue / 分支**：`moujunlin/lori-backend#234` · [`dandelion-memory`](file:///home/maomao/dandelion-memory) (`feat/dandelion-v2-3`)  
> **配套比赛与派工档案**：[`docs/COMPETITION_AND_DISPATCH.md`](file:///home/maomao/dandelion-memory/docs/COMPETITION_AND_DISPATCH.md)

---

## 1. Scope、废除表与全局不变量

### 1.1 一句话 Scope
将写入协议收敛为纯 6 字段 `ops` 状态机，解耦一对多只读原话切片（`raw_slice`）与细粒度事实单元（`fact_item`），实现现存事实主表与原话的双向联动及单事务原子落库。

### 1.2 废除与替换表（严禁新旧双轨并存）

| 废除的 v2.2 旧做法 | v2.3 替换做法 | 消除的缺陷 |
| :--- | :--- | :--- |
| `_SYSTEM_PROMPT` 索要 11 个字段（含 `fact_text, prev_state, update_reason, source_tier`） | Prompt 只留纯 6 字段 `ops`（`op, target_id, subject, key, value, seq` + 可选 `date`），展示文本与层源由后端生成 | 消除弱模型 JSON 膨胀与新旧字段冲突 |
| `backfill_raw_slices_for_seq` 拿单条事实的 `(subject, tag, source_tier)` 覆写共享 `raw_slice` | `raw_slice` 保持原话中立；仅当某 `seq` 全部事实均为 `draft` 时才将该 `seq` 标为草稿 | 消除单句多第三人合并、草稿抹杀同句真事 |
| 往 `history_chain` 无脑追加，凡 `seq != latest_seq` 一律算 `superseded` | 引入同值重申幂等判重；同 `seq` 内对同槽位连改时，按节点精确墓碑化前序 `fact_item` | 消除重申事实误标过期、同轮连改残留旧事实 |
| 靠循环局部变量 `dirty_timelines` 绑向量；在预算装箱前提前改写 `emitted_seqs` | 事务末尾统一取存活 `latest_seq` 对应 `fact_item` 的向量；`pack_results` 入选时才登记 `emitted_seqs` | 消除乱序/遗忘回退后向量脱节、落选大卡吞邻居 |

### 1.3 四个全局不变量（Invariants）
1. **INV-1（现存事实状态机）**：
   - **同值幂等**：归一化 `(key, value)` 未变时视为重申，不新增覆盖节点，不将旧 `seq` 记入 `superseded_seqs`。
   - **异值覆盖**：单值槽位状态变更时，旧状态对应的 `fact_item` 置 `tombstoned = 1`，旧 `seq` 记入 `superseded_seqs`。
   - **多值并列**：`op = "add"`（或宠物/爱好/亲友等天然多值类目）且值不同时，派生 `#F2` 独立槽位；后续更新 `#F2`（含漏传 `target_id`）时按词面最高重叠度精准命中 `#F2`。
2. **INV-2（弱模型解析与删除守门）**：
   - 解析器兼容 `"ops"`、`"tag_updates"`、`"facts"`（并存时合并去重）；有效事实为空或 `schema_broken` 时，同时合并 `regex_fallback["facts"]` 与 `forget_actions`。
   - 任何删除动作（`op = "delete"` 或 `forget_actions`）必须命中原话 `has_explicit_forget_keyword`；无遗忘词时强制转为 `op = "set"` 状态更新。
3. **INV-3（单事务落盘与单一事实源向量同步）**：
   - Stage 2 一次性批量调用 `embedder.embed`；Stage 3 在单个 `with self.store.transaction():` 内完成落盘。
   - 事务收尾时，所有变动时间线的 `timeline_summary` 向量统一取自其最终存活 `latest_seq` 对应的活跃 `fact_item` 向量。
4. **INV-4（1-to-N 细粒度检索与双向联动）**：
   - **正向保送（Index-to-Raw Pull）**：粗排 Top-10 且 `tl_rrf >= 0.35` 的主表项，顺 `latest_seq` 拉取对应非草稿 `raw_slice`（`rrf_norm = max(existing, 0.85 * tl_rrf)`）。
   - **按 `(pmid, tier, subject)` 分组**：同一原话下的不同第三人（如 `Rachel`、`Bob`）拆为独立候选卡，未在 Query 提及时各自乘以 `× 0.35` 降权。
   - **槽位级过期判定**：若原话下全部事实均已过期，整卡标记 `is_superseded = True`（非历史查询乘以 `× 0.25` 降权）并加顶部提示；若既有过期事实又有存活事实（半死半活），卡片不降权，仅在卡内注明已过期槽位的最新值。

---

## 2. 数据模型（Schema v2.3）

| 表 / 视图 | 主键 | 核心字段与职责 |
| :--- | :--- | :--- |
| `add_requests` | `request_id` | `user_id, session_id, payload_hash, created_at`（请求级幂等账本） |
| `user_state` | `user_id` | `last_seq, pending_forget_confirm`（单调序号与跨批遗忘锁） |
| `parent_messages` | `msg_id` | `user_id, session_id, seq, role, speaker, content, session_ts, tombstoned`（只读原话） |
| `tag_timelines` / `living_facts` | `(user_id, subject, tag)` | `short_id` (`F1..Fn` 惰性补号), `latest_seq`, `current_state`, `superseded_seqs` (JSON), `history_chain`, `tombstoned` |
| `search_chunks` | `chunk_id` | `kind` (`raw_slice` \| `fact_item` \| `timeline_summary`), `seq, subject, tag, source_tier, display_text, search_text, embedding, tombstoned` |

---

## 3. 写入链路（`/add`）三阶段流水线

1. **Stage 1（无锁抽取）**：
   - 读取 `user_state` 与活跃主表 `get_living_facts(user_id)`（注入 `[F1] subject | key = state (#seq)`）。
   - 切分中立 `raw_slice`（短句 `<= 120` 字附带上一句前 80 字 `context_prefix`）。
   - 并发调用 `DashScopeExtractor`（纯 6 字段 `ops`），经 `sanitize_extraction_result` 完成别名合并、删除词守门与正则兜底。
2. **Stage 2（无锁批量向量化）**：
   - 整理 `pending_raw_chunks`、`prepared_facts` 与 `valid_forgets`，发起**单次 `await self.embedder.embed(...)`**。
3. **Stage 3（单事务落盘与向量同步）**：
   - 写入 `parent_messages`、`user_state` 与 `raw_slice`（仅全草稿 `seq` 标为 `user_draft_fiction`）。
   - 按 `seq` 升序执行 `upsert_tag_timeline`：完成槽位匹配、同值幂等跳过、异值旧 `fact_item` 墓碑化。
   - 按 `seq` 升序执行 `execute_forget_action`。
   - 对本批所有变动槽位，按主表最终存活的 `latest_seq` 提取对应 `fact_item` 向量写入 `timeline_summary`，登记 `add_requests` 后提交事务。

---

## 4. 检索链路（`/search`）四步流水线

1. **混合粗排**：Dense + BM25 + Time Range 加权 RRF 融合。
2. **主表保送与细粒度分组**：
   - Top-10 且 `tl_rrf >= 0.35` 的主表项保送其 `latest_seq` 非草稿 `raw_slice`。
   - 按 `(parent_msg_id, source_tier, subject)` 聚合同质候选卡，区分“全过期卡（`is_superseded=True`）”与“半死半活卡（仅标过期槽位提示，不降权）”。
3. **邻居净化与 Rerank 精排**：
   - 过滤纯草稿邻居；`assistant` 前后邻居限 `<= 240` 字，`user` 邻居限 `<= 1000` 字。
   - `qwen3-rerank` 精排后，未提及的第三人卡片 `× 0.35`，全过期卡片（非历史查询）`× 0.25`。
4. **预算装箱内去重（`pack_results`）**：
   - 遍历候选卡片，剥离已在 `emitted_seqs` 中的邻居行后检查 Token 预算；仅当卡片确定入选时，才将其输出的 `seq` 记入 `emitted_seqs`。

---

## 5. 测试矩阵（`T1..T7`）

| 编号 | 覆盖契约 | 输入场景 | 必过断言 |
| :--- | :--- | :--- | :--- |
| **T1** | 全过期降权 + **半死半活原话保护** | 1. `#1` 房贷 `$350k` -> `#2` `$400k`<br>2. `#3` 同句含「住上海 + 花生过敏」，`#4` 仅改「搬到北京」 | 1. `#1` 整卡降权并标更新提示<br>2. 搜花生过敏时 `#3` 不降权高分命中，且卡内注明「居住/城市」已于 `#4` 更新为北京。 |
| **T2** | 多槽位共存 + **漏传 `target_id` 匹配 `#F2`** + 幂等 + 同轮连改 | 1. `add` 柯基 (`F1`) 与蓝猫 (`F2`)<br>2. `op="set", target_id=None` 更新蓝猫体重<br>3. 重复输入相同蓝猫状态<br>4. 同 `seq` 先设 A 再改 B | 1. 漏传 `target_id` 精准更新 `F2`，不覆写 `F1`<br>2. 重复同值不把前序 `seq` 标进 `superseded_seqs`<br>3. 同 `seq` 连改后 A 被墓碑化，仅 B 存活。 |
| **T3** | 纯 6 字段 Prompt + **无遗忘词 `delete` 拦截** + `schema_broken` 兜底 | 1. 检查 `_SYSTEM_PROMPT`<br>2. 原话 `"sold my bike"`（无遗忘词）配 `op="delete"`<br>3. `schema_broken` 触发 `regex_fallback` | 1. Prompt 不含 `prev_state / update_reason / source_tier`<br>2. `op="delete"` 转为状态更新，不抹除记录<br>3. `regex_fallback` 的事实与遗忘同时生效。 |
| **T4** | Index-to-Raw Pull | `#5` 口语说腰疼，主表记 `exercise constraint`，搜抽象词 | `#5` 原话通过 `latest_seq=5` 保送进入返回结果。 |
| **T5** | **同句真事+草稿共存** + 助手邻居 `<=240` 字 + **装箱内去重** | 1. `#1` 同句含小说 `draft` 与真实近况<br>2. `#2` 为 500 字 AI 回复<br>3. 超预算落选大卡（含邻居 `#10`）与入选小卡 `#10` | 1. `#1` 真事不被草稿抹杀<br>2. `#2` 不被内联<br>3. 落选大卡不污染 `emitted_seqs`，小卡 `#10` 正常输出。 |
| **T6** | **同句双第三人拆分**与独立降权 | `#1` 同句提 `Rachel` 喝美式、`Bob` 喝抹茶 | `Rachel` 与 `Bob` 拆成两张卡；搜 `Bob` 时 `Bob` 排第一，`Rachel` 乘 `0.35` 降权。 |
| **T7** | 单事务原子性 + **乱序/遗忘回退向量同步** | 1. 先写 `ts=200`（状态 B）再写 `ts=100`（状态 A）<br>2. 写 #1(A) -> #2(B) 后在 #3 显式遗忘 B | 1. 乱序后主表仍为 B，时间线向量等于 B 的向量<br>2. 遗忘回退后主表回 A，时间线向量非空且等于 A 的向量。 |
