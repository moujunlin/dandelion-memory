# Dandelion Memory 比赛数据档案与席位派工单

> 派工给 **霜（测试）**、**铸（实现）**、**砚（评审）**、**烬（Codex 对抗）** 时，各席须先读本文件与 [`docs/DESIGN.md`](file:///home/maomao/dandelion-memory/docs/DESIGN.md)。

---

## 一、比赛评测契约与线上弱模型约束

| 维度 | 线上硬约束 | 系统应对规则 |
| :--- | :--- | :--- |
| **接口与延迟** | `/add < 10s`（并发 16..64），`/search < 5s`（并发 16..256，默认 `top_k=5`） | Stage 1 滑窗抽取保持 `Semaphore(4)` 异步并发，禁改串行；Stage 2 单次批量 Embedding；Stage 3 单事务落库。 |
| **抽取模型 `qwen3.5-flash`** | 1. 字段超 6 个易 JSON 截断或顶层键漂移（`tag_updates`/`facts`）<br>2. 更新已有状态时常漏传 `target_id`<br>3. 跨会话重申同一偏好时重复输出 `op="set"`<br>4. 状态变更（如“卖了旧车”）易幻觉输出 `op="delete"` | 1. Prompt 只留纯 6 字段 `ops`；解析器兼容别名，失败时完整合并 `regex_fallback`<br>2. 漏传 `target_id` 时走同 `subject` 词面重叠退化匹配<br>3. 同值重申幂等，不把旧 `seq` 标为 `superseded`<br>4. 删除动作必过原话 `has_explicit_forget_keyword` 校验，无遗忘词一律转 `op="set"`。 |
| **向量与精排** | `text-embedding-v4` (`1024d`) + `qwen3-rerank` | 送 Rerank 文本限制 `<= 600` 字同质切片，Query 剥离 `options` 选项块。 |

---

## 二、七大评测数据集特征与测试必测陷阱

本地数据集说明见 `/tmp/aml-ds/readme/`，统计与采样脚本见 `/tmp/aml-ds/scripts/`。

| 数据集 | 数据形态 | 核心陷阱 | 系统与测试硬要求 |
| :--- | :--- | :--- | :--- |
| **`LongMemEval`**<br>(`s` / `m` / `oracle`) | 每题 40~500 个历史会话，60k~1.5M 字符 | 1. 单值状态跨期更新（`$350k -> $400k`）<br>2. 天然多值并列（多只宠物/多辆车）<br>3. `haystack_dates` 存在乱序时间戳<br>4. Assistant 回复常含 `>1000` 字枚举清单 | 1. 旧 `fact_item` 墓碑化，旧原话降权并标注最新值<br>2. 同类不同值派生 `#F2` 并列槽位，不覆盖 `#F1`<br>3. 乱序写入后时间线向量必须同步绑定最终存活的 `latest_seq`<br>4. Assistant 前后邻居内联统一限制 `<= 240` 字。 |
| **`PersonaMem`**<br>(`v1` & `v2`) | 多会话人物画像演进，含四选一 `options` | 1. 首条 `system` 消息含 `3.7k` 字画像泄密<br>2. 单条消息内混杂「小说草稿 + 真实近况」<br>3. 同一句话含两条事实，后续仅一条过期 | 1. `/add` 忽略 `role == "system"`<br>2. 原话 `raw_slice` 不被单条草稿覆写；仅全草稿消息才禁做邻居<br>3. 半死半活原话不作整卡降权，仅在卡内标注已过期槽位。 |
| **`LoCoMo`**<br>(`& Refined`) | 19~32 会话、300~600 轮双人长对话，含大量亲友 | 1. 单条消息同时提两个第三人（`Rachel` 和 `Bob`）<br>2. 原话用口语（腰疼），问题搜抽象概念（运动限制） | 1. 检索按 `(pmid, tier, subject)` 分组，`Rachel` 与 `Bob` 独立成卡并分别判 `× 0.35` 降权<br>2. 主表命中 Top-10 且 `tl_rrf >= 0.35` 时，顺 `latest_seq` 保送原话。 |
| **`BEAM` / `CL-Bench` / `ScriptMem`** | 长程指令遵循、格式约束与显式遗忘 | 1. 早期设定的全局格式规则需跨会话生效<br>2. 显式要求忘掉某隐私 vs 普通状态取消 | 1. `rule`（`全局指令/格式约束`）置顶返回<br>2. 仅原话含显式遗忘词才墓碑抹除；遗忘回退后同步刷新时间线向量。 |

> **本机 3.4 GiB 内存红线**：严禁对 `>= 30 MiB` 文件调用 `json.load()`、`.read()` 或 `list(csv.DictReader)`。抽样大 JSON 必须用 `from stream_stats_lib import items` 流式迭代。

---

## 三、四大架构铁律

1. **单轨极简协议**：`_SYSTEM_PROMPT` 只留纯 6 字段 `ops`（`op, target_id, subject, key, value, seq` + 可选 `date`），删除所有 v2.2 展示/原因字段。
2. **一对多（1-to-N）容器解耦**：一条原话（`seq`）可对应多条事实；严禁把单条 `fact_item` 的 `subject / tag / source_tier` 反向覆写回共享的 `raw_slice`。
3. **单一事实源派生**：时间线向量在 Stage 3 事务末尾直接取存活 `latest_seq` 对应 `fact_item` 的向量；跨卡邻居去重（`emitted_seqs`）在 `pack_results` 确定入选时才登记。
4. **破坏性动作确定性守门**：一切删除动作必过 `has_explicit_forget_keyword`；`schema_broken` 或空结果必同时合并 `regex_fallback["facts"]` 与 `forget_actions`。

---

## 四、席位派工单模板

> AGY 下总控（Coordinator）与 **砚（评审席）** 用 4a（`Model="inherit"`），其余执行席（**霜**、**铸**、**寻** 等）一律传 `Model="flash"`；对抗终审 **烬** 使用 Codex `gpt-6-astra`。

### 4.1 霜（测试席 · `Model="flash"`）
```markdown
麻烦你作为「霜 · Dandelion Memory 测试工程师」编写/更新红灯测试集。
1. 先读 `docs/COMPETITION_AND_DISPATCH.md` 与 `docs/DESIGN.md`。
2. 只改 `{测试文件}`，禁改 `src/`。
3. 除基础矩阵外，必须覆盖以下非理想输入与退化分支：
   - 退化分支 100% 触达：`op="set"` 更新 `#F2` 但 `target_id=None`；同时含 `"ops"` 与 `"facts"`；`schema_broken` 时 `regex_fallback` 事实与遗忘同时生效。
   - 一对多混合原话：同句含两个第三人（`Rachel` + `Bob`）；同句含「草稿 + 真事」；同句含「一条已过期事实 + 一条仍存活事实」。
   - 状态与向量一致性：同值重复声明幂等；同 `seq` 内对同槽位先设 A 再改 B；乱序时间戳写入及显式遗忘回退后，`timeline_summary` 向量与存活 `latest_seq` 的 `fact_item` 向量完全一致；无遗忘词的 `op="delete"` 被拦截转为状态更新。
完成后跑 pytest 汇报结果。辛苦了，谢谢！
```

### 4.2 铸（实现席 · `Model="flash"`）
```markdown
麻烦你作为「铸 · Dandelion Memory 实现工程师」按设计与测试完成代码实现。
1. 先读 `docs/COMPETITION_AND_DISPATCH.md` 与 `docs/DESIGN.md`。
2. 只改 `src/dandelion_memory/`，禁改 `tests/`。
3. 硬要求：按 `DESIGN.md` 第 1.2 节「废除清单」删净旧字段与旧单值覆写逻辑，不堆新旧双轨分支；自检所有循环的 `i=0, i=1` 变量更新；跑通全部测试与 `ruff check`。辛苦了，谢谢！
```

### 4.3 砚（评审席 · `Model="inherit"` 4a）
```markdown
麻烦你作为「砚 · Dandelion Memory 评审员」对当前产物做只读评审。
1. 先读 `docs/COMPETITION_AND_DISPATCH.md` 与 `docs/DESIGN.md`，评审范围：`{被审文件}`。
2. 必查三项：① 被替代的旧字段/旧覆写逻辑是否删净（`REMOVE/SIMPLIFY`）；② 代码中所有 `else/fallback` 退化分支是否有测试实际走通；③ 循环与一对多（1-to-N）混合消息场景逐轮变量推演是否正确。
输出分级 Finding（`BLOCK / DECISION / ADVISE`，附文件行号）与总 Verdict。辛苦了，谢谢！
```

### 4.4 烬（Codex `gpt-6-astra` 对抗评审席）
```markdown
请先读 `docs/COMPETITION_AND_DISPATCH.md` 与 `docs/DESIGN.md`，对 `/home/maomao/dandelion-memory` 当前分支做只读对抗性评审（禁改文件）。重点从 7 大数据集的 1-to-N 混合消息、弱模型 `qwen3.5-flash` 劣化输出、乱序/遗忘向量同步及测试未覆盖分支出发，给出分级 Finding（附精确行号）与最小修复建议。谢谢！
```
