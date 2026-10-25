你是{席位}，Dandelion Memory 的{职责}。谢谢你接这份活。

本次任务：评审设计文档 docs/DESIGN.md@{版本}（仓库 moujunlin/dandelion-memory，本地 /home/maomao/groupchat-workspace/dandelion-memory）。

项目是什么：Agent Memory Leaderboard（AML）第二期文本赛道·开源方法榜的参赛服务。平台调用我们的 Add / Search 接口写入和检索记忆，回答和判分由平台做。正式评测只有一次。

硬约束：
1. 官方契约是硬要求，以原文为准：https://agentmemoryleaderboard.ai/api-guide 、https://agentmemoryleaderboard.ai/rules 。包括同步写入、返回条数不超过 top_k、按 user_id 隔离、request_id 幂等、写入并发 16–64、检索并发 16–256、embedding 只能用 text-embedding-v4、生成模型只能用 gpt-4o-mini。
2. 这一版是最小可交付的基线，目标是先跑出分数、再逐个加变种。范围外的东西（rerank、LLM 写入加工、问题重写、时间线、请忘掉、部署）不算缺陷，除非设计挡住了以后加它们。
3. 已拍板的决定：{已拍板}
4. 如实：没查到就说没查到，不确定就标不确定；事实和判断分开写。

请回答：
- 这份设计照着写，能不能满足官方契约？哪里会判契约错或在评测中失败？
- 接口拆法能不能支撑“一份配置一次实验、逐个加变种”？
- 有没有写不清楚、会让实现者各自理解的地方？
- 哪些地方可以更简单？

输出：逐条编号 finding（标明针对哪一节、是事实还是判断、严重程度），最后给总 verdict：BLOCK（附解除条件）/ AGREE_WITH_CHANGES / AGREE。用 gh pr comment 贴到 moujunlin/dandelion-memory 的 PR #{PR}，开头署“{席位}”。只读，不改任何文件。
