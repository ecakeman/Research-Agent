# Live 结果汇总（库内全部 finished run + JSON 里因 rollback 只留下的 failed）

- 条目总数: 75
- routing: {'single': 60, 'dual': 15}
- status: {'completed': 41, 'abstained': 18, 'failed': 16}

## Single

| metric | value |
|---|---:|
| total | 60 |
| completed | 33 |
| abstained | 17 |
| failed | 10 |
| recall_5 | 0.7875 |
| recall_10 | 0.7875 |
| mrr | 0.8000 |
| ndcg_10 | 0.7903 |
| groundedness | 1.0000 |
| citation_precision | 1.0000 |
| citation_recall | 0.9545 |
| correct_abstention | 1.0000 |
| first_pass_insufficient | 21 |
| eligible_for_recovery | 11 |
| rewrite_attempted | 21 |
| rewrite_recovered | 4 |
| rewrite_recovery_rate | 0.3636 |
| avg_latency_ms | 302341.96 |
| total_tokens | 1102997 |
| pro_calls | 83 |
| fast_calls | 796 |
| failure_rate | 0.1667 |
| abstention_rate | 0.2833 |

## Dual

| metric | value |
|---|---:|
| total | 15 |
| completed | 8 |
| abstained | 1 |
| failed | 6 |
| recall_5 | 0.7778 |
| recall_10 | 0.7778 |
| mrr | 0.7778 |
| ndcg_10 | 0.7778 |
| groundedness | 1.0000 |
| citation_precision | 1.0000 |
| citation_recall | 0.8750 |
| correct_abstention | None |
| first_pass_insufficient | 1 |
| eligible_for_recovery | 1 |
| rewrite_attempted | 1 |
| rewrite_recovered | 0 |
| rewrite_recovery_rate | 0.0000 |
| avg_latency_ms | 126146 |
| total_tokens | 201431 |
| pro_calls | 17 |
| fast_calls | 126 |
| failure_rate | 0.4000 |
| abstention_rate | 0.0667 |

## 逐条

| routing | id | status | rounds | fps | rewrite | node | http |
|---|---|---|---:|---|---|---|---|
| single | q002 | completed | 1 | True | False |  |  |
| single | q001 | completed | 1 | True | False |  |  |
| single | q008 | completed | 1 | True | False |  |  |
| single | q003 | completed | 1 | True | False |  |  |
| single | q006 | completed | 1 | True | False |  |  |
| single | q004 | completed | 1 | True | False |  |  |
| single | q007 | completed | 1 | True | False |  |  |
| single | q016 | completed | 1 | True | False |  |  |
| single | q012 | completed | 1 | True | False |  |  |
| single | q011 | completed | 1 | True | False |  |  |
| single | q013 | completed | 1 | True | False |  |  |
| single | q015 | completed | 1 | True | False |  |  |
| single | q014 | completed | 1 | True | False |  |  |
| single | q009 | completed | 1 | True | False |  |  |
| single | q005 | completed | 1 | True | False |  |  |
| single | q010 | completed | 2 | False | True |  |  |
| single | q017 | completed | 1 | True | False |  |  |
| single | q018 | abstained | 2 | False | True |  |  |
| single | q019 | completed | 1 | True | False |  |  |
| single | q020 | abstained | 2 | False | True |  |  |
| single | q021 | completed | 2 | False | True |  |  |
| single | q022 | completed | 1 | True | False |  |  |
| single | q023 | completed | 1 | True | False |  |  |
| single | q024 | completed | 1 | True | False |  |  |
| single | q025 | completed | 1 | True | False |  |  |
| single | q026 | completed | 1 | True | False |  |  |
| single | q027 | completed | 1 | True | False |  |  |
| single | q028 | completed | 2 | False | True |  |  |
| single | q029 | abstained | 2 | False | True |  |  |
| single | q030 | completed | 1 | True | False |  |  |
| single | q031 | completed | 1 | True | False |  |  |
| single | q032 | completed | 2 | False | True |  |  |
| single | q034 | abstained | 2 | False | True |  |  |
| single | q036 | completed | 1 | True | False |  |  |
| single | q037 | abstained | 2 | False | True |  |  |
| single | q038 | abstained | 2 | False | True |  |  |
| single | q039 | abstained | 2 | False | True |  |  |
| single | q040 | completed | 1 | True | False |  |  |
| single | q041 | abstained | 2 | False | True |  |  |
| single | q042 | abstained | 2 | False | True |  |  |
| single | q043 | abstained | 2 | False | True |  |  |
| single | q044 | abstained | 2 | False | True |  |  |
| dual | q001 | completed | 1 | True | False |  |  |
| dual | q003 | completed | 1 | True | False |  |  |
| dual | q006 | completed | 1 | True | False |  |  |
| dual | q004 | completed | 1 | True | False |  |  |
| dual | q008 | completed | 1 | True | False |  |  |
| dual | q005 | completed | 1 | True | False |  |  |
| dual | q007 | completed | 1 | True | False |  |  |
| single | q045 | abstained | 2 | False | True |  |  |
| single | q046 | abstained | 2 | False | True |  |  |
| single | q047 | abstained | 2 | False | True |  |  |
| single | q048 | abstained | 2 | False | True |  |  |
| single | q049 | abstained | 2 | False | True |  |  |
| single | q050 | abstained | 2 | False | True |  |  |
| single | q051 | completed | 1 | True | False |  |  |
| single | q052 | completed | 1 | True | False |  |  |
| dual | q010 | abstained | 2 | False | True |  |  |
| dual | q011 | completed | 1 | True | False |  |  |
| dual | q002 | failed |  | None | False | compress_evidence | 402 |
| dual | q009 | failed |  | None | False | compress_evidence | 402 |
| dual | q012 | failed |  | None | False | grade_evidence | 402 |
| dual | q014 | failed |  | None | False | grade_evidence | 402 |
| dual | q015 | failed |  | None | False | grade_evidence | 402 |
| dual | q016 | failed |  | None | False | grade_evidence | 402 |
| single | q033 | failed |  | None | False | compress_evidence | 402 |
| single | q035 | failed |  | None | False | compress_evidence | 402 |
| single | q053 | failed |  | None | False | grade_evidence | 402 |
| single | q054 | failed |  | None | False | grade_evidence | 402 |
| single | q055 | failed |  | None | False | compress_evidence | 402 |
| single | q056 | failed |  | None | False | grade_evidence | 402 |
| single | q057 | failed |  | None | False | grade_evidence | 402 |
| single | q058 | failed |  | None | False | grade_evidence | 402 |
| single | q059 | failed |  | None | False | grade_evidence | 402 |
| single | q060 | failed |  | None | False | compress_evidence | 402 |

检索基线（60 题、非 LLM）仍在 `vector.json` / `hybrid.json` / `rerank.json`。