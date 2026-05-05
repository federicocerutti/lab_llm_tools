# Bia token-speed benchmark

## Baseline

| Model | Completion tokens | Seconds | Tokens/sec |
| --- | ---: | ---: | ---: |
| qwen35-4b | 62 | 4.62 | 13.43 |
| ministral3-3b | 58 | 2.53 | 22.91 |
| phi4-mini | 64 | 3.57 | 17.92 |

## Load test

| Total concurrent | Model | Mean tokens/sec | Std dev | Success | Failures |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | qwen35-4b | 14.78 | 0.00 | 1 | 0 |
| 3 | ministral3-3b | 21.94 | 0.00 | 1 | 0 |
| 3 | phi4-mini | 18.16 | 0.00 | 1 | 0 |
| 6 | qwen35-4b | 7.67 | 0.05 | 2 | 0 |
| 6 | ministral3-3b | 22.95 | 0.53 | 2 | 0 |
| 6 | phi4-mini | 18.16 | 0.25 | 2 | 0 |
| 9 | qwen35-4b | 9.89 | 3.87 | 3 | 0 |
| 9 | ministral3-3b | 19.92 | 2.93 | 3 | 0 |
| 9 | phi4-mini | 17.79 | 0.67 | 3 | 0 |
| 12 | qwen35-4b | 10.68 | 3.39 | 4 | 0 |
| 12 | ministral3-3b | 19.79 | 2.81 | 4 | 0 |
| 12 | phi4-mini | 16.76 | 1.07 | 4 | 0 |
| 15 | qwen35-4b | 7.65 | 3.77 | 5 | 0 |
| 15 | ministral3-3b | 19.10 | 1.93 | 5 | 0 |
| 15 | phi4-mini | 14.23 | 3.34 | 5 | 0 |
| 18 | qwen35-4b | 9.67 | 1.64 | 6 | 0 |
| 18 | ministral3-3b | 18.70 | 1.39 | 6 | 0 |
| 18 | phi4-mini | 15.40 | 0.24 | 6 | 0 |
| 21 | qwen35-4b | 8.73 | 2.04 | 7 | 0 |
| 21 | ministral3-3b | 10.88 | 4.53 | 7 | 0 |
| 21 | phi4-mini | 13.84 | 1.62 | 7 | 0 |
| 24 | qwen35-4b | 6.09 | 3.71 | 8 | 0 |
| 24 | ministral3-3b | 15.72 | 4.59 | 8 | 0 |
| 24 | phi4-mini | 12.73 | 1.62 | 8 | 0 |
| 27 | qwen35-4b | 7.93 | 1.82 | 9 | 0 |
| 27 | ministral3-3b | 16.09 | 1.76 | 9 | 0 |
| 27 | phi4-mini | 11.00 | 2.37 | 9 | 0 |
| 30 | qwen35-4b | 7.44 | 1.23 | 10 | 0 |
| 30 | ministral3-3b | 16.20 | 1.31 | 10 | 0 |
| 30 | phi4-mini | 10.71 | 1.39 | 10 | 0 |

Graph: `artifacts/cluster_token_speed.png`

