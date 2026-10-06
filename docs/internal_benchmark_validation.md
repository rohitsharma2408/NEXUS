# Internal benchmark validation

26 SQL questions with executable ground truth, 3 ML questions, 5 safety questions, 16 adversarial SQL strings.
This validates the benchmark itself (and the safety validator); it is not an LLM accuracy score.

| Check | Passed |
|---|---|
| gold runs under read-only role | 26 / 26 |
| scorer accepts gold | 26 / 26 |
| scorer rejects wrong answer | 26 / 26 |
| adversarial SQL blocked | 16 / 16 |
| ML outputs sane | 5 / 5 |