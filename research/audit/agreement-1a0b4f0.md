# Grader vs checker agreement

Loaded: slice1_A=43, slice1_B=43, slice2_A=42, slice2_B=42, slice3_A=42, slice3_B=42

## 1. Inter-grader agreement (127 specs labelled by both)

- **shown**: 127/127 = 100.0% raw, Cohen's kappa 1.000
- **spoken**: 127/127 = 100.0% raw, Cohen's kappa 1.000

### Disagreements on `shown` (0)


### Disagreements on `spoken` (0)


## 2. Consensus

- shown: absent=12, correct=89, wrong=26
- spoken: absent=10, correct=91, wrong=26

## 3. Checker against consensus (disputed and n/a excluded)

### shown  (checker `check_answer`)

| checker \ graders | correct | not-correct |
|---|---|---|
| correct | 89 | 1 |
| not-correct | 0 | 37 |

agree 126/127; excluded: {}

### spoken  (checker `check_spoken`)

| checker \ graders | correct | not-correct |
|---|---|---|
| correct | 58 | 0 |
| not-correct | 4 | 26 |

agree 84/88; excluded: {'checker n/a': 39}

## 4. Cases

### shown: FALSE PASS (1)

- `E1c-work|qwen3-4b-2507|staged:work=1|dev-nested-loop-big-o|s0`
  - checker: correct | states 16 want 16
  - A: absent | 'n^2'
  - B: absent | 'n^2'

### shown: FALSE FAIL (0)


### spoken: FALSE PASS (0)


### spoken: FALSE FAIL (4)

- `E1-methods|qwen3-4b-2507|oneshot|dev-quadratic-roots|s0`
  - checker: not_found | no value beyond the request's [5.0, 6.0, 0.0, 5.0, 6.0]
  - A: correct | 'Solving gives $x = 2$ or $x = 3$. These are the roots of the equation.'
  - B: correct | 'Solving gives $x = 2$ or $x = 3$. These are the roots of the equation.'
- `E1b-methods|qwen3-4b-2507|staged:n=4,repair=1,calc=1|dev-two-dice-seven|s0`
  - checker: wrong | result 6 (cued) want 0.166667
  - A: correct | 'The probability of rolling a sum of 7 is exactly one-sixth'
  - B: correct | 'The probability of rolling a sum of 7 is exactly one-sixth'
- `E1c-work|qwen3-4b-2507|staged:n=4,repair=1,calc=1,work=1|dev-two-dice-seven|s0`
  - checker: wrong | result 6 (cued) want 0.166667
  - A: correct | 'The probability that the sum is 7 is exactly 1/6'
  - B: correct | 'The probability that the sum is 7 is exactly 1/6, as there are 6 favorable outcomes out of 36 total possible outcomes.'
- `E1c-work|qwen3-4b-2507|staged:work=1|dev-nested-loop-big-o|s0`
  - checker: wrong | result 1 (last) want 16
  - A: correct | 'So, the total inner-loop steps are 4 times 4, which is 16.'
  - B: correct | 'the total inner-loop steps are 4 times 4, which is 16'

## 5. wrong_steps / unspeakable per model/method

| grader | model | method | specs | specs w/ wrong_steps | wrong_steps | specs w/ unspeakable | unspeakable |
|---|---|---|---|---|---|---|---|
| A | gemma-4-12b | staged:n=4,repair=1,calc=1,work=1 | 5 | 1 | 1 | 0 | 0 |
| A | granite-4.2-3b-self | staged:calc=1,work=1 | 2 | 0 | 0 | 1 | 3 |
| A | granite-4.2-3b-sft-smoke | staged:calc=1,work=1 | 2 | 2 | 8 | 0 | 0 |
| A | qwen3-4b-2507 | oneshot | 20 | 2 | 5 | 20 | 80 |
| A | qwen3-4b-2507 | oneshot:free=1 | 20 | 0 | 0 | 10 | 33 |
| A | qwen3-4b-2507 | staged | 13 | 7 | 12 | 10 | 26 |
| A | qwen3-4b-2507 | staged:free=1 | 10 | 5 | 10 | 6 | 15 |
| A | qwen3-4b-2507 | staged:n=4,repair=1 | 10 | 4 | 8 | 8 | 17 |
| A | qwen3-4b-2507 | staged:n=4,repair=1,calc=1 | 13 | 4 | 9 | 10 | 27 |
| A | qwen3-4b-2507 | staged:n=4,repair=1,calc=1,plans=5 | 12 | 5 | 11 | 11 | 33 |
| A | qwen3-4b-2507 | staged:n=4,repair=1,calc=1,work=1 | 10 | 1 | 1 | 7 | 22 |
| A | qwen3-4b-2507 | staged:work=1 | 10 | 1 | 1 | 9 | 27 |
| B | gemma-4-12b | staged:n=4,repair=1,calc=1,work=1 | 5 | 0 | 0 | 0 | 0 |
| B | granite-4.2-3b-self | staged:calc=1,work=1 | 2 | 0 | 0 | 2 | 3 |
| B | granite-4.2-3b-sft-smoke | staged:calc=1,work=1 | 2 | 2 | 12 | 0 | 0 |
| B | qwen3-4b-2507 | oneshot | 20 | 2 | 4 | 20 | 104 |
| B | qwen3-4b-2507 | oneshot:free=1 | 20 | 0 | 0 | 10 | 42 |
| B | qwen3-4b-2507 | staged | 13 | 6 | 10 | 10 | 39 |
| B | qwen3-4b-2507 | staged:free=1 | 10 | 3 | 8 | 8 | 19 |
| B | qwen3-4b-2507 | staged:n=4,repair=1 | 10 | 3 | 8 | 8 | 25 |
| B | qwen3-4b-2507 | staged:n=4,repair=1,calc=1 | 13 | 3 | 5 | 10 | 40 |
| B | qwen3-4b-2507 | staged:n=4,repair=1,calc=1,plans=5 | 12 | 3 | 13 | 11 | 43 |
| B | qwen3-4b-2507 | staged:n=4,repair=1,calc=1,work=1 | 10 | 0 | 0 | 7 | 29 |
| B | qwen3-4b-2507 | staged:work=1 | 10 | 1 | 1 | 9 | 32 |
