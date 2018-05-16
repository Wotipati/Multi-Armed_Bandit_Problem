# Multi-Armed Bandit Problem

**This python scripts implements several following simple algorithms for solving the Multi-Armed Bandit Problem.**

- Greedy
- Epsilon-greedy
- UCB1


## Example
You can set some parameters in `multi-armed_bandit_problem.py`.

If there are **four slot machines** (winning rate:[0.5,0.4,0.3,0.2]) and player is allowed to pull the lever **10000 times**:


```python:multi-armed_bandit_problem.py
def main():
    # set parameter
    true_rate = np.array([0.5,0.4,0.3,0.2])  # Winning rates
    iteration = 10000                        # Total round
```

</br>

To try this script, you can run by:
```
python multi-armed_bandit_problem.py
```
(It will take some time...)

### Results
This script visualizes the learning processes in each method.


---

### References
[これからの強化学習](http://www.morikita.co.jp/books/book/3034)
