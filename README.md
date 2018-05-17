# Multi-Armed Bandit Problem
![2](https://user-images.githubusercontent.com/26996041/40157009-a88de258-59d7-11e8-9e7c-4b9a0803ac72.png)

</br>

**This python scripts implements several following simple algorithms for solving the Multi-Armed Bandit Problem.**
- Greedy
- Epsilon-greedy
- UCB1


## Example
You can set some parameters in `multi-armed_bandit_problem.py`.

If there are **four slot machines** (winning rate:[0.5,0.4,0.3,0.2]) and player is allowed to pull the lever **5000 times**:


```python:multi-armed_bandit_problem.py
def main():
    # set parameter
    true_rate = np.array([0.5,0.4,0.3,0.2])  # Winning rates
    iteration = 5000                        # Total round
```

</br>

To try this script, you can run by:
```
python multi-armed_bandit_problem.py
```
(It will take some time...)

## Results
This script visualizes the learning processes in each method.  
<div align="center">
<img src="https://user-images.githubusercontent.com/26996041/40157059-e6662f72-59d7-11e8-8429-4581c5c279c5.png">  
</br>

<img width="607" src="https://user-images.githubusercontent.com/26996041/40157063-e9e7a9c8-59d7-11e8-928a-24fde44e86a6.png">

</div>
---

### References
[これからの強化学習](http://www.morikita.co.jp/books/book/3034)
