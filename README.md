## Acknowledgement
We borrow some code from PatchCore (https://github.com/amazon-science/patchcore-inspection) (Apache-2.0 License)
We borrow some code from TorchMetrics (https://github.com/Lightning-AI/metrics) (Apache-2.0 License)

## Installation and reproduction steps for main results
1. Clone this repository
2. Install the required packages from requirements.txt
3. Download and extract the MVTec dataset (https://www.mvtec.com/company/research/datasets/mvtec-ad)
4. Use the SQLite database file (./studies.db) with the best trials for the main experiments in the paper (we removed all non pareto optimal trials so loading is faster) or create your own database (we recommend PostgreSQL for distributed searching).
5. (Optional) Run your own search with the following command by replacing the parameters $1-8:
```python search.py --accelerator auto --study_name $1 --n_trials $2 --k $3 --seed $4 --category $5 --test_set_search $6 --dataset_dir $7 --db_url $8```
For example:
```python search.py --accelerator auto --study_name 2023_03_11_13_00_00_n2000_k1_s0_carpet_False --n_trials 2000 --k 1 --seed 0 --category carpet --test_set_search False --dataset_dir "../MVTec" --db_url sqlite:///studies.db```
7. Execute cell 1 of results.ipynb to load the pareto optimal trials.
8. Use cell 2 to generate the results for table 2 of the AutoPatch paper. Use k=1,2,4 with search_type="few" for the corresponding columns in the table, and k=4 with search_type="full" for search on test column in the paper.
You need to put the right dataset directory in cell 2 of results.ipynb and the code will rerun the evaluation code (which means training and testing the models corresponding to the best-performing trials, this should take approximately 15 minutes for one column of table 2 of the AutoPatch paper)
Alternatively, one can take the raw output from the second cell of results.ipynb from resultsk1.txt, resultsk2.txt, resultsk4.txt and resultsk4test.txt for the results in table 2 of the AutoPatch paper.
9. Use cell 3 to generate figure 2 of the AutoPatch paper. This code does not need to rerun evaluation and should execute relatively quickly.
Alternatively, one can take figure2.png for the raw output of this cell.
9. Use our forked repository of PatchCore (https://anonymous.4open.science/r/patchcore-fork) to generate the PatchCore baseline results in table 2 of the AutoPatch paper.
Alternatively, one can take the raw output from the provided command in the README in the fork from results.txt in the fork.