import numpy as np

# Vector of budgets => we will have to sort it
budgets = np.array([115, 140, 145])
# Here we assume to have it (=STEP 1)
value_budget = np.array([[-np.inf, 100, 110], [34, 84, 150], [30, 40, 70]])
# Get, dynamically, the number of rows and cols
value_budget_rows, value_budget_cols = value_budget.shape

n_budgets = budgets.size            # How many budgets do we have
n_campaigns = value_budget_rows     # How many campaigns do we have

print(f"Num budgets = {n_budgets}")
print(f"Num campaigns = {n_campaigns}")

# Initializations
prev_campaign = np.empty(n_budgets)
curr_campaign = np.zeros(n_budgets)
tmp_campaign = np.zeros(n_budgets)
tmp_max_sum = np.empty(n_budgets)

for i in range(0, n_campaigns):
    if i == 0:
        curr_campaign = tmp_campaign
    elif i == 1:
        curr_campaign = tmp_campaign
    else:
        tmp_campaign = value_budget[i, :]
        for j in range(0, n_budgets):
            pos_budget_leq_j = [x for x in range(len(budgets)) if budgets[x] <= budgets[j]]
            tmp_max_sum = np.append(tmp_max_sum, prev_campaign[pos_budget_leq_j])
            print(tmp_max_sum)
            print(pos_budget_leq_j)

    #tmp_campaign...
