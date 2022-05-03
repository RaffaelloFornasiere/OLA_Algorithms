import numpy as np


# Vector of budgets => we will have to sort it
budgets = np.array([0, 10, 20, 30, 40, 50, 60, 70])
# Here we assume to have it (=STEP 1)
value_budget = np.array([[-np.inf, 90, 100, 105, 110, -np.inf, -np.inf, -np.inf],
                         [0, 82, 90, 92, -np.inf, -np.inf, -np.inf, -np.inf],
                         [0, 80, 83, 85, 86, -np.inf, -np.inf, -np.inf],
                         [-np.inf, 90, 110, 115, 118, 120, -np.inf, -np.inf],
                         [-np.inf, 111, 130, 138, 142, 148, 155, -np.inf]])
# Get, dynamically, the number of rows and cols
value_budget_rows, value_budget_cols = value_budget.shape

n_budgets = budgets.size            # How many budgets do we have
n_campaigns = value_budget_rows     # How many campaigns do we have

print(f"Num budgets = {n_budgets}")
print(f"Num campaigns = {n_campaigns}")
print("--------------------------------")

# Initializations
prev_campaign = np.empty(n_budgets)
tmp_campaign = np.zeros(n_budgets)

for i in range(0, n_campaigns):
    curr_campaign = np.zeros(n_budgets)
    if i == 0:
        curr_campaign = tmp_campaign
    elif i == 1:
        curr_campaign = value_budget[i-1, :]
    else:
        tmp_campaign = value_budget[i-1, :]
        #print(f"Tmp C{i}: {tmp_campaign}\n-------")
        #print(f"Prev campaign (C{i-1}): {prev_campaign}\n---------")
        for j in range(0, n_budgets):
            tmp_max_sum = np.empty(j+1)
            pos_budget_prev_campaign = [x for x in range(len(budgets)) if budgets[x] <= budgets[j]]
            #print(f"Budget indexes = {pos_budget_prev_campaign}\n---------")
            if j == 0:
                tmp_max_sum[j] = tmp_campaign[np.max(pos_budget_prev_campaign)] + prev_campaign[0]
            else:
                for k in range(0, j+1):
                    pos_budget_tmp_campaign = np.max(pos_budget_prev_campaign) - k
                    #print(f"i={i}; j={j}; k={k}")
                    #print(f"PREV CAMPAIGN INSIDE = {prev_campaign}")
                    #print(f"tmp_campaign[pos_budget_prev_campaign]: {tmp_campaign[pos_budget_tmp_campaign]} + prev_campaign[k]: {prev_campaign[k]}")
                    tmp_max_sum[k] = tmp_campaign[pos_budget_tmp_campaign] + prev_campaign[k]
            #print(f"Campaign c_{i} Max Vector: {tmp_max_sum}")
            curr_campaign[j] = np.max(tmp_max_sum)
            #print(f"\n----------\ncurr_campaign[{j}]={curr_campaign[j]}\n*********")
        #print(f"prev_campaign[k] = {tmp_campaign[pos_budget_prev_campaign]}")
        #print(f"pos budget prev campaign: {pos_budget_prev_campaign}")
            #tmp_max_sum = np.append(tmp_max_sum, prev_campaign[pos_budget_leq_j])
            #print(tmp_max_sum)
            #print(pos_budget_leq_j)

    prev_campaign = curr_campaign

    print(f"Campaign c_{i}: {curr_campaign}")

curr_campaign_corr = curr_campaign - 100

print(f"\n\nSubtracting Budget: {curr_campaign_corr}")
idx_best_budget = max( (v, i) for i, v in enumerate(curr_campaign_corr) )[1]

print(f"B* (Best Budget) = {budgets[idx_best_budget]}")
