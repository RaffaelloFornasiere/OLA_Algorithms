import numpy
import numpy as np
from scipy.optimize import linear_sum_assignment
from munkres import Munkres


# remove offset from rows
def step1(m: numpy.ndarray):
    for i in range(m.shape[0]):
        m[i, :] = m[i, :] - np.min(m[i, :])


# remove offset from cols
def step2(m: numpy.ndarray):
    for i in range(m.shape[1]):
        m[:, i] = m[:, i] - np.min(m[:, i])


def step3(m: numpy.ndarray):
    dim = m.shape[0]
    # indexes of assigned rows
    rows_assigned = np.array([], dtype=int)
    # assignment matrix
    assignments = np.zeros(m.shape, dtype=int)

    # assign trivial elements (1 iif is the only one in that row and in that col)
    for i in range(0, dim):
        for j in range(0, dim):
            if m[i, j] == 0 and np.sum(assignments[:, i]) == 0 and np.sum(assignments[i, :]) == 0:
                assignments[i, j] = 1
                rows_assigned = np.append(rows_assigned, i)

    # the marked rows as the non-assigned rows
    rows = np.linspace(0, dim - 1, dim)
    marked_rows = np.setdiff1d(rows, rows_assigned).astype(int)
    new_marked_rows = marked_rows.copy()
    marked_cols = np.array([])

    # for each marked row I mark all (unmarked) corresponding cols, and for all new marked cols
    # I mark all
    while len(new_marked_rows) > 0:
        new_marked_cols = np.array([], dtype=int)
        for nr in new_marked_rows:
            zero_cols = np.argwhere(m[nr, :] == 0).reshape(-1)
            new_marked_cols = np.append(new_marked_cols, np.setdiff1d(zero_cols, marked_cols))
        marked_cols = np.append(marked_cols, new_marked_cols)
        new_marked_rows = np.array([], dtype=int)

        for nc in new_marked_cols:
            new_marked_rows = np.append(new_marked_rows, np.argwhere(assignments[:, nc] == 1).reshape(-1))
        marked_rows = np.unique(np.append(marked_rows, new_marked_rows))

    return np.setdiff1d(rows, marked_rows).astype(int), np.unique(marked_cols)


def step5(m: numpy.ndarray, covered_rows: np.ndarray, covered_cols: np.ndarray):
    uncovered_rows = np.setdiff1d(np.linspace(0, m.shape[0] - 1, m.shape[0]), covered_rows).astype(int)
    uncovered_cols = np.setdiff1d(np.linspace(0, m.shape[1] - 1, m.shape[1]), covered_cols).astype(int)
    min_val = np.max(m)
    for i in uncovered_rows.astype(int):
        for j in uncovered_cols.astype(int):
            if m[i, j] < min_val:
                min_val = m[i, j]

    for i in uncovered_rows.astype(int):
        m[i, :] -= min_val

    for j in covered_cols.astype(int):
        m[:, j] += min_val

    return m


def find_rows_single_zero(m: numpy.ndarray):
    for i in range(0, m.shape[0]):
        if np.sum(m[i, :] == 0) == 1:
            j = np.argwhere(m[i, :] == 0).reshape(-1)[0]
            return i, j
    return False


def find_cols_single_zero(m: numpy.ndarray):
    for i in range(0, m.shape[1]):
        if np.sum(m[:, i] == 0) == 1:
            j = np.argwhere(m[:, i] == 0).reshape(-1)[0]
            return i, j
    return False


def assign_single_zero_lines(m: numpy.ndarray, assignment: np.ndarray):
    val = find_rows_single_zero(m)
    while val:
        i, j = val[0], val[1]
        m[i, j] += 1
        m[:, j] += 1
        assignment[i, j] = 1
        val = find_rows_single_zero(m)
        print("while1")

    val = find_cols_single_zero(m)
    count = 0
    while val:
        count += 1
        if count > 4:
            print("warning")
        i, j = val[0], val[1]
        m[i, j] += 1
        m[i, :] += 1
        assignment[i, j] = 1
        val = find_cols_single_zero(m)
        print("while2")

    return assignment


def first_zero(m: np.ndarray):
    return np.argwhere(m == 0)[0][0], np.argwhere(m == 0)[0][1]


def final_assignment(init_m: np.ndarray, m: np.ndarray):
    assignment = np.zeros(m.shape, dtype=int)
    assignment = assign_single_zero_lines(m, assignment)
    while np.sum(m == 0) > 0:
        print("s")
        i, j = first_zero(m)
        assignment[i, j] = 1
        m[i, :] += 1
        m[:, j] += 1
        assignment = assign_single_zero_lines(m, assignment)

    return assignment * init_m, assignment


def hungarian_algorithm(mc: np.ndarray):
    m = mc.copy()
    step1(m)
    step2(m)
    n_lines = 0
    max_length = np.maximum(m.shape[0], m.shape[1])
    i = 0
    while n_lines != max_length:
        lines = step3(m)
        n_lines = len(lines[0]) + len(lines[1])
        print("n_lines: ", n_lines)
        if n_lines == max_length:
            step5(m, lines[0], lines[1])
    result = final_assignment(mc, m)
    print("finish hungarian algorithm")
    return result


def run(a: np.ndarray):
    res = hungarian_algorithm(np.array(a))
    print("\nOptimal Matching:\n", res[1], "\n Value: ", np.sum(res[0]))


# course implementation
a = np.random.randint(100, size=(3, 3))
run(a)
print("\nOptimal Matching:\n", res[1], "\n Value: ", np.sum(res[0]))


# random implementation found online
m = Munkres()
indexes = m.compute(a)
total = 0
for row, column in indexes:
    value = a[row][column]
    total += value
    print(f'({row}, {column}) -> {value}')
print(f'total cost: {total}')
