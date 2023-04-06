# bruteforce methode for the knapsack problem by recurrence


def bruteforce(weight, profit, capacity, best=0, money=0):
    if len(weight) == 0:
        return money
    for i in range(len(weight)):
        if len(weight) == 0:
            return money
        c = capacity
        w = weight.copy()
        p = profit.copy()
        m = money
        if w[i] <= c:
            c -= w[i]
            m += p[i]

        w.pop(i)
        p.pop(i)

        best = max(best, bruteforce(w, p, c, best, m))
    return best


print(bruteforce([5, 8, 7, 3], [3, 6, 2, 4], 10))
