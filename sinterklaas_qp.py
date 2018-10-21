import cvxpy as cvx
import numpy as np
import attr
import math


@attr.s
class Child(object):
    name = attr.ib()
    age = attr.ib()
    sex = attr.ib()


kids = [
    Child('lisa', 12, 1),
    Child('jan', 8, -1),
    Child('piet', 9, -1),
    Child('dirk', 16, -1),
]

friends = [
    # (0, 3)  # dirk and lisa
    (1, 3)  # piet and dirk
]

# for k in range(15):
for k in range(20):
    kids.append(Child(str(k), 10 + (k % 7), -1))


groups = ['group ' + str(g) for g in range(3)]

n = len(kids)
m = len(groups)


def group_means(nums, group_num):
    nums = sorted(nums)
    means = []

    group_size = len(nums) / group_num
    for g in range(group_num):
        begin = g * group_size
        end = begin + group_size
        total = sum(nums[int(math.ceil(begin)):int(math.floor(end))])
        frac = begin % 1
        total += frac * nums[int(begin)]
        frac = 1 - (end % 1)
        if frac < 1:
            total += frac * nums[int(end)]
        mean = total / group_size
        means.append(mean)

    return np.array(means)


print('Children:')
for kid in kids:
    print(kid)
print()

print('Siblings:')
for pair in friends:
    print("pair: {!r}".format(pair))
    print([kids[p] for p in pair], 'are siblings')
print()


ages = [kid.age for kid in kids]
means = group_means(ages, len(groups))

scores = np.array([np.abs(kid.age - means)**2 for kid in kids])
# print("scores: {!r}".format(scores))

# x = cvx.Variable(shape=(n, m), name="kid_in_group")
x = cvx.Variable(shape=(n, m), name="kid_in_group", boolean=True)

constraints = [x >= 0, x <= 1]

# ages = cvx.Parameter(shape=(2, n))
# ages.value = [[kid.age, kid.age] for kid in kids]


# x_flat = x.flatten()

# age_horizontal = np.hstack(m * [age_matrix])
# age_repeat = np.vstack(m * [age_horizontal])

opt = cvx.multiply(scores, x)
obj = cvx.Minimize(cvx.sum(opt))



# Every kid should be in one group only
for k in range(n):
    number_of_groups = sum(x[k, g] for g in range(m))
    constraints.append(number_of_groups == 1)

# Group size between 8 and 10
for g in range(m):
    groupsize = sum(x[k, g] for k in range(n))
    # constraints.append(groupsize == 9)
    constraints.append(groupsize >= 8)
    constraints.append(groupsize <= 10)

# Friends must be in the same group
for (friend1, friend2) in friends:
    constraints.append(x[friend1, :] == x[friend2, :])

problem = cvx.Problem(obj, constraints)

solution = problem.solve(verbose=True)

print("solution: {!r}".format(solution))

print("x.value: {!r}".format(x.value))



for k, kid in enumerate(kids):
    g = np.argmax(x.value[k, :])
    print(kid, groups[g])
