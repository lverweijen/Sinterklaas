import cvxpy as cvx
import numpy as np
import attr
import math
from pprint import pprint
from collections import defaultdict


@attr.s
class Child(object):
    name = attr.ib()
    age = attr.ib()
    sex = attr.ib()


@attr.s
class Leader(object):
    name = attr.ib()


registrations = [
    [Child('lisa', 12,'f')],
    [Child('piet', 9, 'm')],
    [Child('jan', 8, 'm'), Child('dirk', 12, 'm'), Child('anne', 9, 'f')],
    [Child('bert', 6, 'm'), Child('ernie', 8, 'm'), Leader('dad of bert & ernie')],
]

# Random children
for k in range(20):
    if k % 6 < 3:
        gender = 'm'
    else:
        gender = 'f'
    child = Child(str(k), 5 + (k % 7), gender)
    registrations.append([child])

# Random leaders
for k in range(4):
    registrations.append([Leader(str(k))])

print("Registrations:")
for registration in registrations:
    print(registration)

print()

# Divide participants in groups
kids = []
leaders = []
kid_relations = []
leader_relations = []
kid_leader_relations = []

for registration in registrations:
    first_kid = None
    first_adult = None

    for participant in registration:
        if isinstance(participant, Child):
            kid_index = len(kids)
            kids.append(participant)
            if first_kid is not None:
                kid_relations.append((first_kid, kid_index))
            else:
                first_kid = kid_index

            if first_adult is not None:
                kid_leader_relations.append((kid_index, first_adult))

        elif isinstance(participant, Leader):
            adult_index = len(leaders)
            leaders.append(participant)
            if first_adult is not None:
                leaders.append((first_adult, adult_index))
            else:
                first_adult = adult_index

            if first_kid is not None:
                kid_leader_relations.append((first_kid, adult_index))

groups = ['group ' + str(g + 1) for g in range(3)]

n = len(kids)
m = len(groups)
r = len(leaders)


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

print('Leaders:')
for leader in leaders:
    print(leader)
print()

ages = [kid.age for kid in kids]
means = group_means(ages, len(groups))

scores = np.array([np.abs(kid.age - means)**2 for kid in kids])

x = cvx.Variable(shape=(n, m), name="kid_in_group", boolean=True)
y = cvx.Variable(shape=(r, m), name="adult_in_group", boolean=True)

genders = np.array([(kid.sex == 'f') - (kid.sex == 'm') for kid in kids])

age_opt = cvx.sum(cvx.multiply(scores, x))
sex_opt = cvx.sum((genders @ x) ** 2)
leader_opt = cvx.sum(y @ means)

obj = cvx.Minimize(age_opt + sex_opt + leader_opt)

constraints = []

# Every kid should be in one group only
for k in range(n):
    number_of_groups = cvx.sum(x[k, :])
    constraints.append(number_of_groups == 1)

# Every leader should be in one group only
for l in range(r):
    number_of_groups = cvx.sum(y[l, :])
    constraints.append(number_of_groups == 1)

# Group size between 8 and 10
for g in range(m):
    groupsize = cvx.sum(x[:, g])
    constraints.append(groupsize >= 8)
    constraints.append(groupsize <= 10)

# At most 2 leaders a group
for g in range(m):
    leaders_in_group = cvx.sum(y[:, g])
    constraints.append(leaders_in_group <= 2)

# Friends must be in the same group
for (friend1, friend2) in kid_relations:
    constraints.append(x[friend1, :] == x[friend2, :])

for (friend1, friend2) in leader_relations:
    constraints.append(y[friend1, :] == y[friend2, :])

for (friend1, friend2) in kid_leader_relations:
    constraints.append(x[friend1, :] == y[friend2, :])

problem = cvx.Problem(obj, constraints)

pprint("problem.get_problem_data(): {!r}".format(problem.get_problem_data(cvx.ECOS_BB)))
print()

solution = problem.solve(verbose=False, solver=cvx.ECOS_BB)
print("problem.status: {!r}".format(problem.status))
print("solution: {!r}".format(solution))
print("age_opt.value: {!r}".format(age_opt.value))
print("sex_opt.value: {!r}".format(sex_opt.value))
print("leader_opt.value: {!r}".format(leader_opt.value))
print()

group_members = defaultdict(list)

for l, leader in enumerate(leaders):
    g = np.argmax(y.value[l, :])
    group_members[groups[g]].append(leader)

for k, kid in enumerate(kids):
    g = np.argmax(x.value[k, :])
    # print(kid, groups[g])
    group_members[groups[g]].append(kid)

for group in sorted(group_members.keys()):
    members = group_members[group]
    print(group)
    for member in members:
        print(member)
    print()

