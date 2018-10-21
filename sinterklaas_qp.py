import cvxpy as cvx
import numpy as np
import attr
import math


@attr.s
class Child(object):
    name = attr.ib()
    age = attr.ib()
    sex = attr.ib()


@attr.s
class Leader(object):
    name = attr.ib()


participants = [
    Child('lisa', 12, 1),
    Child('piet', 9, -1),
    [Child('jan', 8, -1), Child('dirk', 16, -1), Child('anne', 9, 1)],
    [Child('bert', 6, -1), Child('ernie', 8, -1), Leader('dad of bert & ernie')],
]

# Random children
for k in range(20):
    # gender = 2 * (k % 2) - 1
    # gender = 2 * (k < 10) - 1
    gender = 2 * (k & 7 < 3) - 1
    participants.append(Child(str(k), 5 + (k % 7), gender))

# Random leaders
for k in range(5):
    participants.append(Leader(str(k)))

print("Participants:")
for participant in participants:
    print(participant)

print()

# Divide participants in groups
kids = []
leaders = []
kid_relations = []
leader_relations = []
kid_leader_relations = []

for participant in participants:
    if not isinstance(participant, list):
        participant = [participant]

    first_kid = None
    first_adult = None

    for subparticipant in participant:
        if isinstance(subparticipant, Child):
            kid_index = len(kids)
            kids.append(subparticipant)
            if first_kid is not None:
                kid_relations.append((first_kid, kid_index))
            else:
                first_kid = kid_index

            if first_adult is not None:
                kid_leader_relations.append((kid_index, first_adult))

        elif isinstance(subparticipant, Leader):
            adult_index = len(leaders)
            leaders.append(subparticipant)
            if first_adult is not None:
                leaders.append((first_adult, adult_index))
            else:
                first_adult = adult_index

            if first_kid is not None:
                kid_leader_relations.append((first_kid, adult_index))



groups = ['group ' + str(g) for g in range(3)]

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

# print('Siblings:')
# for pair in kid_relations:
#     print("pair: {!r}".format(pair))
#     print([kids[p] for p in pair], 'are siblings')
# print()


ages = [kid.age for kid in kids]
means = group_means(ages, len(groups))

scores = np.array([np.abs(kid.age - means)**2 for kid in kids])

x = cvx.Variable(shape=(n, m), name="kid_in_group", boolean=True)
y = cvx.Variable(shape=(r, m), name="adult_in_group", boolean=True)

genders = np.array([kid.sex for kid in kids])

age_opt = cvx.sum(cvx.multiply(scores, x))
# sex_opt = 0
# sex_opt = 1 * cvx.sum(cvx.abs(genders @ x))
# sex_opt = 1 * cvx.sum(cvx.abs(genders @ x**2))
# sex_opt = 1 * cvx.sum(cvx.multiply(genders, x), 0) ** 2
# sex_opt = cvx.sum(cvx.sum(cvx.multiply(cvx.hstack(3 * [[genders]]), x), 0) ** 2)
sex_opt = cvx.sum((genders * x) ** 2)
leader_opt = 0.1 * cvx.sum(y)
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

solution = problem.solve(verbose=False)

print("solution: {!r}".format(solution))
print("age_opt.value: {!r}".format(age_opt.value))
print("sex_opt.value: {!r}".format(sex_opt.value))
print("leader_opt.value: {!r}".format(leader_opt.value))
print()

# print("x.value: {!r}".format(x.value))


from collections import defaultdict
group_members = defaultdict(list)

for k, kid in enumerate(kids):
    g = np.argmax(x.value[k, :])
    # print(kid, groups[g])
    group_members[groups[g]].append(kid)

for l, leader in enumerate(leaders):
    g = np.argmax(y.value[l, :])
    # print(leader, groups[g])
    group_members[groups[g]].append(leader)


for group, members in group_members.items():
    print(group)
    for member in members:
        print(member)
    print()

