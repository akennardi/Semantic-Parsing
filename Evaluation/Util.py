import random

from Class.Tree import Tree
from operator import itemgetter


random.seed(1)


def convert_to_tree(r_list, i_left, i_right, form_manager):
    t = Tree()
    level = 0
    left = -1
    for i in range(i_left, i_right):
        if r_list[i] == form_manager.get_symbol_idx('('):
            if level == 0:
                left = i
            level = level + 1
        elif r_list[i] == form_manager.get_symbol_idx(')'):
            # print("closing")
            level = level - 1
            if level == 0:
                if i == left + 1:
                    c = r_list[i]
                else:
                    c = convert_to_tree(r_list, left + 1, i, form_manager)
                # print("tree add")
                t.add_child(c)
        elif level == 0:
            # print("child")
            t.add_child(r_list[i])
    return t


def norm_tree(r_list, form_manager):
    # print("starting norm tree")
    # print(r_list)
    # test = convert_to_tree(r_list, 0, len(r_list), form_manager)
    # print("test")
    # print(test)
    q = [convert_to_tree(r_list, 0, len(r_list), form_manager)]
    # print("after convert")
    head = 0
    # for t in q:
    while head < len(q):
        # print("head; {}, len q: {}\n".format(head, len(q)))
        t = q[head]
        # print('string')
        # print(t.to_string())
        # print('num')
        # print(t.num_children)
        # print(form_manager.get_symbol_idx('and')) = 6
        # print(form_manager.get_symbol_idx('or')) =53
        # if this level is "and/or" operator
        # print('children')
        # print(t.children)
        # print(form_manager.get_symbol_idx('and'))
        # print(form_manager.get_symbol_idx('or'))

        if len(t.children) > 0:
            if (t.children[0] == form_manager.get_symbol_idx('and')) or (
                    t.children[0] == form_manager.get_symbol_idx('or')):
                # sort the following subchildren
                # k = {}
                k = []
                # debug
                # print ('number of children:{}'.format(len(t.children)))
                for i in range(1, len(t.children)):
                    if isinstance(t.children[i], Tree):
                        # print("tree inside and/or if statement")
                        # print(t.children[i].to_string())
                        # print('tree child ', t.children[i].to_string())
                        # k[t.children[i].to_string()] = i
                        k.append((t.children[i].to_string(), i))
                        # print (i)
                        # print ('----------append done----------')
                    else:
                        # print("not a tree child")
                        # print('reg child ', str(t.children[i]))
                        # k[str(t.children[i])] = i
                        k.append((str(t.children[i]), i))
                sorted_t_dict = []
                # print('len k ', len(k))
                k.sort(key=itemgetter(0))
                # for key1 in sorted(k):
                for key1 in k:
                    sorted_t_dict.append(t.children[key1[1]])
                # print(len(t.children))
                # print(len(sorted_t_dict))
                # print("print sorted")
                # print(sorted(k))
                # print(sorted_t_dict)
                # print(t.to_string())
                # print(len(t.children))
                # print(t.num_children)
                # print('len ', len(sorted_t_dict))
                # print('dict ', sorted_t_dict)
                # print('num children ', t.num_children)
                for i in range(t.num_children - 1):
                    # print('i ', i)
                    t.children[i + 1] = \
                        sorted_t_dict[i]
        # add children to q
        for i in range(len(t.children)):
            if isinstance(t.children[i], Tree):
                # print("this is a tree: {}".format(t.children[i].to_string()))
                q.append(t.children[i])

        head = head + 1
    return q[0]


def is_all_same(c1, c2):
    if len(c1) == len(c2):
        all_same = True
        for j in range(len(c1)):
            if c1[j] != c2[j]:
                all_same = False
                break
        return all_same
    else:
        return False


def compute_accuracy(candidate_list, reference_list):
    # if len(candidate_list) != len(reference_list):
    #     print("candidate list has length {}, reference list has length {}\n".format(len(candidate_list),
    #                                                                                 len(reference_list)))

    len_min = min(len(candidate_list), len(reference_list))
    c = 0
    error =[]
    for i in range(len_min):
        # print(candidate_list[i])
        # print(reference_list[i])
        if is_all_same(candidate_list[i], reference_list[i]):
            # print("above was all same")
            c = c + 1
        if not is_all_same(candidate_list[i], reference_list[i]):
            error.append(i)

    # print('correct prediction: {}'.format(c))
    # print('list of error:{}'.format(error))
    return c / float(len_min)


def compute_tree_accuracy(candidate_list_, reference_list_, form_manager):
    candidate_list = []
    # print('number of candidates:{}'.format(len(candidate_list_)))
    # print('number of references:{}'.format(len(reference_list_)))
    for i in range(len(candidate_list_)):
        # print("candidate\n\n")
        candidate_list.append(norm_tree(candidate_list_[i], form_manager).to_list(form_manager))
    reference_list = []
    for i in range(len(reference_list_)):
        # print("reference\n\n")
        reference_list.append(norm_tree(reference_list_[i], form_manager).to_list(form_manager))
    return compute_accuracy(candidate_list, reference_list)
