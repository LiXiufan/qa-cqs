result = {'00000000100001000000': 25, '00001000110000000000': 22, '00001000110000100000': 20}
mea_corre = [None, None, None, None, None, None, None, None, None, None, None, None, None, [14, 0], [8, 1], [10, 2], [15, 3], [4, 4], [9, 5], [13, 6]]


mea_corre_dict = {}
q_count = 0
for i in mea_corre:
    if i is not None:
        mea_corre_dict[str(i[0])] = int(i[1])
        q_count += 1
print(mea_corre_dict)

result_after = {}
for outcome in result.keys():
    reduced_str_lst = [0 for _ in range(q_count)]
    value = result[outcome]
    string_list = list(outcome)
    for k in mea_corre_dict.keys():
        corre = mea_corre_dict[k]
        reduced_str_lst[corre] = string_list[int(k)]
    reduced_str = "".join(reduced_str_lst)
    if reduced_str in result_after.keys():
        result_after[reduced_str] += value
    else:
        result_after[reduced_str] = value
print(result_after)
