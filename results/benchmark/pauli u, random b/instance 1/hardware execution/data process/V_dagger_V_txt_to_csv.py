import pandas as pd
file1 = open("V_dagger_V.txt", 'r')
file1_reversed = reversed(list(file1.readlines()))
id_list_pre = [i.rstrip() for i in file1_reversed]
print(id_list_pre)
file1.close()



counter = 0
num_term = 3
tree_depth = 4
ip_idxes = [[0 for _ in range(tree_depth)] for _ in range(tree_depth)]
for i in range(tree_depth):
    for j in range(i, tree_depth):
        element_idxes = [[0 for _ in range(num_term)] for _ in range(num_term)]
        for k in range(num_term):
            for l in range(num_term):
                element_idxes[k][l] = (id_list_pre[counter],id_list_pre[counter+1])
                counter += 2
        ip_idxes[i][j] = element_idxes

# Create DataFrame
V_dagger_V_df = pd.DataFrame(ip_idxes)
# Save to CSV
V_dagger_V_csv_filename = "V_dagger_V_formal.csv"
V_dagger_V_df.to_csv(V_dagger_V_csv_filename, index=False)
