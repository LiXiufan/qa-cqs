# file1 = open("messages.txt", "a")
# L = ["This is Delhi \n", "This is Paris \n", "This is London \n"]
# file1.writelines(L)
# file1.close()
#
# # Append-adds at last
# file1 = open("messages.txt", "a")  # append mode
# file1.write("Today \n")
# file1.close()

from numpy import array
tree_depth = 2
num_term = 2


ip_idxes = [[[] for _ in range(tree_depth)] for _ in range(tree_depth)]
for i in range(tree_depth):
    for j in range(i, tree_depth):
        element_idxes = [[str(0) for _ in range(num_term)] for _ in range(num_term)]
        for k in range(num_term):
            for l in range(num_term):
                position = str(i)+ str(j)+ str(k)+ str(l)
                print(element_idxes[k][l])
                element_idxes[k][l] = position
        ip_idxes[i][j] = element_idxes
print(ip_idxes)
print()

zeros = array([[_ for _ in range(tree_depth)] for _ in range(tree_depth)])
for i in range(tree_depth):
    for j in range(i, tree_depth):
        element_idxes = ip_idxes[i][j]
        print(element_idxes)
        print()
        item = 0
        for k in range(num_term):
            for l in range(num_term):
                item += 1
                inner_product = element_idxes[k][l]
                print(inner_product)
                print()
        zeros[i][j] = item
print(zeros)