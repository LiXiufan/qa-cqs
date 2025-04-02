from braket.aws import AwsQuantumTask
import pandas as pd



def retrieve_data(ip_id):
    task = AwsQuantumTask(arn=ip_id)
    status = task.state()
    if status != 'COMPLETED':
        count = None
    else:
        count = task.result().measurement_counts
    return count

def retrieve_original_data_V_dagger_V(num_term, tree_depth, ip_idxes):
    ip_counts = [[0 for _ in range(tree_depth)] for _ in range(tree_depth)]
    for i in range(tree_depth):
        for j in range(i, tree_depth):
            element_idxes = eval(ip_idxes[i][j])
            element_counts = [[0 for _ in range(num_term)] for _ in range(num_term)]
            for k in range(num_term):
                for l in range(num_term):
                    ip_id = element_idxes[k][l]
                    count_r = retrieve_data(ip_id[0])
                    count_i = retrieve_data(ip_id[1])
                    element_counts[k][l] = (count_r, count_i)
            ip_counts[i][j] = element_counts
    return ip_counts


def retrieve_original_data_q(num_term, tree_depth, ip_idxes):
    ip_counts = [0 for _ in range(tree_depth)]
    for i in range(tree_depth):
        element_idxes = ip_idxes[i]
        element_counts = [0 for _ in range(num_term)]
        for k in range(num_term):
            ip_id = eval(element_idxes[k])
            count_r = retrieve_data(ip_id[0])
            count_i = retrieve_data(ip_id[1])
            element_counts[k] = (count_r, count_i)
        ip_counts[i] = element_counts
    return ip_counts


# retrieve hardware result
V_dagger_V_csv_filename = "V_dagger_V_formal.csv"
q_csv_filename = "q_formal.csv"
V_dagger_V_idxes = pd.read_csv(V_dagger_V_csv_filename).values.tolist()
q_idxes = pd.read_csv(q_csv_filename).values.tolist()

V_dagger_V_counts = retrieve_original_data_V_dagger_V(5, 8, ip_idxes=V_dagger_V_idxes)
# Create DataFrame
V_dagger_V_df = pd.DataFrame(V_dagger_V_counts)
# Save to CSV
V_dagger_V_counts_csv_filename = "V_dagger_V_counts.csv"
V_dagger_V_df.to_csv(V_dagger_V_counts_csv_filename, index=False)


q_counts = retrieve_original_data_q(5, 8, ip_idxes=q_idxes)
# Create DataFrame
q_df = pd.DataFrame(q_counts)
# Save to CSV
q_counts_csv_filename = "q_counts.csv"
q_df.to_csv(q_counts_csv_filename, index=False)



