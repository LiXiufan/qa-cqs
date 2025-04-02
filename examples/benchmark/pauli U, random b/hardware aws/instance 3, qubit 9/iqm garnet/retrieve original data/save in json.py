from braket.aws import AwsQuantumTask
import pandas as pd
import json


def save_json(ip_id, V_or_q, params):
    r_i = ["real", "imag"]
    for i in range(2):
        task = AwsQuantumTask(arn=ip_id[i])
        status = task.state()
        if status != 'COMPLETED':
            return ValueError("I am sorry, your current task is in the status of", status)
        else:
            result = task.result()
            if V_or_q == "V":
                file_name = "V_dagger_V_" + str(params[0]) + "_" + str(params[1]) + "_element_" + str(params[2]) + "_" + str(params[3]) + "_term_" + r_i[i] + ".json"
            elif V_or_q == "q":
                file_name = "q_" + str(params[0]) + "_element_" + str(params[1]) + "_term_" + r_i[i] + ".json"
            else:
                raise ValueError("We only record either `V` or `q`.")
            with open(file_name, 'w', encoding='utf-8') as file:
                json.dump(str(result), file, ensure_ascii=False, indent=4)

def retrieve_original_data_V_dagger_V(num_term, tree_depth, ip_idxes):
    for i in range(tree_depth):
        for j in range(i, tree_depth):
            element_idxes = eval(ip_idxes[i][j])
            for k in range(num_term):
                for l in range(num_term):
                    ip_id = element_idxes[k][l][0]
                    save_json(ip_id, "V", [i, j, k, l])


def retrieve_original_data_q(num_term, tree_depth, ip_idxes):
    for i in range(tree_depth):
        element_idxes = ip_idxes[i]
        for k in range(num_term):
            ip_id = eval(element_idxes[k])[0]
            save_json(ip_id, "q", [i, k])


# retrieve hardware result
V_dagger_V_csv_filename = "V_dagger_V_formal.csv"
q_csv_filename = "q_formal.csv"
V_dagger_V_idxes = pd.read_csv(V_dagger_V_csv_filename).values.tolist()
q_idxes = pd.read_csv(q_csv_filename).values.tolist()

retrieve_original_data_V_dagger_V(5, 10, ip_idxes=V_dagger_V_idxes)

retrieve_original_data_q(5, 10, ip_idxes=q_idxes)




