from evomol.evaluation.logp import log_p
from evomol.representation import Molecule

if __name__ == "__main__":
    import os

    from evomol import default_parameters as dp
    from evomol import evaluation

    dp.setup_default_parameters()

    print((log_p(Molecule("C=CCC=C"))))

    with open("./results/" + strategy + "_walk/" + only_valid_str + "/" + str(n_steps) + "/"
              + path[0] + "/" + "_".join([action.__name__ for action in MolecularGraph.action_space]) + "/"
              + evaluation_function_str + "walk.csv", "r", newline='') as file:
        df = pd.read_csv(file, converters={"action_context": lambda x: ast.literal_eval(x.strip())
                                                                       if isinstance(x, str) and x.strip() else None,
                                           "action": lambda x: x if isinstance(x, str) else None})

        # dict_test = ast.literal_eval(df['action_context'])
        print(df['action'][1])

        exit()

