import torch
import numpy as np
import random
import os


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)




def save_exp_result(args, trainer, result_dict, path=None):
    if path is None:
        path = f"./experiment/{args.algorithm}"
    name = f"{args.dataset}_{args.model}_{args.test_score}_{args.loss}loss"
    if args.multi_instance_learning == "True":
        path = f"{path}/mil"
        name = f"{name}_{args.extract_feature_model}"
    save_path = os.path.join(path, name)

    i = 0
    while True:
        if os.path.exists(save_path + f"{i}"):
            i += 1
            save_path = os.path.join(path, name+f"{i}")
        else:
            break

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "result.txt"), "w") as f:
        for key in result_dict.keys():
            f.write(f"{key}: {result_dict[key]}\n")
        f.write("\nDetailed Setup \n")
        args_dict = vars(args)
        for k, v in args_dict.items():
            if v is not None:
                f.write(f"{k}: {v}\n")
    if args.save_model == "True":
        torch.save(trainer.net.state_dict(), os.path.join(save_path, "model.pth"))
        if args.adapter == "True":
            torch.save(trainer.adapter.adapter_net.state_dict(), os.path.join(save_path, "adapter.pth"))

