import torch
import numpy as np
import random
import os
import datetime

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_exp_result(args, trainer, result_dict, path=None):
    current_time = datetime.datetime.now()
    month_day = current_time.strftime("%b%d")
    if path is None:
        path = f"./experiment/{args.algorithm}"
    name = f"{args.dataset}_{args.model}_{args.loss}loss"
    if args.multi_instance_learning == "True":
        path = f"{path}/mil"
        name = f"{args.extract_feature_model}_{name}"
        if args.kfold is not None:
            name = f"{name}_cross_validation"

    save_path = os.path.join(path, month_day)
    save_path = os.path.join(save_path, args.test_score)

    i = 0
    while True:
        if os.path.exists(os.path.join(save_path, f"{name}_{i}")):
            i += 1
        else:
            break
    folder_path = os.path.join(save_path, f"{name}_{i}")
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, "result.txt"), "w") as f:
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

