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
    name = f"{args.dataset}_{args.model}_{args.loss}loss"
    save_path = os.path.join(path, name)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "result.txt"), "w") as f:
        f.write('Epoch, Coverage, Top1 Accuracy, Average Size\n')
        f.write(f"{args.epochs}, {result_dict["Coverage"]}, {result_dict["Top1Accuracy"]}, {result_dict["AverageSetSize"]}\n\n")
        f.write("Detailed Setup \n")
        args_dict = vars(args)
        for k, v in args_dict.items():
            if v is not None:
                f.write(f"{k}: {v}\n")
    torch.save(trainer.net.state_dict(), os.path.join(save_path, "model.pth"))
    if args.adapter == "True":
        torch.save(trainer.adapter.adapter_net.state_dict(), os.path.join(save_path, "adapter.pth"))
