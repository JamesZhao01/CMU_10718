import subprocess
import itertools


lrs = [3e-3, 1e-3, 3e-4]
epochs = [50]
negative_nums = [200]
embedding_dims = [128]
temps = [0.1, 0.5, 1.0, 2.0, 10.0]

params = list(itertools.product(lrs,
                                epochs,
                                negative_nums,
                                embedding_dims,
                                temps
                                ))

for param in params:
    lr, epoch, negative_num, embedding_size, temp = param
    print(f"Submitting job for LR: {lr}")
    flag = f"--learning_rate {lr}" if lr else ""
    flag += f" --epochs {epoch}" if epoch else ""
    flag += f" --negative_num {negative_num}" if negative_num else ""
    flag += f" --embedding_size {embedding_size}" if embedding_size else ""
    flag += f" --temperature {temp}" if temp else ""
    subprocess.run(f"sbatch run-train.sh {flag}", shell=True)
                                        
print(f"Submitted total {len(params)} jobs!")