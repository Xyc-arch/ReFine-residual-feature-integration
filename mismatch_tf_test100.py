# mismatch_tf_test100.py

import torch, random, json, os, copy
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from train_eval_test100 import (
    train_model, train_linear_prob,
    train_enhanced_model, train_distillation,
    evaluate_model
)
from model_def_test100.model_def10_tf import (
    TransformerClassifier,
    EnhancedTransformer,
    BaselineAdapterTransformer,
    BigTransformer
)

# reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# --------------------------
# CIFAR-100 Student data split
# --------------------------
def load_student_data(seed, raw_size=4000):
    rand = random.Random(seed)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    ds = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=tf
    )
    idx = list(range(len(ds)))
    rand.shuffle(idx)
    raw_idx = idx[:raw_size]
    return Subset(ds, raw_idx), torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=tf
    )

# --------------------------
# CIFAR-10 Teacher data split
# --------------------------
def load_teacher_data(pretrain_size=10000, seed=42):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    ds = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=tf
    )
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return Subset(ds, idx[:pretrain_size]), torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=tf
    )

# --------------------------
# Main experiment loop
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "./results_test100/mismatch_tf_reverse.json"

    num_epochs_teacher = 60
    num_epochs_student = 30
    num_runs = 5

    # 1) Train or load teacher on CIFAR-10
    teacher_path = "./model_test100/mismatch_tf_teacher.pt"
    if os.path.exists(teacher_path):
        teacher = torch.load(teacher_path).to(device)
    else:
        train_ds, test_teacher_ds = load_teacher_data()
        loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
        teacher = BigTransformer().to(device)
        train_model(teacher, loader, num_epochs_teacher, device)
        torch.save(teacher, teacher_path)

    # fixed CIFAR-100 test set
    _, test_student_ds = load_student_data(seed=42)
    test_loader = DataLoader(test_student_ds, batch_size=32, shuffle=False, num_workers=2)

    # metrics
    metrics = {
        "baseline": {}, "linear_prob": {},
        "enhanced_concat": {}, "baseline_adapter": {}
    }
    for m in metrics:
        metrics[m] = {"acc": [], "auc": [], "f1": [], "min_cacc": []}

    for run in range(num_runs):
        seed = 42 + run
        raw_ds, _ = load_student_data(seed)
        raw_loader = DataLoader(raw_ds, batch_size=32, shuffle=True, num_workers=2)

        # a) baseline student
        student = TransformerClassifier().to(device)
        train_model(student, raw_loader, num_epochs_student, device)
        a,u,f,mn = evaluate_model(student, test_loader, device)
        metrics["baseline"]["acc"].append(a)
        metrics["baseline"]["auc"].append(u)
        metrics["baseline"]["f1"].append(f)
        metrics["baseline"]["min_cacc"].append(mn)

        # b) linear probe on teacher
        lp = copy.deepcopy(teacher)
        for p in lp.parameters(): p.requires_grad=False
        for p in lp.classifier.parameters(): p.requires_grad=True
        lp = lp.to(device)
        train_linear_prob(lp, raw_loader, num_epochs_student, device)
        a,u,f,mn = evaluate_model(lp, test_loader, device)
        metrics["linear_prob"]["acc"].append(a)
        metrics["linear_prob"]["auc"].append(u)
        metrics["linear_prob"]["f1"].append(f)
        metrics["linear_prob"]["min_cacc"].append(mn)

        # c) enhanced concat
        enh = EnhancedTransformer().to(device)
        train_enhanced_model(enh, raw_loader, teacher, num_epochs_student, device)
        a,u,f,mn = evaluate_model(
            enh, test_loader, device,
            enhanced=True, external_model=teacher
        )
        metrics["enhanced_concat"]["acc"].append(a)
        metrics["enhanced_concat"]["auc"].append(u)
        metrics["enhanced_concat"]["f1"].append(f)
        metrics["enhanced_concat"]["min_cacc"].append(mn)

        # d) adapter
        adp = BaselineAdapterTransformer(copy.deepcopy(teacher)).to(device)
        train_model(adp, raw_loader, num_epochs_student, device)
        a,u,f,mn = evaluate_model(adp, test_loader, device)
        metrics["baseline_adapter"]["acc"].append(a)
        metrics["baseline_adapter"]["auc"].append(u)
        metrics["baseline_adapter"]["f1"].append(f)
        metrics["baseline_adapter"]["min_cacc"].append(mn)

        print(f"Run {run+1}/{num_runs} | "
              f"Base={metrics['baseline']['acc'][-1]:.2f}% "
              f"LP={metrics['linear_prob']['acc'][-1]:.2f}% "
              f"ENH={metrics['enhanced_concat']['acc'][-1]:.2f}% "
              f"ADP={metrics['baseline_adapter']['acc'][-1]:.2f}%")

    # aggregate
    final = {}
    for k,v in metrics.items():
        A = np.array(v["acc"]);  U = np.array(v["auc"])
        F = np.array(v["f1"]);   M = np.array(v["min_cacc"])
        final[k] = {
            "acc_mean":float(A.mean()),    "acc_std":float(A.std()),
            "auc_mean":float(U.mean()),    "auc_std":float(U.std()),
            "f1_mean":float(F.mean()),     "f1_std":float(F.std()),
            "min_cacc_mean":float(M.mean()),"min_cacc_std":float(M.std())
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path,"w") as f:
        json.dump(final, f, indent=2)
    print("Done, results saved to", save_path)

if __name__ == "__main__":
    main()
