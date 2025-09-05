import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os, json, random, sys

# ensure project root is in path if run as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_def_test10.model_def import BigCNN
from train_eval import train_model, evaluate_model
from more_baselines.base_train_eval import train_lora, train_dann_gate

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def load_data_split(seed, flip_ratio=1.0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    total_indices = np.arange(len(trainset))
    rng = np.random.RandomState(seed)
    rng.shuffle(total_indices)

    pretrain_indices = total_indices[:10000]
    raw_indices      = total_indices[10000:10000+4000]

    pretrain_subset = Subset(trainset, pretrain_indices)
    raw_set         = Subset(trainset, raw_indices)

    class RandomLabelDataset(Dataset):
        def __init__(self, subset, num_classes=10, flip_ratio=1.0):
            self.subset = subset
            self.num_classes = num_classes
            self.flip_ratio = flip_ratio
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            image, true_label = self.subset[idx]
            if np.random.rand() < self.flip_ratio:
                wrong_label = np.random.randint(0, self.num_classes - 1)
                if wrong_label >= true_label:
                    wrong_label += 1
                return image, wrong_label
            else:
                return image, true_label

    pretrain_dataset = RandomLabelDataset(pretrain_subset, flip_ratio=flip_ratio)
    return pretrain_dataset, raw_set, testset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_epochs = 60
    other_epochs    = 30
    num_runs = 5
    flip_ratios = [0.8, 0.4, 0.0]

    for flip_ratio in flip_ratios:
        save_path = f"./results_test10_base/noise_{flip_ratio}.json"

        pretrain_dataset, raw_set, test_dataset = load_data_split(seed=42, flip_ratio=flip_ratio)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=64, shuffle=True, num_workers=2)
        raw_loader      = DataLoader(raw_set, batch_size=64, shuffle=True, num_workers=2)
        test_loader     = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

        print("\n=== Pretraining External Model (BigCNN) on 10k Random (Corrupted) Samples ===")
        model_save_path = f"./model_test10/noise_{flip_ratio}.pt"
        if os.path.exists(model_save_path):
            external_model = torch.load(model_save_path).to(device)
            print("Loaded external model from:", model_save_path)
        else:
            external_model = BigCNN().to(device)
            train_model(external_model, pretrain_loader, pretrain_epochs, device)
            torch.save(external_model, model_save_path)
            print("Trained and saved external model to:", model_save_path)

        ext_acc, ext_auc, ext_f1, ext_minc = evaluate_model(external_model, test_loader, device)
        print(f"External Model Evaluation: Acc={ext_acc:.2f}% | AUC={ext_auc:.4f} | F1={ext_f1:.4f} | MinCAcc={ext_minc:.2f}%")

        metrics = {
            "lora":      {"acc": [], "auc": [], "f1": [], "min_cacc": []},
            "dann_gate": {"acc": [], "auc": [], "f1": [], "min_cacc": []},
        }

        for run_idx in range(num_runs):
            run_seed = 42 + run_idx
            print(f"\n=== flip_ratio={flip_ratio} | Run {run_idx+1}/{num_runs}, seed={run_seed} ===")
            _, raw_set_run, _ = load_data_split(seed=run_seed, flip_ratio=flip_ratio)
            run_loader = DataLoader(raw_set_run, batch_size=64, shuffle=True, num_workers=2)

            print("Training LoRA...")
            acc, auc, f1, minc = train_lora(external_model, run_loader, test_loader, device, epochs=other_epochs)
            metrics["lora"]["acc"].append(acc)
            metrics["lora"]["auc"].append(auc)
            metrics["lora"]["f1"].append(f1)
            metrics["lora"]["min_cacc"].append(minc)

            print("Training DANN-Gate...")
            acc, auc, f1, minc = train_dann_gate(external_model, run_loader, test_loader, device, epochs=other_epochs)
            metrics["dann_gate"]["acc"].append(acc)
            metrics["dann_gate"]["auc"].append(auc)
            metrics["dann_gate"]["f1"].append(f1)
            metrics["dann_gate"]["min_cacc"].append(minc)

            print(f"[Run {run_idx+1}] "
                  f"LoRA Acc={metrics['lora']['acc'][-1]:.2f}% | "
                  f"DANN-Gate Acc={metrics['dann_gate']['acc'][-1]:.2f}%")

        # aggregate mean/std
        final_results = {}
        for method, m_dict in metrics.items():
            acc_arr  = np.array(m_dict["acc"])
            auc_arr  = np.array(m_dict["auc"])
            f1_arr   = np.array(m_dict["f1"])
            minc_arr = np.array(m_dict["min_cacc"])
            final_results[method] = {
                "acc_mean": float(acc_arr.mean()),   "acc_std": float(acc_arr.std()),
                "auc_mean": float(auc_arr.mean()),   "auc_std": float(auc_arr.std()),
                "f1_mean":  float(f1_arr.mean()),    "f1_std":  float(f1_arr.std()),
                "min_cacc_mean": float(minc_arr.mean()), "min_cacc_std": float(minc_arr.std())
            }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as fp:
            json.dump(final_results, fp, indent=2)

        print(f"\nAll done. Final mean/std results saved to: {save_path}")

if __name__ == "__main__":
    main()
