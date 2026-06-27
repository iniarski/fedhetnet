import sys
import pandas as pd

source = sys.argv[1]

df = pd.read_csv(source)

metrics = df.iloc[-1].to_dict()

print("\\begin{table}")
print("\\caption{TODO}")
print("\\label{tab:TODO}")
print("\\centering")
print("\\begin{tabular}{| c | c | c | c | c |}")
print("\\hline")
print("TODO & Accuracy & Precision & Recall & F1 \\\\ \\hline")
print(f"TODO & {100 * metrics['val/acc']:.2f}\\% & {100 * metrics['val/prec']:.2f}\\% & {100 * metrics['val/rec']:.2f}\\% & {100 * metrics['val/f1']:.2f}\\% \\\\ \\hline")
print("\\end{tabular}")
print("\\end{table}")

print()

print("\\begin{table}")
print("\\caption{TODO}")
print("\\label{tab:TODO}")
print("\\centering")
print("\\begin{tabular}{| c | c | c | c |}")
print("\\hline")
print("Class & Precision & Recall & F1 \\\\ \\hline")
for _class in range(7):
    print(f"{_class} & {100 * metrics[f'val/class_{_class}_prec']:.2f}\\% & {100 * metrics[f'val/class_{_class}_rec']:.2f}\\% & {100 * metrics[f'val/class_{_class}_f1']:.2f}\\% \\\\ \\hline")
print("\\end{tabular}")
print("\\end{table}")
