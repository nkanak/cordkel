import matplotlib.pyplot as plt
import numpy
import os
import json

plt.rcParams['font.family'] = 'Palatino Linotype'

results = []
for fpath in os.listdir('results'):
    with open('results/' + fpath) as f:
        results.append(json.load(f))

aa_pm_val_accuracies = [res['nn'][0]['history']['val_accuracy'] for res in results]
aa_j_val_accuracies = [res['nn'][7]['history']['val_accuracy'] for res in results]

aa_pm_losses = [res['nn'][0]['history']['loss'] for res in results]
aa_j_losses = [res['nn'][7]['history']['loss'] for res in results]

aa_pm_val_losses = [res['nn'][0]['history']['val_loss'] for res in results]
aa_j_val_losses = [res['nn'][7]['history']['val_loss'] for res in results]

# prepare plots
fig, [ax_val_acc, ax_losses] = plt.subplots(1, 2, figsize=(11, 5))

ax_val_acc.set_xticks([1, 2, 3, 4, 5])
ax_val_acc.set_xlabel('epoch')
ax_val_acc.set_ylabel('accuracy')
ax_val_acc.plot([1, 2, 3, 4, 5], numpy.mean(aa_pm_val_accuracies, axis=0), label='AA_PM', marker='.')
ax_val_acc.plot([1, 2, 3, 4, 5], numpy.mean(aa_j_val_accuracies, axis=0), label='AA_J', marker='*')

ax_losses.set_xticks([1, 2, 3, 4, 5])
ax_losses.set_xlabel('epoch')
ax_losses.set_ylabel('loss')
ax_losses.plot([1, 2, 3, 4, 5], numpy.mean(aa_pm_val_losses, axis=0), label='Val AA_PM', marker='.')
ax_losses.plot([1, 2, 3, 4, 5], numpy.mean(aa_j_val_losses, axis=0), label='Val AA_J', marker='*')
ax_losses.plot([1, 2, 3, 4, 5], numpy.mean(aa_pm_losses, axis=0), label='Train AA_PM', marker='s')
ax_losses.plot([1, 2, 3, 4, 5], numpy.mean(aa_j_losses, axis=0), label='Train AA_J', marker='d')

ax_val_acc.set_title('(a) Comparison of validation accuracies')
ax_losses.set_title('(b) Comparison of losses')

ax_val_acc.grid(linestyle='--')
ax_losses.grid(linestyle='--')


ax_val_acc.legend(loc='lower right')
ax_losses.legend(loc='upper right')
#plt.show()
plt.savefig('accuracy_plots.svg', dpi=1000, bbox_inches='tight')