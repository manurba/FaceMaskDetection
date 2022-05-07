import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('datos.csv')

ax = data.plot(x='Epoch n',y='Loss')
ax.set_ylabel('Loss')
ax.set_title('Validation Loss of the Model')
ax.set_xlabel('Epoch Number')
ax.set_xticks([0, 4, 9, 14, 19, 24])
ax.set_xticklabels(['Epoch 1','Epoch 5', 'Epoch 10', 'Epoch 15', 'Epoch 20', 'Epoch 25'])

plt.grid()

plt.show()