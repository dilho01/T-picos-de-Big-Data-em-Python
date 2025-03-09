import tkinter as tk
from tkinter import messagebox
import pandas as pd  # type: ignore
from sklearn import tree  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

# Treinamento do modelo (executado uma única vez)
df = pd.read_csv('dados/iris.csv')
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Criação da interface gráfica
root = tk.Tk()
root.title("Classificador Iris")

# Labels e campos de entrada
tk.Label(root, text="Sepal Length:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
entrada1 = tk.Entry(root)
entrada1.grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="Sepal Width:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
entrada2 = tk.Entry(root)
entrada2.grid(row=1, column=1, padx=5, pady=5)

tk.Label(root, text="Petal Length:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
entrada3 = tk.Entry(root)
entrada3.grid(row=2, column=1, padx=5, pady=5)

tk.Label(root, text="Petal Width:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
entrada4 = tk.Entry(root)
entrada4.grid(row=3, column=1, padx=5, pady=5)

def classificar():
    try:
        sepal_length = float(entrada1.get())
        sepal_width = float(entrada2.get())
        petal_length = float(entrada3.get())
        petal_width = float(entrada4.get())
    except ValueError:
        messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos.")
        return

    # Preparar os dados para a predição
    df_novo = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                           columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
    resultado = clf.predict(df_novo)
    messagebox.showinfo("Resultado", f"Predição: {resultado[0]}")

# Botão para acionar a classificação
btn_classificar = tk.Button(root, text="Classificar", command=classificar)
btn_classificar.grid(row=4, column=0, columnspan=2, pady=10)

root.mainloop()
