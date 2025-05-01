import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

# Carregamento e treinamento do modelo
df = pd.read_csv('dados/iris.csv')
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Criação da janela principal
root = tk.Tk()
root.title("Classificador de Íris")
root.geometry("420x520")
root.resizable(False, False)
root.configure(bg="#f0f4f8")

# Estilo geral
style = ttk.Style()
style.theme_use("default")
style.configure("TFrame", background="#f0f4f8")
style.configure("TLabel", background="#f0f4f8", font=("Segoe UI", 12))
style.configure("Custom.TEntry", font=("Segoe UI", 12), padding=6)
style.configure("Accent.TButton", background="#4CAF50", foreground="white", font=("Segoe UI", 12, "bold"), padding=8)
style.map("Accent.TButton", background=[("active", "#45a049")], foreground=[("active", "white")])

# Frame principal
frame = ttk.Frame(root, padding=20, style="TFrame")
frame.pack(expand=True, fill="both")

# Título
titulo = tk.Label(frame, text="Classificador de Flores Íris", font=("Segoe UI", 18, "bold"), bg="#f0f4f8", fg="#333")
titulo.grid(row=0, column=0, columnspan=2, pady=15)

# Função para criar campos de entrada bonitos
def criar_campo(label_texto, row):
    ttk.Label(frame, text=label_texto).grid(row=row, column=0, padx=5, pady=10, sticky="e")
    entrada = ttk.Entry(frame, width=24, style="Custom.TEntry")
    entrada.grid(row=row, column=1, padx=5, pady=10)
    return entrada

# Inputs
entrada_sepal_length = criar_campo("Comprimento da Sépala (cm):", 1)
entrada_sepal_width = criar_campo("Largura da Sépala (cm):", 2)
entrada_petal_length = criar_campo("Comprimento da Pétala (cm):", 3)
entrada_petal_width = criar_campo("Largura da Pétala (cm):", 4)

# Função para classificar
def classificar():
    try:
        sepal_length = float(entrada_sepal_length.get())
        sepal_width = float(entrada_sepal_width.get())
        petal_length = float(entrada_petal_length.get())
        petal_width = float(entrada_petal_width.get())

        if any(x <= 0 for x in [sepal_length, sepal_width, petal_length, petal_width]):
            messagebox.showerror("Erro", "Os valores devem ser maiores que zero.")
            return

        df_novo = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=['sepal length', 'sepal width', 'petal length', 'petal width']
        )
        resultado = clf.predict(df_novo)[0]
        messagebox.showinfo("Resultado", f"A flor é: {resultado}")

    except ValueError:
        messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos.")

# Botão de classificar
btn_classificar = ttk.Button(frame, text="Classificar Flor", command=classificar, style="Accent.TButton")
btn_classificar.grid(row=5, column=0, columnspan=2, pady=30)

# Iniciar a aplicação
root.mainloop()
