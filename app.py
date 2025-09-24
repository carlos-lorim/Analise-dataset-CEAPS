from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

# 1️⃣ Definir diretório base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2️⃣ Carregar modelo SVD
modelo_path = os.path.join(BASE_DIR, "modelo_svd.pkl")
with open(modelo_path, "rb") as f:
    modelo = pickle.load(f)

# 3️⃣ Carregar dados de filmes (ajustando as colunas do MovieLens 100k)
filmes_path = os.path.join(BASE_DIR, "projeto", "u.item")
filmes = pd.read_csv(
    filmes_path,
    sep="|",
    encoding="latin-1",
    header=None,  # sem cabeçalho no arquivo original
    usecols=[0, 1],  # apenas movieId e título
    names=["movieId", "title"]
)

# 4️⃣ Criar app Flask
app = Flask(__name__)

# 5️⃣ Criar endpoint de recomendação
@app.route("/recomendar", methods=["POST"])
def recomendar():
    dados = request.get_json()
    user_id = dados.get("user_id")
    
    if user_id is None:
        return jsonify({"erro": "user_id não fornecido"}), 400
    
    # Calcular as 5 primeiras recomendações (exemplo simples)
    recomendacoes = []
    for _, row in filmes.head(5).iterrows():
        movie_id = int(row["movieId"])
        title = row["title"]
        pred = modelo.predict(user_id, movie_id)
        recomendacoes.append({
            "movieId": movie_id,
            "title": title,
            "rating_pred": float(pred.est)  # garantir JSON serializável
        })
    
    return jsonify(recomendacoes)

# 6️⃣ Rodar API
if __name__ == "__main__":
    app.run(debug=True)
