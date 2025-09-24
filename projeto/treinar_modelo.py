from surprise import SVD, Dataset, Reader
import pickle
import pandas as pd
import os

# Caminho absoluto relativo ao script
caminho_atual = os.path.dirname(os.path.abspath(__file__))
ratings = pd.read_csv(os.path.join(caminho_atual, "u.data"), sep="\t", names=['userId', 'movieId', 'rating', 'timestamp'])

# 2️⃣ Preparar dados para o Surprise
reader = Reader(rating_scale=(1,5))
data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)

# 3️⃣ Treinar modelo SVD
trainset = data.build_full_trainset()
modelo = SVD()
modelo.fit(trainset)

# 4️⃣ Salvar modelo em arquivo 'modelo_svd.pkl' na mesma pasta
with open("modelo_svd.pkl", "wb") as f:
    pickle.dump(modelo, f)

print("Modelo treinado e salvo com sucesso!")
