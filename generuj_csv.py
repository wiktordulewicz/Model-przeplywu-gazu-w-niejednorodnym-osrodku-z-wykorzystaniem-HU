import numpy as np
import pandas as pd

# Ustawienie ziarna losowości dla powtarzalności wyników
np.random.seed(42)

nx, ny = 30, 30  # siatka 30x30 komórek
i_idx = []
j_idx = []
phi_list = []
k_list = []

for i in range(nx):
    for j in range(ny):
        i_idx.append(i)
        j_idx.append(j)
        
        # Generowanie porowatości w rozsądnych granicach (1% -50%)
        phi = np.random.uniform(0.01, 0.50)
        phi_list.append(phi)
        
        # Fizyczna korelacja przepuszczalności z porowatością (z lekkim szumem geologicznym)
        # Im większa porowatość, tym wyższa przepuszczalność w mD
        log_k = 12.0 * phi + np.random.normal(0, 0.25) - 0.5
        k = 10**log_k
        
        # Ograniczenie wartości fizycznych (od 0.1 mD do 800 mD)
        k_list.append(max(0.1, min(k, 800.0)))

# Tworzenie ramki danych
df = pd.DataFrame({
    'I': i_idx,
    'J': j_idx,
    'Porowatosc': phi_list,
    'Przepuszczalnosc': k_list
})

# Zapis do pliku CSV
df.to_csv('dane_zloza.csv', index=False)
print("Sukces! Wygenerowano plik 'dane_zloza.csv' gotowy do wgrania do aplikacji.")