"""
badanie_hu.py - wartość jednostek hydraulicznych w modelu dynamicznym.

PYTANIE BADAWCZE
Praca Ha Quang Man, Le i Jarzyny (2021) kończy się na modelu statycznym:
wykazuje, że podział 570 rdzeni ze złoża gazu Z na sześć jednostek
hydraulicznych pozwala odtworzyć przepuszczalność ze współczynnikiem
determinacji 0,97. Nie odpowiada natomiast na pytanie, czy przekłada się to
na prognozę wydobycia - a to dopiero decyduje o przydatności metody
w modelowaniu złożowym. Ten skrypt odpowiada na nie pomiarem.

SCHEMAT EKSPERYMENTU
Odtworzona jest sytuacja, z której wynika sens stosowania jednostek
hydraulicznych: przepuszczalność zmierzono tylko w rdzeniach z nielicznych
odwiertów, porowatość znana jest wszędzie z karotażu i sejsmiki. Rdzenie
pobierane są z otoczenia odwiertów, nie z losowo rozrzuconych komórek -
publikacja opisuje 570 próbek z 11 odwiertów.

Porównywane są trzy sposoby odtworzenia pola przepuszczalności:

  1. Jedna globalna korelacja k(phi) dopasowana do rdzeni - podejście
     klasyczne, które publikacja przytacza jako punkt odniesienia (Fig. 2d,
     R^2 = 0,7 dla całego zbioru).
  2. Jednostki hydrauliczne: podział rdzeni wg log(FZI), średnie FZI
     w jednostce, przepuszczalność ze wzoru (5).
  3. Globalne elementy hydrauliczne (GHE) wg Corbett i in. (2003):
     przypisanie rdzeni do szablonu o z góry ustalonych wartościach FZI.
     Metoda nie wymaga dopasowywania czegokolwiek, więc powinna być
     odporna na małą liczbę rdzeni.

Każde z pól trafia następnie do symulatora, a wynik porównywany jest
z modelem odniesienia zbudowanym na prawdziwym polu przepuszczalności.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from statyka import (ModelStatycznyZloza, oblicz_fzi, _dopasuj_w_skali_log,
                     przewidz_k, k_z_fzi, przypisz_ghe)
from pvt import TermodynamikaGazu
from symulator_fvm import SymulatorFVM
from generator_zloza import generuj_zloze_Z, FZI_ZLOZE_Z

H = 20.0
P_POCZ = 170.0
CZAS_DNI = 1825.0
DT_DNI = 15.0
LICZBA_HU = 6                      # jak w publikacji

# Odwierty pracują na zadanym wydatku (półka produkcyjna). Jest to warunek
# konieczny, aby eksperyment w ogóle mierzył wpływ przepuszczalności:
# przy sterowaniu samym ciśnieniem dennym złoże zostaje po pięciu latach
# sczerpane do P_wf, a produkcja skumulowana staje się wielkością czysto
# objętościową, zależną od porowatości, nie od przepuszczalności. Sprawdzono
# to pomiarem - dziesięciokrotna zmiana całego pola k zmieniała produkcję
# o mniej niż 0,1 %. Przy sterowaniu wydatkiem przepuszczalność decyduje
# natomiast o długości półki, która zmienia się wtedy o kilkadziesiąt procent.
WYDATEK_ODWIERTU = 1_200_000.0     # [m3 std/d]
ODWIERTY = [{'nazwa': 'O-1', 'i': 8, 'j': 8, 'p_dna': 50.0, 'q_max': WYDATEK_ODWIERTU},
            {'nazwa': 'O-2', 'i': 21, 'j': 21, 'p_dna': 50.0, 'q_max': WYDATEK_ODWIERTU},
            {'nazwa': 'O-3', 'i': 8, 'j': 21, 'p_dna': 50.0, 'q_max': WYDATEK_ODWIERTU},
            {'nazwa': 'O-4', 'i': 21, 'j': 8, 'p_dna': 50.0, 'q_max': WYDATEK_ODWIERTU}]


def dlugosc_polki(symulator):
    """Średni czas pracy odwiertów na zadanym wydatku [dni]."""
    return float(np.mean([sum(t == 'wydatek' for t in historia) * DT_DNI
                          for historia in symulator.historia_trybu]))


def komorki_rdzeniowe(nx, ny, liczba_odwiertow, promien, rng):
    """
    Wybiera komórki z danymi rdzeniowymi: otoczenia losowo rozmieszczonych
    odwiertów rdzeniowanych. Odwzorowuje rzeczywistość, w której rdzenie
    pochodzą z odwiertów, a nie z przypadkowych punktów złoża.
    """
    ii, jj = np.indices((nx, ny))
    maska = np.zeros((nx, ny), dtype=bool)
    for _ in range(liczba_odwiertow):
        i0 = rng.integers(2, nx - 2)
        j0 = rng.integers(2, ny - 2)
        maska |= (ii - i0)**2 + (jj - j0)**2 <= promien**2
    return np.flatnonzero(maska.ravel())


def pole_globalne(phi, k, tren):
    """Jedna korelacja log k = a*phi + b dopasowana do wszystkich rdzeni."""
    return przewidz_k(_dopasuj_w_skali_log(phi[tren], k[tren]), phi)


def pole_hu(phi, k, tren, nx, ny, liczba_hu=LICZBA_HU, random_state=0):
    """
    Metoda jednostek hydraulicznych wg Ha Quang Man i in. (2021):
    podział rdzeni wg log(FZI), średnie FZI w każdej jednostce, przypisanie
    jednostek pozostałym komórkom i wyznaczenie k ze wzoru (5).
    """
    _, _, fzi_tren = oblicz_fzi(phi[tren], k[tren], m=1.0)
    log_fzi = np.log10(np.maximum(fzi_tren, 1e-8)).reshape(-1, 1)
    km = KMeans(n_clusters=liczba_hu, n_init=10, random_state=random_state)
    etykiety_tren = km.fit_predict(log_fzi)
    # Średnie FZI jednostki - "mean FZI values were then used to construct
    # the porosity-permeability relationship within each HFU"
    srednie_fzi = np.array([fzi_tren[etykiety_tren == c].mean()
                            if (etykiety_tren == c).any() else np.nan
                            for c in range(liczba_hu)])

    ii, jj = np.indices((nx, ny))
    cechy = np.column_stack([(phi - phi.mean()) / phi.std(),
                             (ii.ravel() - ii.mean()) / ii.std(),
                             (jj.ravel() - jj.mean()) / jj.std()])
    n_sasiadow = min(5, len(tren))
    etykiety = KNeighborsClassifier(n_neighbors=n_sasiadow).fit(
        cechy[tren], etykiety_tren).predict(cechy)
    fzi_komorek = np.where(np.isnan(srednie_fzi[etykiety]),
                           np.nanmean(srednie_fzi), srednie_fzi[etykiety])
    return np.clip(k_z_fzi(fzi_komorek, phi), 1e-3, 1e4)


def pole_ghe(phi, k, tren, nx, ny, random_state=0):
    """
    Metoda globalnych elementów hydraulicznych wg Corbett i in. (2003):
    rdzenie przypisywane są do szablonu o ustalonych wartościach FZI, więc
    nie dopasowuje się żadnych współczynników.
    """
    _, _, fzi_tren = oblicz_fzi(phi[tren], k[tren], m=1.0)
    numery_tren, _ = przypisz_ghe(fzi_tren)

    ii, jj = np.indices((nx, ny))
    cechy = np.column_stack([(phi - phi.mean()) / phi.std(),
                             (ii.ravel() - ii.mean()) / ii.std(),
                             (jj.ravel() - jj.mean()) / jj.std()])
    n_sasiadow = min(5, len(tren))
    numery = KNeighborsClassifier(n_neighbors=n_sasiadow).fit(
        cechy[tren], numery_tren).predict(cechy)
    from statyka import FZI_GHE
    fzi_komorek = np.array(FZI_GHE)[numery - 1]
    return np.clip(k_z_fzi(fzi_komorek, phi), 1e-3, 1e4)


def symuluj(df, k_pole, pvt):
    zloze = ModelStatycznyZloza(100.0, 100.0)
    df = df.copy()
    df['Przepuszczalnosc'] = k_pole
    zloze.wczytaj_dane(df)
    s = SymulatorFVM(zloze, pvt, P_POCZ, ODWIERTY, h=H)
    s.uruchom(CZAS_DNI, DT_DNI)
    return s


def nrmse(odniesienie, model):
    odniesienie, model = np.asarray(odniesienie), np.asarray(model)
    return float(np.sqrt(np.mean((model - odniesienie)**2)) / np.ptp(odniesienie) * 100)


METODY = {'globalna': pole_globalne, 'HU': pole_hu, 'GHE': pole_ghe}


def main(liczba_realizacji=8, liczba_odwiertow_rdzeniowych=6, promien_rdzeni=1):
    pvt = TermodynamikaGazu(0.58, 60.0)
    wiersze = []

    for r in range(liczba_realizacji):
        df = generuj_zloze_Z(seed=62021 + r)
        phi = df['Porowatosc'].to_numpy()
        k_prawdziwe = df['Przepuszczalnosc'].to_numpy()
        nx = ny = int(np.sqrt(len(df)))

        odniesienie = symuluj(df, k_prawdziwe, pvt)
        gp_odn = odniesienie.produkcja_skumulowana
        polka_odn = dlugosc_polki(odniesienie)

        rng = np.random.default_rng(500 + r)
        tren = komorki_rdzeniowe(nx, ny, liczba_odwiertow_rdzeniowych, promien_rdzeni, rng)

        wiersz = {'realizacja': r, 'n_rdzeni': len(tren),
                  'gp_odniesienia_mln': gp_odn / 1e6, 'polka_odniesienia_dni': polka_odn}
        for nazwa, funkcja in METODY.items():
            k_pole = (funkcja(phi, k_prawdziwe, tren) if nazwa == 'globalna'
                      else funkcja(phi, k_prawdziwe, tren, nx, ny))
            s = symuluj(df, k_pole, pvt)
            wiersz[f'rmse_k_{nazwa}'] = float(np.sqrt(np.mean(
                (np.log10(k_pole) - np.log10(k_prawdziwe))**2)))
            wiersz[f'blad_gp_{nazwa}'] = (s.produkcja_skumulowana - gp_odn) / gp_odn * 100
            wiersz[f'blad_polki_{nazwa}'] = ((dlugosc_polki(s) - polka_odn) / polka_odn * 100
                                             if polka_odn > 0 else np.nan)
            wiersz[f'nrmse_q_{nazwa}'] = nrmse(odniesienie.historia_wydatku,
                                               s.historia_wydatku)
        wiersze.append(wiersz)
        print(f"  realizacja {r + 1}/{liczba_realizacji} ({len(tren)} komórek rdzeniowych)")

    return pd.DataFrame(wiersze)


def badanie_warunkow(liczba_realizacji=5):
    """
    Bada, od czego zależy skuteczność metody jednostek hydraulicznych:
    od wielkości stref geologicznych względem rozstawu odwiertów oraz od
    liczby odwiertów rdzeniowanych. Ocena prowadzona jest na polu
    przepuszczalności (RMSE log k), bez uruchamiania symulacji.
    """
    from scipy.ndimage import label

    print("\nA. WPŁYW WIELKOŚCI STREF GEOLOGICZNYCH (6 odwiertów rdzeniowanych)\n")
    print("  zasięg stref | śr. wielkość strefy | globalna |   HU   | przewaga HU")
    wyniki_stref = []
    for zasieg in (1.0, 2.5, 5.0, 8.0, 12.0):
        bg, bh, rozmiary = [], [], []
        for r in range(liczba_realizacji):
            df = generuj_zloze_Z(seed=62021 + r, zasieg_hu=zasieg)
            phi, k = df['Porowatosc'].to_numpy(), df['Przepuszczalnosc'].to_numpy()
            hu = df['HU_zrodlowa'].to_numpy().reshape(30, 30)
            rozmiary.append(900 / sum(label(hu == u)[1] for u in np.unique(hu)))
            tren = komorki_rdzeniowe(30, 30, 6, 1, np.random.default_rng(500 + r))
            blad = lambda kp: float(np.sqrt(np.mean((np.log10(kp) - np.log10(k))**2)))
            bg.append(blad(pole_globalne(phi, k, tren)))
            bh.append(blad(pole_hu(phi, k, tren, 30, 30)))
        przewaga = (np.mean(bg) - np.mean(bh)) / np.mean(bg) * 100
        wyniki_stref.append((zasieg, np.mean(rozmiary), np.mean(bg), np.mean(bh), przewaga))
        print(f"      {zasieg:4.1f}     |    {np.mean(rozmiary):5.1f} komórek    |  "
              f"{np.mean(bg):.3f}   | {np.mean(bh):.3f}  |  {przewaga:+6.1f} %")
    print("  (odległość między odwiertami rdzeniowanymi: ok. 10-12 komórek)")

    print("\nB. WPŁYW LICZBY ODWIERTÓW RDZENIOWANYCH (strefy ok. 117 komórek)\n")
    print("  odwiertów | komórek | globalna |   HU   |  GHE   | przewaga HU | przewaga GHE")
    for n_odw, promien in ((3, 1), (6, 1), (11, 1), (11, 2), (20, 2)):
        bg, bh, bghe, liczby = [], [], [], []
        for r in range(liczba_realizacji):
            df = generuj_zloze_Z(seed=62021 + r, zasieg_hu=8.0)
            phi, k = df['Porowatosc'].to_numpy(), df['Przepuszczalnosc'].to_numpy()
            tren = komorki_rdzeniowe(30, 30, n_odw, promien, np.random.default_rng(500 + r))
            liczby.append(len(tren))
            blad = lambda kp: float(np.sqrt(np.mean((np.log10(kp) - np.log10(k))**2)))
            bg.append(blad(pole_globalne(phi, k, tren)))
            bh.append(blad(pole_hu(phi, k, tren, 30, 30)))
            bghe.append(blad(pole_ghe(phi, k, tren, 30, 30)))
        print(f"     {n_odw:2d}     |   {np.mean(liczby):3.0f}   |  {np.mean(bg):.3f}   | "
              f"{np.mean(bh):.3f}  | {np.mean(bghe):.3f}  |   "
              f"{(np.mean(bg) - np.mean(bh)) / np.mean(bg) * 100:+6.1f} %  |   "
              f"{(np.mean(bg) - np.mean(bghe)) / np.mean(bg) * 100:+6.1f} %")
    return wyniki_stref


def pulap_metody(liczba_realizacji=5):
    """
    Wyznacza pułap metody: błąd, jaki dałaby znajomość przynależności każdej
    komórki do jednostki. Różnica względem wyniku rzeczywistego pokazuje, ile
    traci się na przypisywaniu jednostek komórkom bez rdzeni.
    """
    print("\nC. PUŁAP METODY (6 odwiertów rdzeniowanych, strefy ok. 24 komórek)\n")
    wyniki = {'korelacja globalna': [], 'HU (klasyfikacja z phi i położenia)': [],
              'HU (przynależność znana)': [], 'pułap: prawdziwe FZI jednostek': []}
    for r in range(liczba_realizacji):
        df = generuj_zloze_Z(seed=62021 + r)
        phi, k = df['Porowatosc'].to_numpy(), df['Przepuszczalnosc'].to_numpy()
        hu_prawda = df['HU_zrodlowa'].to_numpy()
        tren = komorki_rdzeniowe(30, 30, 6, 1, np.random.default_rng(500 + r))
        blad = lambda kp: float(np.sqrt(np.mean((np.log10(kp) - np.log10(k))**2)))

        srednie = {}
        for hu in np.unique(hu_prawda[tren]):
            maska = hu_prawda[tren] == hu
            _, _, f = oblicz_fzi(phi[tren][maska], k[tren][maska], 1.0)
            srednie[hu] = float(f.mean())
        domyslne = float(np.mean(list(srednie.values())))

        wyniki['korelacja globalna'].append(blad(pole_globalne(phi, k, tren)))
        wyniki['HU (klasyfikacja z phi i położenia)'].append(blad(pole_hu(phi, k, tren, 30, 30)))
        wyniki['HU (przynależność znana)'].append(blad(k_z_fzi(
            np.array([srednie.get(h, domyslne) for h in hu_prawda]), phi)))
        wyniki['pułap: prawdziwe FZI jednostek'].append(blad(k_z_fzi(
            np.array([FZI_ZLOZE_Z[h - 1] for h in hu_prawda]), phi)))
    for nazwa, wartosci in wyniki.items():
        print(f"  {nazwa:<38} RMSE log k = {np.mean(wartosci):.3f}")


if __name__ == '__main__':
    print("Złoże gazu Z (Ha Quang Man i in., 2021), 6 jednostek hydraulicznych")
    print(f"{len(ODWIERTY)} odwierty eksploatacyjne na półce "
          f"{WYDATEK_ODWIERTU / 1e3:.0f} tys. m³/d, {CZAS_DNI:.0f} dni\n")
    tabela = main()

    print("\n" + "=" * 76)
    print("WYNIKI - średnia z 8 realizacji (w nawiasie odchylenie standardowe)\n")
    print(f"Liczba komórek z rdzeniami: {tabela['n_rdzeni'].mean():.0f} z 900 "
          f"({tabela['n_rdzeni'].mean() / 9:.1f} %)\n")
    print(f"Półka produkcyjna w modelu odniesienia: "
          f"{tabela['polka_odniesienia_dni'].mean():.0f} dni\n")
    print(f"{'metoda':>10} | {'RMSE log k':>16} | {'błąd półki [%]':>16} | "
          f"{'NRMSE krzywej [%]':>17} | {'błąd Gp [%]':>13}")
    print("-" * 76)
    for nazwa in METODY:
        rk = tabela[f'rmse_k_{nazwa}']
        bp = tabela[f'blad_polki_{nazwa}'].abs()
        nq = tabela[f'nrmse_q_{nazwa}']
        bg = tabela[f'blad_gp_{nazwa}'].abs()
        print(f"{nazwa:>10} | {rk.mean():7.3f} ({rk.std():5.3f}) | "
              f"{bp.mean():7.2f} ({bp.std():5.2f}) | {nq.mean():8.2f} ({nq.std():5.2f}) | "
              f"{bg.mean():5.2f} ({bg.std():4.2f})")

    print("\nPrzewaga nad korelacją globalną:")
    for nazwa in ('HU', 'GHE'):
        for opis, kol in (("pole przepuszczalności", 'rmse_k'),
                          ("długość półki        ", 'blad_polki'),
                          ("krzywa wydobycia     ", 'nrmse_q')):
            odn = tabela[f'{kol}_globalna'].abs().mean()
            war = tabela[f'{kol}_{nazwa}'].abs().mean()
            print(f"   {nazwa:>4} {opis}: {(odn - war) / odn * 100:+6.1f} %")

    tabela.to_csv('wyniki_badania_hu.csv', index=False)

    pulap_metody()
    badanie_warunkow()
    print("\nSzczegóły symulacji zapisano w wyniki_badania_hu.csv")