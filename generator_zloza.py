"""
generator_zloza.py - generatory syntetycznych modeli statycznych złoża.

Moduł udostępnia dwa generatory:

1. generuj_zloze() - model dowolny, o zadawanej liczbie facji i parametrach
   rozkładu. Służy do badania zachowania metody w kontrolowanych warunkach,
   na przykład do porównania złoża jednofacjalnego z wielofacjalnym.

2. generuj_zloze_realistyczne() - model skalibrowany do danych rdzeniowych
   opisanych w literaturze, opisany szczegółowo poniżej.

Pole losowe o korelacji przestrzennej uzyskiwane jest przez wygładzenie szumu
białego filtrem Gaussa - prosty odpowiednik symulacji geostatystycznej
o zadanym zasięgu korelacji.

MODEL SKALIBROWANY - CO POCHODZI Z DANYCH RZECZYWISTYCH
Parametry statystyczne pochodzą z pracy Ellabad, Corbett i Straub (2001,
SCA 2001-50), opisującej triasowe złoże klastyczne (osady rzeczne) w Afryce
Północnej, scharakteryzowane na podstawie rdzeni i testów odwiertu:

  * cztery jednostki hydrauliczne o wskaźnikach FZI równych
    2,509 / 1,233 / 0,685 / 0,323 um (Fig. 2 w cytowanej pracy),
  * zależność k(Phi) w obrębie każdej jednostki wynikająca wprost z równania
    Amaefule i in. (1993), odtwarzająca wykres poro-perm z Fig. 3
    (Phi do 0,25, k od ok. 0,01 do ok. 100 mD),
  * udziały jednostek dobrane tak, aby odtworzyć wykres Lorenza z Fig. 4:
    jednostka najlepsza (HU1) wnosi ok. 70 % pojemności przepływowej przy
    ok. 20 % pojemności magazynowej,
  * rozkład przestrzenny typu pikselowego o dużej zmienności, odpowiadający
    wariantowi "very heterogeneous", który w cytowanej pracy jako jedyny dał
    zgodność z danymi z testów odwiertu.

CO JEST WYMYŚLONE
Konkretne położenie jednostek w siatce oraz realizacja szumu. Praca źródłowa
podaje statystyki i wnioski, nie udostępnia pola przepuszczalności, a autorzy
sami zaznaczają, że geologiczne pochodzenie gruboziarnistych stref nie jest
w tym złożu oczywiste i modelowali je stochastycznie.

W PRACY INŻYNIERSKIEJ należy to opisać wprost: dane są syntetyczne, lecz ich
statystyki (rozkład FZI, zależność k-Phi, podział pojemności przepływowej
i magazynowej) odtwarzają udokumentowane złoże. Nie wolno przedstawiać ich
jako pomiarów.

UWAGA DOTYCZĄCA NUMERACJI
Praca źródłowa numeruje HU1 jako jednostkę najlepszą. Moduł statyka.py
numeruje odwrotnie (HU 1 = najniższe FZI), więc kolumna HU_zrodlowa w pliku
wyjściowym używa konwencji publikacji, a etykiety wyznaczone przez K-Means
będą względem niej odwrócone. Jest to wyłącznie kwestia oznaczeń.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

def _pole_skorelowane(rng, nx, ny, sigma_x, sigma_y=None):
    """
    Pole losowe o zadanej korelacji przestrzennej, o średniej 0 i odchyleniu 1.
    Podanie różnych zasięgów dla osi daje strefy wydłużone w jednym kierunku.
    """
    z = gaussian_filter(rng.standard_normal((nx, ny)), (sigma_x, sigma_y or sigma_x),
                        mode='reflect')
    return (z - z.mean()) / z.std()


# --------------------------------------------------- generator ogólny
def generuj_zloze(nx=30, ny=30, dx=100.0, dy=100.0, seed=42,
                  phi_sr=0.15, phi_odch=0.03, zasieg_phi=4.0,
                  liczba_facji=1, rozrzut_log_k=0.15, zasieg_facji=5.0,
                  k_ref=10.0, nachylenie=25.0, kontrast_facji=0.6):
    """
    Zwraca ramkę danych z kolumnami I, J, x, y, Porowatosc, Przepuszczalnosc
    oraz (gdy liczba_facji > 1) Facja - prawdziwy podział, do oceny klasyfikacji.

    Parametry:
      phi_sr, phi_odch   - średnia i odchylenie porowatości
      zasieg_phi         - zasięg korelacji przestrzennej porowatości w blokach
      liczba_facji       - liczba facji o różnych zależnościach k(phi)
      kontrast_facji     - rozstaw poziomów log10(k) między facjami
      rozrzut_log_k      - rozrzut log10(k) wewnątrz facji (szum pomiarowy)
      k_ref, nachylenie  - k = k_ref * exp(nachylenie * (phi - phi_sr)) dla facji bazowej

    Kolejność wierszy: I rośnie najwolniej, J najszybciej (zgodnie z układem
    tablic (nx, ny) w module statycznym).
    """
    rng = np.random.default_rng(seed)
    phi = np.clip(phi_sr + phi_odch * _pole_skorelowane(rng, nx, ny, zasieg_phi), 0.04, 0.32)

    if liczba_facji <= 1:
        log_k = np.log10(k_ref) + nachylenie * (phi - phi_sr) / np.log(10.0)
        facja = np.ones((nx, ny), dtype=int)
    else:
        # Facje jako strefy przestrzenne wycięte z gładkiego pola losowego
        pole_facji = _pole_skorelowane(rng, nx, ny, zasieg_facji)
        progi = np.quantile(pole_facji, np.linspace(0, 1, liczba_facji + 1)[1:-1])
        facja = np.digitize(pole_facji, progi) + 1

        log_k = np.zeros((nx, ny))
        for f in range(1, liczba_facji + 1):
            maska = facja == f
            przesuniecie = (f - (liczba_facji + 1) / 2.0) * kontrast_facji
            # każda facja ma własny poziom i własne nachylenie zależności k(phi)
            nachylenie_f = nachylenie * (0.6 + 0.4 * f / liczba_facji)
            log_k[maska] = (np.log10(k_ref) + przesuniecie
                            + nachylenie_f * (phi[maska] - phi_sr) / np.log(10.0))

    log_k = log_k + rozrzut_log_k * rng.standard_normal((nx, ny))
    k = np.clip(10.0**log_k, 1e-2, 1e4)

    ii, jj = np.indices((nx, ny))
    dane = {
        'I': ii.ravel(), 'J': jj.ravel(),
        'x': (ii.ravel() + 0.5) * dx, 'y': (jj.ravel() + 0.5) * dy,
        'Porowatosc': phi.ravel(), 'Przepuszczalnosc': k.ravel(),
    }
    if liczba_facji > 1:
        dane['Facja'] = facja.ravel()
    return pd.DataFrame(dane)


# -------------------------- generator skalibrowany do danych z publikacji
# Ellabad i in. (2001), Fig. 2 - wskaźniki FZI czterech jednostek [um]
FZI_JEDNOSTEK = (2.509, 1.233, 0.685, 0.323)

# Udziały jednostek w liczbie komórek, dobrane do warunku z wykresu Lorenza
# (HU1: ok. 20 % pojemności magazynowej i ok. 70 % pojemności przepływowej).
# Ponieważ pola porowatości i jednostek są niezależne, udział jednostki w
# pojemności przepływowej wynosi w przybliżeniu c_i*FZI_i^2 / suma(c*FZI^2),
# co pozwoliło dobrać te wartości rachunkowo, a nie przez zgadywanie.
UDZIALY_DOMYSLNE = (0.20, 0.23, 0.37, 0.20)

STALA_RQI = 0.0314          # RQI = 0,0314 sqrt(k/phi), k w mD

# ---- wariant węglanowy: Nooruddin i in. (2011), SPE 149047 --------------
# Złoże węglanowe we wschodniej prowincji Arabii Saudyjskiej, siedem jednostek
# hydraulicznych (Tabela 2), wykładnik cementacji m = 2,00 wyznaczony
# z pomiarów właściwości elektrycznych rdzeni. Wartości FZI_m odczytane
# w przybliżeniu z histogramu (Fig. 1) i wykresu prawdopodobieństwa
# skumulowanego (Fig. 2): rozkład obejmuje mniej więcej zakres 1-40
# z maksimum w okolicy 4-6. Udziały dobrane tak, aby odtworzyć ten kształt.
FZI_WEGLANOWE = (40.0, 22.0, 13.0, 8.0, 5.0, 3.2, 2.0)
UDZIALY_WEGLANOWE = (0.03, 0.07, 0.15, 0.25, 0.25, 0.17, 0.08)
M_WEGLANOWE = 2.0


def k_z_fzi(fzi, phi, m=1.0):
    """
    Odwrócenie zależności Amaefule i in. (1993) w wariancie zmodyfikowanym
    Nooruddina i in. (2011):
        RQI = 0,0314 sqrt(k/phi) = FZI_m * phi_z * phi^(m-1) = FZI_m * phi^m/(1-phi)
    skąd      k = (FZI_m / 0,0314)^2 * phi^(2m+1) / (1 - phi)^2   [mD]
    Dla m = 1 wzór sprowadza się do postaci klasycznej.
    """
    return (fzi / STALA_RQI)**2 * phi**(2.0 * m + 1.0) / (1.0 - phi)**2


def generuj_zloze_realistyczne(nx=30, ny=30, dx=100.0, dy=100.0, seed=2001,
                               phi_sr=0.125, phi_odch=0.035, phi_min=0.03, phi_max=0.25,
                               zasieg_phi=3.5, zasieg_hu=2.0, anizotropia_stref=2.5,
                               rozrzut_log_k=0.18, udzialy=UDZIALY_DOMYSLNE,
                               fzi=FZI_JEDNOSTEK, m=1.0):
    """
    Zwraca ramkę danych z kolumnami:
      I, J, x, y, Porowatosc, Przepuszczalnosc, HU_zrodlowa, FZI_zrodlowe

    zasieg_hu i anizotropia_stref sterują geometrią stref: wartości większe od 1
    w anizotropii wydłużają strefy w kierunku x, co odpowiada kierunkowości
    osadów rzecznych. Mniejszy zasieg_hu daje model bardziej "pikselowy".
    """
    if abs(sum(udzialy) - 1.0) > 1e-6:
        raise ValueError("Udziały jednostek muszą sumować się do 1.")
    if len(udzialy) != len(fzi):
        raise ValueError("Liczba udziałów musi odpowiadać liczbie wartości FZI.")

    rng = np.random.default_rng(seed)

    # Porowatość: pole skorelowane przestrzennie, przycięte do zakresu z Fig. 3
    phi = np.clip(phi_sr + phi_odch * _pole_skorelowane(rng, nx, ny, zasieg_phi),
                  phi_min, phi_max)

    # Przypisanie jednostek: pole skorelowane cięte kwantylami wg zadanych udziałów.
    # Strefy wydłużone w osi x odwzorowują kierunkowość osadów rzecznych.
    pole_hu = _pole_skorelowane(rng, nx, ny, zasieg_hu * anizotropia_stref, zasieg_hu)

    # Ortogonalizacja względem pola porowatości. Przy 900 komórkach i polach
    # skorelowanych przestrzennie liczba niezależnych stref jest niewielka, więc
    # pojedyncza realizacja potrafi wytworzyć przypadkową korelację jednostek
    # z porowatością; ponieważ k rośnie z phi w trzeciej potędze, zaburzałoby
    # to podział pojemności przepływowej. Usunięcie składowej wspólnej czyni
    # rozmieszczenie jednostek niezależnym od porowatości, co odpowiada
    # obserwacji autorów, że zależność k(Phi) jest odrębna dla każdej jednostki.
    phi_std = (phi - phi.mean()) / phi.std()
    beta = float(np.mean(pole_hu * phi_std))
    pole_hu = pole_hu - beta * phi_std
    pole_hu = (pole_hu - pole_hu.mean()) / pole_hu.std()

    # HU1 (najlepsza) odpowiada najwyższym wartościom pola, dlatego progi liczone
    # są z udziałów w odwrotnej kolejności - od jednostki najgorszej.
    progi = np.quantile(pole_hu, np.cumsum(udzialy[::-1])[:-1])
    hu = len(udzialy) - np.digitize(pole_hu, progi)

    # Przepuszczalność z zależności Amaefule dla FZI danej jednostki plus rozrzut
    k = np.zeros((nx, ny))
    fzi_pole = np.zeros((nx, ny))
    for numer, wartosc_fzi in enumerate(fzi, start=1):
        maska = hu == numer
        fzi_pole[maska] = wartosc_fzi
        k[maska] = k_z_fzi(wartosc_fzi, phi[maska], m=m)
    k *= 10.0**(rozrzut_log_k * rng.standard_normal((nx, ny)))
    k = np.clip(k, 1e-3, 1e4)

    ii, jj = np.indices((nx, ny))
    return pd.DataFrame({
        'I': ii.ravel(), 'J': jj.ravel(),
        'x': (ii.ravel() + 0.5) * dx, 'y': (jj.ravel() + 0.5) * dy,
        'Porowatosc': phi.ravel(),
        'Przepuszczalnosc': k.ravel(),
        'HU_zrodlowa': hu.ravel(),
        'FZI_zrodlowe': fzi_pole.ravel(),
        'm_zrodlowe': np.full(nx * ny, float(m)),
    })


# ---- złoże gazu Z: Ha Quang Man, Le i Jarzyna (2021) -------------------
# Mioceńskie złoże gazu w zapadlisku przedkarpackim w południowo-wschodniej
# Polsce, osady deltowe, 570 próbek rdzeniowych z 11 odwiertów z głębokości
# 253-1154 m. Autorzy wydzielili sześć jednostek hydraulicznych na podstawie
# wykresu prawdopodobieństwa log(FZI) oraz klasteryzacji hierarchicznej
# metodą Warda. Wartości poniżej pochodzą z Tabeli 2 publikacji.
#
#                       HU1     HU2     HU3     HU4     HU5     HU6
FZI_ZLOZE_Z        = (0.283,  0.734,  1.379,  2.100,  3.510,  5.850)   # średnie FZI [um]
UDZIALY_ZLOZE_Z    = (28/570, 58/570, 89/570, 117/570, 214/570, 64/570)
PHI_SR_ZLOZE_Z     = (0.160,  0.210,  0.240,  0.257,  0.260,  0.270)   # średnia porowatość
PHI_MIN_ZLOZE_Z    = (0.070,  0.078,  0.150,  0.203,  0.189,  0.229)
PHI_MAX_ZLOZE_Z    = (0.233,  0.251,  0.292,  0.315,  0.320,  0.306)
# Współczynniki determinacji k(FZI_sr) względem k rdzeniowego, Tabela 2
R2_ZLOZE_Z         = (0.728,  0.888,  0.645,  0.743,  0.603,  0.411)


def generuj_zloze_Z(nx=30, ny=30, dx=100.0, dy=100.0, seed=62021,
                    zasieg_hu=2.5, anizotropia_stref=1.8, rozrzut_log_k=None):
    """
    Model statyczny odtwarzający statystyki złoża gazu Z opisanego przez
    Ha Quang Man, Le i Jarzynę (2021) - mioceńskie osady deltowe zapadliska
    przedkarpackiego w Polsce.

    Z publikacji pochodzą: liczba jednostek (sześć), ich średnie wskaźniki FZI,
    liczności, zakresy i średnie porowatości (Tabela 2) oraz zależność
    k = 1014,24 FZI^2 phi^3/(1-phi)^2 (wzór 5), identyczna z klasycznym
    równaniem Kozeny-Carmana w ujęciu Amaefule i in. (1993).

    Rozrzut przepuszczalności wewnątrz każdej jednostki dobrano tak, aby
    odtworzyć współczynniki determinacji z Tabeli 2 (od 0,41 dla HU6 do 0,89
    dla HU2) - inaczej niż w pozostałych generatorach, gdzie jest on jednakowy.
    Dzięki temu dane mają realistyczny, niejednorodny rozrzut: jednostki
    o najwyższej przepuszczalności są opisane najsłabiej.

    Wymyślone pozostaje wyłącznie rozmieszczenie jednostek w siatce; publikacja
    podaje statystyki rdzeni, a nie mapę złoża.
    """
    rng = np.random.default_rng(seed)
    n_hu = len(FZI_ZLOZE_Z)

    # Rozmieszczenie jednostek: pole skorelowane cięte kwantylami wg liczności
    pole = _pole_skorelowane(rng, nx, ny, zasieg_hu * anizotropia_stref, zasieg_hu)
    progi = np.quantile(pole, np.cumsum(UDZIALY_ZLOZE_Z)[:-1])
    hu = np.digitize(pole, progi) + 1

    # Porowatość: w każdej jednostce własny rozkład wg Tabeli 2
    phi = np.zeros((nx, ny))
    for numer in range(1, n_hu + 1):
        maska = hu == numer
        if not maska.any():
            continue
        sr = PHI_SR_ZLOZE_Z[numer - 1]
        lo, hi = PHI_MIN_ZLOZE_Z[numer - 1], PHI_MAX_ZLOZE_Z[numer - 1]
        odch = (hi - lo) / 4.0          # zakres ok. +/- 2 odchylenia
        phi[maska] = np.clip(sr + odch * rng.standard_normal(int(maska.sum())), lo, hi)

    # Przepuszczalność ze wzoru (5) plus rozrzut odtwarzający R^2 z Tabeli 2.
    # Dla regresji log-log zachodzi R^2 = 1 - s_reszt^2 / s_calk^2, więc rozrzut
    # wynika z rozpiętości log k w jednostce i docelowego R^2.
    k = np.zeros((nx, ny))
    fzi_pole = np.zeros((nx, ny))
    for numer in range(1, n_hu + 1):
        maska = hu == numer
        if not maska.any():
            continue
        wartosc_fzi = FZI_ZLOZE_Z[numer - 1]
        fzi_pole[maska] = wartosc_fzi
        k_wzor = k_z_fzi(wartosc_fzi, phi[maska], m=1.0)
        if rozrzut_log_k is None:
            r2 = R2_ZLOZE_Z[numer - 1]
            rozrzut = np.std(np.log10(k_wzor)) * np.sqrt(max(1.0 / max(r2, 0.05) - 1.0, 0.0))
        else:
            rozrzut = rozrzut_log_k
        k[maska] = k_wzor * 10.0**(rozrzut * rng.standard_normal(int(maska.sum())))
    k = np.clip(k, 1e-3, 1e4)

    ii, jj = np.indices((nx, ny))
    return pd.DataFrame({
        'I': ii.ravel(), 'J': jj.ravel(),
        'x': (ii.ravel() + 0.5) * dx, 'y': (jj.ravel() + 0.5) * dy,
        'Porowatosc': phi.ravel(),
        'Przepuszczalnosc': k.ravel(),
        'HU_zrodlowa': hu.ravel(),
        'FZI_zrodlowe': fzi_pole.ravel(),
        'm_zrodlowe': np.full(nx * ny, 1.0),
    })


def generuj_zloze_weglanowe(nx=30, ny=30, dx=100.0, dy=100.0, seed=149047,
                            phi_sr=0.20, phi_odch=0.065, phi_min=0.05, phi_max=0.40,
                            zasieg_phi=3.0, zasieg_hu=2.5, anizotropia_stref=1.0,
                            rozrzut_log_k=0.12):
    """
    Model skalibrowany do złoża węglanowego opisanego przez Nooruddina i in.
    (2011): siedem jednostek hydraulicznych i wykładnik cementacji m = 2,00.

    W odróżnieniu od wariantu piaskowcowego strefy nie są wydłużone w jednym
    kierunku (anizotropia_stref = 1), ponieważ w złożu węglanowym rozkład
    jednostek wynika z procesów diagenetycznych, a nie z kierunkowości osadów
    rzecznych.

    Ten zbiór służy pokazaniu, że kryterium doboru wykładnika m wykrywa
    podwyższoną wartość tam, gdzie faktycznie występuje - w przeciwieństwie do
    danych piaskowcowych, opisanych klasycznym FZI.

    O ROZRZUCIE WEWNĄTRZ JEDNOSTEK (rozrzut_log_k)
    Siedem jednostek dzieli w skali log10(FZI_m) odstęp około 0,20, a rozrzut
    wewnątrz jednostki wynosi połowę rozrzutu log10(k). Rozróżnialność zależy
    od ilorazu tych dwóch wielkości i zmierzono ją na trzech realizacjach:

        rozrzut   odstęp/odchylenie   wykryta liczba jednostek   R^2 korelacji
          0,20           2,0                   1-2                   0,96
          0,15           2,7                   2 (czasem 7)          0,97
          0,12           3,4                   7 (powtarzalnie)      0,98
          0,09           4,5                   7                     0,98

    Przyjęto 0,12, czyli najwyższy rozrzut, przy którym podział na siedem
    jednostek jest odtwarzalny. Jest to zarazem idealizacja: w publikacji
    współczynniki R^2 korelacji k-Phi w obrębie jednostek wynosiły około
    0,84-0,96, a więc rozrzut rzeczywistych danych był większy niż przyjęty
    tutaj. Oznacza to, że w prawdziwym złożu liczby jednostek nie da się
    wyznaczyć z samego rozkładu FZI - i rzeczywiście autorzy pracy nie
    wyznaczali jej w ten sposób, lecz z wykresu prawdopodobieństwa
    skumulowanego, opierając się dodatkowo na wcześniejszym podziale na sześć
    elektrofacji. Ustawiając rozrzut na 0,15-0,20 można odtworzyć tę
    trudniejszą sytuację.
    """
    return generuj_zloze_realistyczne(
        nx=nx, ny=ny, dx=dx, dy=dy, seed=seed,
        phi_sr=phi_sr, phi_odch=phi_odch, phi_min=phi_min, phi_max=phi_max,
        zasieg_phi=zasieg_phi, zasieg_hu=zasieg_hu, anizotropia_stref=anizotropia_stref,
        rozrzut_log_k=rozrzut_log_k, udzialy=UDZIALY_WEGLANOWE,
        fzi=FZI_WEGLANOWE, m=M_WEGLANOWE)


def raport_kalibracji(df, h=1.0):
    """
    Porównuje wygenerowane pole z warunkami podanymi przez Ellabad i in. (2001):
    udział każdej jednostki w pojemności magazynowej (phi*h) i przepływowej (k*h).
    """
    phi, k, hu = df['Porowatosc'].to_numpy(), df['Przepuszczalnosc'].to_numpy(), df['HU_zrodlowa'].to_numpy()
    wiersze = []
    for numer in sorted(np.unique(hu)):
        maska = hu == numer
        wiersze.append({
            'HU (wg publikacji)': int(numer),
            'FZI [um]': float(df.loc[maska, 'FZI_zrodlowe'].iloc[0]),
            'Udział komórek [%]': 100.0 * maska.mean(),
            'Pojemność magazynowa [%]': 100.0 * phi[maska].sum() / phi.sum(),
            'Pojemność przepływowa [%]': 100.0 * k[maska].sum() / k.sum(),
            'k min [mD]': k[maska].min(),
            'k maks [mD]': k[maska].max(),
        })
    return pd.DataFrame(wiersze)

if __name__ == '__main__':
    # 1. Modele kontrolne: jedna facja i cztery facje
    for nazwa, kwargs in (('zloze_jednofacjalne.csv', dict(liczba_facji=1)),
                          ('zloze_wielofacjalne.csv', dict(liczba_facji=4))):
        df = generuj_zloze(**kwargs)
        df.to_csv(nazwa, index=False)
        print(f"{nazwa}: {len(df)} komórek, phi {df.Porowatosc.min():.3f}-{df.Porowatosc.max():.3f}, "
              f"k {df.Przepuszczalnosc.min():.2f}-{df.Przepuszczalnosc.max():.1f} mD")

    # 2. Modele skalibrowane do danych z publikacji
    for nazwa, funkcja, opis in (
            ('zloze_realistyczne.csv', generuj_zloze_realistyczne,
             'piaskowiec, 4 jednostki, klasyczne FZI (m = 1) - Ellabad i in. (2001)'),
            ('zloze_weglanowe.csv', generuj_zloze_weglanowe,
             'węglan, 7 jednostek, zmodyfikowane FZI (m = 2,00) - Nooruddin i in. (2011)')):
        df = funkcja()
        df.to_csv(nazwa, index=False)
        print(f"\n{nazwa}  ({opis})")
        print(raport_kalibracji(df).to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
        print(f"  Porowatość: {df.Porowatosc.min():.3f} - {df.Porowatosc.max():.3f} "
              f"(średnia {df.Porowatosc.mean():.3f})")
        print(f"  Przepuszczalność: {df.Przepuszczalnosc.min():.2f} - "
              f"{df.Przepuszczalnosc.max():.0f} mD (mediana {df.Przepuszczalnosc.median():.1f} mD)")