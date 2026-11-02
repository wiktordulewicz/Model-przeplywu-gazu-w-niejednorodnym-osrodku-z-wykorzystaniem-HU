"""
statyka.py - statyczny model złoża 2D i jednostki hydrauliczne (HU).

Metodyka (Amaefule i in., 1993, SPE 26436, wzory 3-5):
  RQI   = 0,0314 sqrt(k/phi)        [um]   k w mD, phi jako ułamek
  phi_z = phi / (1 - phi)           [-]    znormalizowany wskaźnik porowatości
  FZI   = RQI / phi_z               [um]

Wariant zmodyfikowany (Nooruddin i in., 2011, SPE 149047, wzory 6-9) wychodzi
od równania Kozeny-Carmana z wykładnikiem cementacji m, w którym zależność
tortuozyjności od porowatości nie jest linearyzowana:
  k = (1 / (f_g a^2 S_vgr^2)) * phi^(2m+1) / (1 - phi)^2
  RQI = FZI_m * phi_z * phi^(m-1)   ->   FZI_m = RQI / (phi_z * phi^(m-1))
Dla m = 1 wzór sprowadza się do klasycznego FZI. Nooruddin i in. przyjęli
m = 2,00 dla badanego złoża węglanowego, wyznaczone z pomiarów elektrycznych.

Podział na jednostki: K-Means na log10(FZI_m), etykiety numerowane rosnąco
według środków klastrów (HU 1 = najniższa jakość zbiornikowa). Zastosowanie
K-Means do klasyfikacji HFU na podstawie FZI opisują m.in. Nguyen Trung i in.
(2024) oraz Heydari Gholanlo (2021).

Korelacje k(phi) w obrębie jednostki dopasowywane są w skali logarytmicznej,
zgodnie z klasyczną postacią log k = a*phi + b (Amaefule i in., 1993, wzór 1;
Ellabad i in., 2001):
  wykładnicza:  log10(k) = log10(a) + b*phi/ln(10)   ->  k = a*exp(b*phi)
  potęgowa:     log10(k) = log10(a) + b*log10(phi)   ->  k = a*phi^b
R^2 raportowane jest w tej samej (logarytmicznej) przestrzeni.

Liczba jednostek dobierana jest metodą łokcia: krzywa dystorsji K-Means
względem liczby klastrów, punkt łokcia wyznaczany algorytmem Kneedle
(Satopää i in., 2011) na krzywej w skali logarytmicznej.

Pełne opisy bibliograficzne znajdują się w pliku BIBLIOGRAFIA.md.

KOLEJNOŚĆ KOMÓREK: tablice mają kształt (nx, ny); indeks pierwszy to oś X,
drugi to oś Y. Domyślnie przyjmuje się, że w pliku CSV najszybciej zmienia się
współrzędna Y (czyli wiersze idą: x0y0, x0y1, ... x0y_n, x1y0, ...). Jeżeli
plik zawiera kolumny I/J albo x/y, kolejność jest odtwarzana z nich i parametr
kolejnosci jest ignorowany.
"""
import threading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

KOLORY_HU = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

PHI_MIN, PHI_MAX = 1e-3, 0.45
K_MIN = 1e-3


def znajdz_lokiec(x, y):
    """
    Wyznacza punkt łokcia krzywej malejącej i wypukłej algorytmem Kneedle
    (Satopää i in., 2011), tym samym, którego używa KElbowVisualizer
    z pakietu Yellowbrick.

    Obie osie są najpierw sprowadzane do przedziału [0, 1]. Cięciwa łącząca
    końce krzywej ma wtedy równanie y = 1 - x, a krzywa wypukła leży pod nią,
    więc liczona jest krzywa różnicowa d = (1 - x) - y, czyli pionowa odległość
    od cięciwy. Jej maksimum wypada w punkcie największej krzywizny, czyli
    w łokciu.

    Zwraca (x_lokcia, y_lokcia, krzywa_roznicowa) albo (None, None, krzywa),
    gdy krzywa nie ma wyraźnego załamania, czyli maksimum wypada na jej końcu
    albo jest zbyt płytkie.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return None, None, np.zeros_like(x)

    rozpietosc_x = np.ptp(x)
    rozpietosc_y = np.ptp(y)
    if rozpietosc_x == 0 or rozpietosc_y == 0:
        return None, None, np.zeros_like(x)

    x_norm = (x - x.min()) / rozpietosc_x
    y_norm = (y - y.min()) / rozpietosc_y
    if y_norm[-1] > y_norm[0]:          # krzywa rosnąca - odwracamy, by liczyć tak samo
        y_norm = 1.0 - y_norm
    roznica = (1.0 - x_norm) - y_norm

    i = int(np.argmax(roznica))
    # Maksimum na krańcu oznacza brak załamania; próg odsiewa krzywe prawie proste
    if i in (0, len(x) - 1) or roznica[i] < 0.05:
        return None, None, roznica
    return float(x[i]), float(y[i]), roznica


def _styl_wykresu():
    dostepne = plt.style.available
    plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in dostepne else 'default')


def liczba_pl(wartosc, format_='.3f'):
    """Liczba z przecinkiem dziesiętnym, zgodnie z zapisem przyjętym w języku polskim."""
    return format(wartosc, format_).replace('.', ',')


class _FormatterPrzecinek(ScalarFormatter):
    """Znaczniki osi z przecinkiem dziesiętnym zamiast kropki."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace('.', ',')

    def get_offset(self):
        return super().get_offset().replace('.', ',')


# Matplotlib nie jest bezpieczny wątkowo: parser wzorów (mathtext) i silnik
# rysowania mają wspólny stan. Streamlit przy każdej zmianie widżetu uruchamia
# skrypt od nowa w osobnym wątku, więc dwa przebiegi mogą rysować jednocześnie,
# co kończy się błędami ParseException lub zawieszeniem. Każde tworzenie
# i zapisywanie rysunku w aplikacji odbywa się dlatego pod tą blokadą.
BLOKADA_RYSOWANIA = threading.RLock()

_MATHRM = None


def _mathrm_dziala():
    """
    Czy mathtext tej instalacji obsługuje polecenie \\mathrm. Niektóre
    zestawienia wersji matplotlib i pyparsing (np. w starszych dystrybucjach
    Anacondy) zgłaszają przy nim ParseException, dlatego sprawdza się to raz,
    przy pierwszym rysunku.
    """
    with BLOKADA_RYSOWANIA:
        return _sprawdz_mathrm()


def _sprawdz_mathrm():
    global _MATHRM
    if _MATHRM is None:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        try:
            # pełna ścieżka rysowania, także dla pogrubionej czcionki opisu osi
            fig = Figure(figsize=(1, 1))
            FigureCanvasAgg(fig)
            ax = fig.add_subplot()
            ax.set_xlabel(r'a, $x_{\mathrm{ab}}$ [-]', fontweight='bold')
            ax.set_ylabel(r'b, $y_{\mathrm{c,d}}$ [-]')
            fig.canvas.draw()
            _MATHRM = True
        except Exception:
            _MATHRM = False
    return _MATHRM


def sym(litera, indeks=None):
    """
    Symbol wielkości fizycznej do opisu wykresu: litera kursywą, indeks
    opisowy (np. 'wf', 'pom', 'HU') pismem prostym, zgodnie z zasadami zapisu
    wielkości fizycznych. Gdy mathtext nie obsługuje \\mathrm, indeks zostaje
    zapisany kursywą - wykres powstaje, zamiast kończyć się błędem.
    """
    if indeks is None:
        return f'${litera}$'
    tekst_indeksu = r'\mathrm{%s}' % indeks if _mathrm_dziala() else indeks
    return f'${litera}_{{{tekst_indeksu}}}$'


def formatuj_osie_pl(fig):
    """
    Ustawia przecinek dziesiętny na wszystkich osiach liniowych rysunku
    (łącznie z paskami skali barw). Osie logarytmiczne zostają bez zmian -
    ich znaczniki są potęgami dziesięciu.
    """
    for ax in fig.axes:
        for os_ in (ax.xaxis, ax.yaxis):
            if os_.get_scale() == 'linear' and \
                    type(os_.get_major_formatter()) is ScalarFormatter:
                os_.set_major_formatter(_FormatterPrzecinek())
    return fig


def oblicz_fzi(phi, k_mD, m=1.0):
    """Zwraca (RQI, phi_z_mod, FZI_m) dla dowolnej tablicy phi i k."""
    phi = np.asarray(phi, dtype=float)
    k_mD = np.asarray(k_mD, dtype=float)
    rqi = 0.0314 * np.sqrt(k_mD / phi)
    phi_z = phi / (1.0 - phi)
    phi_z_mod = phi_z * phi**(float(m) - 1.0)
    return rqi, phi_z_mod, rqi / phi_z_mod


def _dopasuj_w_skali_log(phi, k_mD):
    """
    Dopasowuje w skali log10 model wykładniczy i potęgowy, zwraca lepszy.
    Używa regresji liniowej (polyfit), więc nie wymaga punktu startowego
    ani nie może się rozbiec jak curve_fit.
    """
    log_k = np.log10(k_mD)
    wyniki = []

    for typ, x in (('exp', phi), ('pow', np.log10(phi))):
        if len(np.unique(x)) < 2:
            continue
        b_log, a_log = np.polyfit(x, log_k, 1)
        reszty = log_k - (a_log + b_log * x)
        ss_res = float(np.sum(reszty**2))
        ss_tot = float(np.sum((log_k - log_k.mean())**2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if typ == 'exp':
            popt = (10.0**a_log, b_log * np.log(10.0))          # k = a*exp(b*phi)
        else:
            popt = (10.0**a_log, b_log)                          # k = a*phi^b
        wyniki.append({'typ': typ, 'popt': popt, 'r2': r2,
                       'rmse_log': float(np.sqrt(ss_res / len(log_k)))})

    if not wyniki:
        k_sr = float(np.mean(k_mD))
        return {'typ': 'mean', 'popt': (k_sr,), 'r2': 0.0,
                'rmse_log': float(np.std(np.log10(k_mD)))}
    return max(wyniki, key=lambda w: w['r2'])


# Szablon globalnych elementów hydraulicznych (GHE) wg Corbett i in. (2003),
# przytoczony w Tabeli 1 pracy Ha Quang Man i in. (2021). Kolejne wartości FZI
# tworzą ciąg geometryczny o ilorazie 2, dzielący przestrzeń porowatości
# i przepuszczalności na dziesięć pasm.
FZI_GHE = (0.0938, 0.1875, 0.375, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0)


def k_z_fzi(fzi, phi):
    """
    Przepuszczalność [mD] z wskaźnika FZI i porowatości - wzór (5) w pracy
    Ha Quang Man i in. (2021), równoważny odwróceniu zależności Amaefule:

        k = 1014,24 * FZI^2 * phi^3 / (1 - phi)^2

    Stała 1014,24 to 1/0,0314^2, czyli odwrotność kwadratu stałej z definicji
    RQI. Jest to postać klasycznego równania Kozeny-Carmana.
    """
    phi = np.asarray(phi, dtype=float)
    return (np.asarray(fzi, dtype=float) / 0.0314)**2 * phi**3 / (1.0 - phi)**2


def przypisz_ghe(fzi, szablon=FZI_GHE):
    """
    Przypisuje każdej próbce element GHE o najbliższej wartości FZI, mierząc
    odległość w skali logarytmicznej (szablon jest ciągiem geometrycznym).
    Zwraca (numery elementów liczone od 1, przypisane wartości FZI).

    Metoda GHE nie wymaga dopasowywania korelacji do danych: wartości FZI są
    ustalone z góry, więc działa nawet przy bardzo małej liczbie rdzeni.
    Corbett i Potter (2004) proponują ją jako szybki sposób typowania skał.
    """
    szablon = np.asarray(szablon, dtype=float)
    log_fzi = np.log10(np.maximum(np.asarray(fzi, dtype=float), 1e-8))
    numery = np.argmin(np.abs(log_fzi[:, None] - np.log10(szablon)[None, :]), axis=1)
    return numery + 1, szablon[numery]


def przewidz_k(model, phi):
    """Przepuszczalność z dopasowanego modelu korelacyjnego."""
    phi = np.asarray(phi, dtype=float)
    if model['typ'] == 'exp':
        a, b = model['popt']
        return a * np.exp(b * phi)
    if model['typ'] == 'pow':
        a, b = model['popt']
        return a * np.maximum(phi, PHI_MIN)**b
    return np.full(phi.shape, float(model['popt'][0]))


class ModelStatycznyZloza:
    def __init__(self, dx=100.0, dy=100.0):
        self.dx = float(dx)
        self.dy = float(dy)
        self.nx = 0
        self.ny = 0
        self.liczba_komorek = 0

        self.phi = None
        self.k_mD = None             # pole używane przez symulator
        self.k_mD_surowe = None      # pole wejściowe, nigdy nie nadpisywane
        self.hu_grid = None

        self.rqi = None
        self.phi_z = None
        self.fzi = None
        self.liczba_hu = 0
        self.m = 1.0
        self.modele_regresji = {}
        self.ostrzezenia = []

    # ------------------------------------------------------------ wczytanie
    def wczytaj_dane(self, df, kolumna_phi='Porowatosc', kolumna_k='Przepuszczalnosc',
                     nx=None, ny=None, kolejnosc='y_najszybciej'):
        """
        Wczytuje pole phi i k z ramki danych. W odróżnieniu od wcześniejszej
        wersji nie uzupełnia ani nie ucina po cichu brakujących wierszy -
        niezgodna liczba komórek kończy się wyjątkiem.
        """
        brakujace = [c for c in (kolumna_phi, kolumna_k) if c not in df.columns]
        if brakujace:
            raise KeyError(f"Brak kolumn {brakujace} w pliku. Dostępne: {list(df.columns)}")

        self.ostrzezenia = []
        df = df.copy()

        # Odtworzenie siatki z kolumn indeksowych lub współrzędnych
        kol_i = 'I' if 'I' in df.columns else ('x' if 'x' in df.columns else None)
        kol_j = 'J' if 'J' in df.columns else ('y' if 'y' in df.columns else None)

        if kol_i is not None and kol_j is not None:
            df = df.sort_values([kol_i, kol_j], kind='mergesort')
            self.nx = int(df[kol_i].nunique())
            self.ny = int(df[kol_j].nunique())
            zrodlo = f"kolumn {kol_i}/{kol_j}"
        elif nx is not None and ny is not None:
            self.nx, self.ny = int(nx), int(ny)
            zrodlo = "parametrów nx/ny"
        else:
            bok = int(round(np.sqrt(len(df))))
            if bok * bok != len(df):
                raise ValueError(
                    f"Plik ma {len(df)} wierszy, co nie jest kwadratem liczby całkowitej. "
                    "Dodaj kolumny I/J albo x/y, albo podaj nx i ny.")
            self.nx = self.ny = bok
            zrodlo = "założenia siatki kwadratowej"

        if self.nx * self.ny != len(df):
            raise ValueError(
                f"Siatka {self.nx}x{self.ny} = {self.nx * self.ny} komórek wyznaczona na podstawie "
                f"{zrodlo} nie zgadza się z liczbą wierszy w pliku ({len(df)}).")

        dane_phi = df[kolumna_phi].to_numpy(dtype=float)
        dane_k = df[kolumna_k].to_numpy(dtype=float)

        if not np.all(np.isfinite(dane_phi)) or not np.all(np.isfinite(dane_k)):
            raise ValueError("Plik zawiera wartości puste lub nieskończone w kolumnach phi lub k.")

        if np.median(dane_phi) > 1.0:
            dane_phi = dane_phi / 100.0
            self.ostrzezenia.append(
                "Porowatość wyglądała na podaną w procentach i została podzielona przez 100.")

        n_phi = int(np.sum((dane_phi < PHI_MIN) | (dane_phi > PHI_MAX)))
        n_k = int(np.sum(dane_k < K_MIN))
        if n_phi:
            self.ostrzezenia.append(
                f"{n_phi} wartości porowatości było poza zakresem {PHI_MIN}-{PHI_MAX} i zostało przyciętych.")
        if n_k:
            self.ostrzezenia.append(
                f"{n_k} wartości przepuszczalności było poniżej {K_MIN} mD i zostało podniesionych do tej wartości.")

        dane_phi = np.clip(dane_phi, PHI_MIN, PHI_MAX)
        dane_k = np.maximum(dane_k, K_MIN)

        if kol_i is not None and kol_j is not None:
            ksztalt_order = 'C'                     # po sortowaniu J zmienia się najszybciej
        else:
            ksztalt_order = 'C' if kolejnosc == 'y_najszybciej' else 'F'

        self.phi = dane_phi.reshape((self.nx, self.ny), order=ksztalt_order)
        self.k_mD_surowe = dane_k.reshape((self.nx, self.ny), order=ksztalt_order)
        self.k_mD = self.k_mD_surowe.copy()
        self.liczba_komorek = self.nx * self.ny

        self.hu_grid = None
        self.modele_regresji = {}
        self.rqi = self.phi_z = self.fzi = None
        return self.nx, self.ny

    def przywroc_surowa_przepuszczalnosc(self):
        """Cofa korektę HU, nie dotykając wyznaczonych jednostek."""
        self.k_mD = self.k_mD_surowe.copy()
        return self.k_mD

    def _sprawdz_dane(self):
        if self.phi is None or self.k_mD_surowe is None:
            raise RuntimeError("Najpierw wczytaj dane złoża.")

    def diagnostyka_liczby_jednostek(self, max_k=10, m=1.0, random_state=42):
        """
        Dane do metody łokcia: dla k = 1 ... max_k dzieli log10(FZI_m) metodą
        K-Means i zapisuje inercję (WCSS, suma kwadratów odległości punktów od
        środków klastrów) oraz dystorsję (inercja podzielona przez liczbę
        punktów - miara domyślna w KElbowVisualizer z pakietu Yellowbrick).

        Zwraca ramkę danych z kolumnami k, inercja, dystorsja.
        """
        self._sprawdz_dane()
        x = self._cechy_do_klastrowania(m)
        wiersze = []
        for k in range(1, int(max_k) + 1):
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(x)
            wiersze.append({'k': k,
                            'inercja': float(km.inertia_),
                            'dystorsja': float(km.inertia_) / len(x)})
        return pd.DataFrame(wiersze)

    @staticmethod
    def lokiec_dystorsji(tabela, miara='dystorsja'):
        """
        Wyznacza punkt łokcia krzywej dystorsji. Zwraca (k, wartość) albo
        (None, None), gdy krzywa nie ma wyraźnego załamania.

        Dwa odstępstwa od najprostszego ujęcia, oba wynikające z pomiarów na
        danych o znanej liczbie jednostek (po cztery realizacje dla zbioru
        piaskowcowego o czterech jednostkach i węglanowego o siedmiu):

          * łokieć wyznaczany jest na logarytmie dystorsji, ponieważ maleje ona
            w przybliżeniu geometrycznie, a w skali logarytmicznej załamanie
            wypada tam, gdzie zmienia się tempo spadku,
          * pomijany jest punkt k = 1, w którym dystorsja równa się całkowitej
            wariancji zbioru i przy normalizacji przeciąga wynik w lewo.

        Trafność wariantów (8 przypadków):
            dystorsja, wszystkie k          0/8   (zawsze wskazywał 3)
            log(dystorsja), wszystkie k     4/8
            log(dystorsja), k >= 2          6/8   <- wariant przyjęty
        W pozostałych dwóch przypadkach wskazanie różniło się o jeden
        od prawdziwej liczby jednostek.
        """
        if tabela is None or len(tabela) < 4:
            return None, None
        k = tabela['k'].to_numpy()[1:]
        wartosci = tabela[miara].to_numpy()[1:]
        if np.any(wartosci <= 0):
            return None, None
        k_lokcia, _, _ = znajdz_lokiec(k, np.log10(wartosci))
        if k_lokcia is None:
            return None, None
        wartosc = float(tabela.loc[tabela['k'] == int(k_lokcia), miara].iloc[0])
        return int(k_lokcia), wartosc

    @classmethod
    def zalecenie_liczby_jednostek(cls, tabela):
        """
        Zamienia krzywą dystorsji na zalecenie. Gdy krzywa nie ma załamania,
        dane nie są wyraźnie pogrupowane i liczbę jednostek trzeba przyjąć
        z przesłanek geologicznych.
        """
        if tabela is None or tabela.empty:
            return {'k_sugerowane': 4, 'pewne': False,
                    'zalecenie': 'Brak danych do oceny.'}

        k_lokcia, _ = cls.lokiec_dystorsji(tabela)
        if k_lokcia is None:
            return {
                'k_sugerowane': 4, 'pewne': False,
                'zalecenie': (
                    'Krzywa dystorsji nie ma wyraźnego załamania, więc metoda łokcia nie '
                    'rozstrzyga podziału. Oznacza to, że jednostki zachodzą na siebie, '
                    'a rozkład FZI jest w praktyce jednomodalny. Liczbę jednostek przyjmij '
                    'z przesłanek geologicznych lub z podziału na elektrofacje, tak jak '
                    'zrobili to Nooruddin i in. (2011).')}

        return {
            'k_sugerowane': k_lokcia, 'pewne': True,
            'zalecenie': (
                f'Przyjmij {k_lokcia} jednostek hydraulicznych - w tym punkcie krzywa '
                'dystorsji wyraźnie się załamuje, czyli dalsze dzielenie zbioru przestaje '
                'istotnie poprawiać zwartość klastrów.')}

    def _cechy_do_klastrowania(self, m):
        _, _, fzi = oblicz_fzi(self.phi.ravel(), self.k_mD_surowe.ravel(), m)
        return np.log10(np.maximum(fzi, 1e-8)).reshape(-1, 1)

    # ---------------------------------------------- jednostki hydrauliczne
    def wyznacz_jednostki_hu(self, liczba_hu=4, m=1.0, random_state=42):
        self._sprawdz_dane()
        self.liczba_hu = int(liczba_hu)
        self.m = float(m)

        x = self._cechy_do_klastrowania(self.m)
        km = KMeans(n_clusters=self.liczba_hu, random_state=random_state, n_init=10)
        etykiety = km.fit_predict(x)

        # Numeracja rosnąco według środków klastrów: HU 1 = najniższe FZI
        porzadek = np.argsort(km.cluster_centers_.ravel())
        mapowanie = {stary: nowy + 1 for nowy, stary in enumerate(porzadek)}
        etykiety = np.array([mapowanie[e] for e in etykiety])

        rqi, phi_z, fzi = oblicz_fzi(self.phi, self.k_mD_surowe, self.m)
        self.rqi, self.phi_z, self.fzi = rqi, phi_z, fzi
        self.hu_grid = etykiety.reshape((self.nx, self.ny))
        return self.hu_grid

    def uwzglednij_wplyw_hu_na_przepuszczalnosc(self):
        """
        Zastępuje przepuszczalność wartościami z korelacji k(phi) dopasowanej
        osobno w obrębie każdej jednostki hydraulicznej.
        """
        if self.hu_grid is None:
            raise RuntimeError("Najpierw wyznacz jednostki hydrauliczne.")

        self.modele_regresji = {}
        nowe_k = np.empty_like(self.k_mD_surowe)

        for hu_id in np.unique(self.hu_grid):
            maska = self.hu_grid == hu_id
            phi_d, k_d = self.phi[maska], self.k_mD_surowe[maska]
            if len(phi_d) < 3:
                k_sr = float(np.mean(k_d))
                model = {'typ': 'mean', 'popt': (k_sr,), 'r2': 0.0,
                         'rmse_log': float(np.std(np.log10(k_d))) if len(k_d) > 1 else 0.0}
            else:
                model = _dopasuj_w_skali_log(phi_d, k_d)
            model['n'] = int(maska.sum())
            self.modele_regresji[int(hu_id)] = model
            nowe_k[maska] = przewidz_k(model, phi_d)

        self.k_mD = np.maximum(nowe_k, K_MIN)
        return self.k_mD

    def diagnostyka_wykladnika_m(self, liczba_hu=4, m_min=1.0, m_max=3.0, krok=0.05,
                                 licz_blad_predykcji=False, powtorzenia=5, random_state=42):
        """
        Przegląda wartości wykładnika m i ocenia każdą z nich wskaźnikiem
        sylwetki podziału log10(FZI_m): im wyższy, tym wyraźniej rozdzielone
        jednostki hydrauliczne. Maksimum wskazuje m najlepiej dopasowane do
        struktury danych.

        Opcjonalnie (licz_blad_predykcji = True) liczony jest też błąd predykcji
        przepuszczalności z walidacji krzyżowej. Na siatce 30x30 miara ta okazała
        się jednak zbyt płaska i zaszumiona, żeby wskazać m, dlatego służy
        wyłącznie do kontroli, a nie do wyboru wartości.

        SKUTECZNOŚĆ METODY, zbadana na danych syntetycznych o znanym m
        (cztery jednostki, siatka 30x30, cztery realizacje dla każdej wartości):
            m zadane 1,0 -> wskazane 1,04
            m zadane 1,5 -> wskazane 1,53
            m zadane 2,0 -> wskazane 2,00
            m zadane 2,5 -> wskazane 2,30 (duży rozrzut między realizacjami)
            m zadane 3,0 -> wskazane 1,39 (metoda zawodzi)
        Metoda jest więc wiarygodna w przedziale ok. 1,0-2,0, czyli tam, gdzie
        mieszczą się piaskowce i większość węglanów, a zawodzi dla wartości
        skrajnych.

        WARUNEK KONIECZNY: liczbę jednostek trzeba ustalić wcześniej, bo wynik
        silnie od niej zależy. Dla tych samych danych o m = 2,0 otrzymano:
        3 jednostki -> m = 1,05; 4 -> m = 1,95; 5 -> m = 2,30; 6 -> m = 2,45.

        UWAGA INTERPRETACYJNA. Wykładnik cementacji jest wielkością
        petrofizyczną zdefiniowaną w prawie Archiego (F = a / phi^m) i wyznacza
        się go z pomiarów elektrycznych na rdzeniach - tak postąpili Nooruddin
        i in. (2011), przyjmując m = 2,00. Wartość dobrana z samych danych
        phi-k NIE jest pomiarem wykładnika cementacji, lecz parametrem
        efektywnym, przy którym podział na jednostki jest najwyraźniejszy.
        W pracy należy ją tak właśnie nazywać.

        Zwraca ramkę danych z kolumnami m, sylwetka oraz, opcjonalnie,
        rmse_hu i rmse_globalna. Metoda nie zmienia stanu obiektu.
        """
        self._sprawdz_dane()
        if liczba_hu < 2:
            raise ValueError("Dobór wykładnika m wymaga co najmniej dwóch jednostek.")

        stan = (self.hu_grid, self.modele_regresji, self.k_mD, self.m,
                self.liczba_hu, self.rqi, self.phi_z, self.fzi)
        wiersze = []
        try:
            for m in np.arange(m_min, m_max + krok / 2, krok):
                m = round(float(m), 3)
                x = self._cechy_do_klastrowania(m)
                etykiety = KMeans(n_clusters=int(liczba_hu), n_init=10,
                                  random_state=random_state).fit_predict(x)
                wiersz = {'m': m, 'sylwetka': float(silhouette_score(x, etykiety))}
                if licz_blad_predykcji:
                    try:
                        w = self.walidacja_krzyzowa(liczba_hu=liczba_hu, m=m,
                                                    powtorzenia=powtorzenia,
                                                    random_state=random_state)
                        wiersz.update(rmse_hu=w['rmse_hu'], rmse_globalna=w['rmse_globalna'])
                    except Exception:
                        wiersz.update(rmse_hu=np.nan, rmse_globalna=np.nan)
                wiersze.append(wiersz)
        finally:
            (self.hu_grid, self.modele_regresji, self.k_mD, self.m,
             self.liczba_hu, self.rqi, self.phi_z, self.fzi) = stan
        return pd.DataFrame(wiersze)

    def znajdz_optymalne_m(self, liczba_hu, m_min=1.0, m_max=3.0, krok=0.05,
                           random_state=42, zwroc_tabele=False):
        """
        Zwraca m o najwyższym wskaźniku sylwetki. Przed użyciem przeczytaj
        opis skuteczności i ograniczeń w diagnostyka_wykladnika_m, a do
        interpretacji wyniku użyj zalecenie_wykladnika_m.
        """
        tabela = self.diagnostyka_wykladnika_m(liczba_hu, m_min, m_max, krok,
                                               random_state=random_state)
        if tabela.empty:
            return (float(m_min), tabela) if zwroc_tabele else float(m_min)
        najlepsze = float(tabela.loc[tabela['sylwetka'].idxmax(), 'm'])
        return (najlepsze, tabela) if zwroc_tabele else najlepsze

    @staticmethod
    def zalecenie_wykladnika_m(tabela, prog_wyrazny=3.0, prog_minimalny=1.0):
        """
        Zamienia krzywą kryterium na jednoznaczne zalecenie.

        Samo maksimum nie wystarcza do decyzji: trzeba wiedzieć, o ile jest
        lepsze od wartości klasycznej m = 1, bo odejście od modelu Amaefule
        i in. (1993) wymaga uzasadnienia. Miarą jest względny przyrost
        wskaźnika sylwetki względem m = 1.

        Progi wyznaczono na danych syntetycznych o znanym m (cztery jednostki,
        siatka 30x30, po cztery realizacje):
            dane o m = 1,0  -> przyrost 0,0-0,3 %
            dane o m = 1,5  -> przyrost 4,4-10,6 %
            dane o m = 2,0  -> przyrost 6,6-10,6 %
            dane o m = 2,5  -> przyrost 1,3-4,3 % (metoda już traci czułość)
        Przyrost poniżej 1 % nie odróżnia się od danych opisanych klasycznym
        FZI, a powyżej 3 % występował wyłącznie tam, gdzie wykładnik naprawdę
        był podwyższony.

        Zwraca słownik z kluczami: m_sugerowane, m_klasyczne, zysk_proc,
        werdykt ('klasyczny' / 'niejednoznaczny' / 'zmodyfikowany'), zalecenie.
        """
        if tabela is None or tabela.empty:
            return {'m_sugerowane': 1.0, 'm_klasyczne': 1.0, 'zysk_proc': 0.0,
                    'werdykt': 'klasyczny',
                    'zalecenie': 'Brak danych do oceny - przyjmij m = 1,0.'}

        i_klas = tabela['m'].sub(1.0).abs().idxmin()
        s_klas = float(tabela.loc[i_klas, 'sylwetka'])
        i_max = tabela['sylwetka'].idxmax()
        m_sug = float(tabela.loc[i_max, 'm'])
        s_max = float(tabela.loc[i_max, 'sylwetka'])
        zysk = (s_max - s_klas) / s_klas * 100.0 if s_klas > 0 else 0.0

        if zysk < prog_minimalny:
            werdykt = 'klasyczny'
            zalecenie = (
                'Przyjmij m = 1,0, czyli klasyczne FZI wg Amaefule i in. (1993). '
                'Podwyższanie wykładnika nie poprawia rozdzielności jednostek '
                f'(przyrost {zysk:.1f} %), więc dane nie dają podstaw do odejścia '
                'od modelu klasycznego.')
            m_sug = 1.0
        elif zysk < prog_wyrazny:
            werdykt = 'niejednoznaczny'
            zalecenie = (
                f'Kryterium wskazuje m = {m_sug:.2f}, ale przewaga nad modelem klasycznym '
                f'jest niewielka ({zysk:.1f} %) i mieści się w zakresie, w którym metoda '
                'bywa myląca. Bezpieczniej przyjąć m = 1,0, chyba że za wyższą wartością '
                'przemawiają przesłanki niezależne: litologia węglanowa albo pomiary '
                'elektryczne rdzeni.')
        else:
            werdykt = 'zmodyfikowany'
            zalecenie = (
                f'Przyjmij m = {m_sug:.2f}. Zmodyfikowany wskaźnik FZI wg Nooruddina '
                f'i in. (2011) rozdziela jednostki wyraźnie lepiej niż klasyczny '
                f'(przyrost {zysk:.1f} %), co uzasadnia odejście od m = 1,0.')

        return {'m_sugerowane': m_sug, 'm_klasyczne': 1.0, 'zysk_proc': zysk,
                'werdykt': werdykt, 'zalecenie': zalecenie}

    def walidacja_krzyzowa(self, liczba_hu=4, m=1.0, udzial_treningowy=0.2,
                           powtorzenia=20, uzyj_wspolrzednych=True, random_state=42):
        """
        Ocena zdolności predykcyjnej korelacji k(phi) w sytuacji odpowiadającej
        praktyce: przepuszczalność znana tylko w części komórek (rdzenie),
        porowatość znana wszędzie (karotaż).

        Porównywane są dwa podejścia, oba dopasowane WYŁĄCZNIE na próbce
        treningowej i oceniane na pozostałych komórkach:
          * jedna globalna korelacja k(phi),
          * korelacje osobne dla każdej jednostki hydraulicznej.

        Kluczowy warunek uczciwości testu: jednostka hydrauliczna komórki
        testowej NIE może być wyznaczona z jej własnej przepuszczalności,
        bo FZI jest funkcją k - takie przypisanie przenosiłoby szukaną
        odpowiedź do danych wejściowych. Dlatego HU komórek testowych jest
        przewidywana klasyfikatorem (k najbliższych sąsiadów) z wielkości
        znanych poza rdzeniem: porowatości i opcjonalnie położenia, które
        odpowiada informacji z modelu geologicznego lub sejsmiki.

        Zwraca RMSE log10(k) obu podejść. Metoda nie zmienia stanu obiektu.
        """
        from sklearn.neighbors import KNeighborsClassifier

        self._sprawdz_dane()
        phi = self.phi.ravel()
        k = self.k_mD_surowe.ravel()
        log_k = np.log10(k)
        _, _, fzi = oblicz_fzi(phi, k, m)
        log_fzi = np.log10(np.maximum(fzi, 1e-8)).reshape(-1, 1)

        # Predyktory dostępne poza rdzeniem (bez udziału k)
        cechy = [(phi - phi.mean()) / phi.std()]
        if uzyj_wspolrzednych:
            ii, jj = np.indices((self.nx, self.ny))
            cechy += [(ii.ravel() - ii.mean()) / ii.std(), (jj.ravel() - jj.mean()) / jj.std()]
        X = np.column_stack(cechy)

        n = len(phi)
        n_tren = max(5 * int(liczba_hu), int(round(udzial_treningowy * n)))
        if n_tren >= n:
            raise ValueError("Zbyt mała próbka, żeby wydzielić zbiór testowy.")

        rng = np.random.default_rng(random_state)
        bledy_hu, bledy_glob, trafnosci = [], [], []
        przyklad = None                  # predykcje z pierwszego losowania - do wykresu

        for nr in range(int(powtorzenia)):
            idx = rng.permutation(n)
            tren, test = idx[:n_tren], idx[n_tren:]

            model_g = _dopasuj_w_skali_log(phi[tren], k[tren])
            bledy_glob.append(np.sqrt(np.mean((np.log10(przewidz_k(model_g, phi[test])) - log_k[test])**2)))

            # Jednostki wyznaczane z rdzeni, następnie rozciągane na złoże klasyfikatorem
            km = KMeans(n_clusters=int(liczba_hu), n_init=10, random_state=random_state)
            etykiety_tren = km.fit_predict(log_fzi[tren])
            klasyfikator = KNeighborsClassifier(n_neighbors=min(5, n_tren)).fit(X[tren], etykiety_tren)
            etykiety_test = klasyfikator.predict(X[test])

            # Trafność klasyfikacji mierzona względem przypisania z prawdziwego FZI
            etykiety_prawdziwe = km.predict(log_fzi[test])
            trafnosci.append(float(np.mean(etykiety_test == etykiety_prawdziwe)))

            pred = np.empty(len(test))
            for c in range(int(liczba_hu)):
                m_tren, m_test = etykiety_tren == c, etykiety_test == c
                if not m_test.any():
                    continue
                model_c = (_dopasuj_w_skali_log(phi[tren][m_tren], k[tren][m_tren])
                           if m_tren.sum() >= 3 else model_g)
                pred[m_test] = przewidz_k(model_c, phi[test][m_test])
            bledy_hu.append(np.sqrt(np.mean((np.log10(np.maximum(pred, K_MIN)) - log_k[test])**2)))
            if nr == 0:
                # numeracja jak w całej aplikacji: HU 1 = najniższe FZI
                ranga = np.argsort(np.argsort(km.cluster_centers_.ravel())) + 1
                przyklad = {'k_rzeczywiste': k[test],
                            'k_globalna': przewidz_k(model_g, phi[test]),
                            'k_hu': np.maximum(pred, K_MIN),
                            'hu': ranga[etykiety_test]}

        return {
            'rmse_hu': float(np.mean(bledy_hu)),
            'rmse_hu_odch': float(np.std(bledy_hu)),
            'rmse_globalna': float(np.mean(bledy_glob)),
            'rmse_globalna_odch': float(np.std(bledy_glob)),
            'poprawa_proc': float((np.mean(bledy_glob) - np.mean(bledy_hu)) / np.mean(bledy_glob) * 100.0),
            'trafnosc_klasyfikacji': float(np.mean(trafnosci)),
            'n_treningowych': int(n_tren),
            'powtorzenia': int(powtorzenia),
            'rmse_hu_lista': [float(b) for b in bledy_hu],
            'rmse_globalna_lista': [float(b) for b in bledy_glob],
            'przyklad': przyklad,
        }

    @staticmethod
    def _rysuj_rozrzut_walidacji(ax, wynik, wariant):
        """Przepuszczalność estymowana względem zmierzonej w komórkach testowych."""
        przyklad = wynik['przyklad']
        k_rz = przyklad['k_rzeczywiste']
        zakres = [max(k_rz.min() / 3, K_MIN), k_rz.max() * 3]
        if wariant == 'globalny':
            k_pred, tytul = przyklad['k_globalna'], '(a) Model globalny'
            ax.scatter(k_rz, k_pred, s=12, alpha=0.5, color='#1f77b4', edgecolors='none')
        else:
            k_pred, tytul = przyklad['k_hu'], '(b) Model HU (K-Means)'
            for c in np.unique(przyklad['hu']):
                m = przyklad['hu'] == c
                ax.scatter(k_rz[m], k_pred[m], s=12, alpha=0.6, edgecolors='none',
                           color=KOLORY_HU[(int(c) - 1) % len(KOLORY_HU)], label=f'HU {int(c)}')
        ax.plot(zakres, zakres, color='#333333', lw=1.2, label='Linia zgodności 1:1')
        ax.plot(zakres, [z * 10 for z in zakres], color='#999999', lw=0.9, ls='--')
        ax.plot(zakres, [z / 10 for z in zakres], color='#999999', lw=0.9, ls='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(zakres)
        ax.set_ylim(zakres)
        ax.set_aspect('equal')
        r2 = 1 - np.sum((np.log10(k_pred) - np.log10(k_rz))**2) / \
            np.sum((np.log10(k_rz) - np.log10(k_rz).mean())**2)
        rmse = np.sqrt(np.mean((np.log10(k_pred) - np.log10(k_rz))**2))
        ax.text(0.04, 0.96, f"$R^2$ = {liczba_pl(r2, '.2f')}\nRMSE = {liczba_pl(rmse)}",
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#cccccc'))
        ax.set_xlabel(f"Przepuszczalność zmierzona, {sym('k', 'pom')} [mD]", fontsize=11)
        ax.set_ylabel(f"Przepuszczalność estymowana, {sym('k', 'est')} [mD]", fontsize=11)
        ax.set_title(tytul, fontsize=12, loc='left')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, which='major', ls='--', color='#e0e0e0', alpha=0.7)

    @staticmethod
    def _rysuj_bledy_walidacji(ax, wynik):
        """Rozkład błędu RMSE z kolejnych losowań próbki rdzeniowej."""
        dane = [wynik['rmse_globalna_lista'], wynik['rmse_hu_lista']]
        ax.boxplot(dane, widths=0.5, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor='#dbe9f6', edgecolor='#1f77b4'),
                   medianprops=dict(color='#d62728', lw=1.5))
        for i, d in enumerate(dane, start=1):
            ax.scatter(np.full(len(d), i) + np.linspace(-0.12, 0.12, len(d)), d,
                       s=12, color='#333333', alpha=0.6, zorder=3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Model globalny', 'Model HU (K-Means)'], fontsize=10)
        ax.set_ylabel('Błąd średniokwadratowy, RMSE (dla log $k$) [-]', fontsize=11)
        ax.set_title(f"(c) Rozkład błędu w {wynik['powtorzenia']} losowaniach",
                     fontsize=12, loc='left')
        ax.grid(True, axis='y', ls='--', color='#e0e0e0', alpha=0.7)

    @classmethod
    def generuj_wykres_rozrzutu_walidacji(cls, wynik, wariant='hu'):
        """
        Osobny wykres (a) lub (b) walidacji krzyżowej: przepuszczalność
        estymowana względem zmierzonej w komórkach testowych (pierwsze
        losowanie). wariant: 'globalny' albo 'hu'.
        """
        _styl_wykresu()
        fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
        cls._rysuj_rozrzut_walidacji(ax, wynik, wariant)
        fig.tight_layout()
        return formatuj_osie_pl(fig)

    @classmethod
    def generuj_wykres_bledu_walidacji(cls, wynik):
        """Osobny wykres (c) walidacji krzyżowej: rozkład błędu w losowaniach."""
        _styl_wykresu()
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
        cls._rysuj_bledy_walidacji(ax, wynik)
        fig.tight_layout()
        return formatuj_osie_pl(fig)

    @classmethod
    def generuj_wykres_walidacji(cls, wynik):
        """
        Wszystkie trzy wykresy walidacji krzyżowej na jednym rysunku
        (przydatne jako jeden rysunek w pracy):
          (a), (b) przepuszczalność estymowana względem zmierzonej w komórkach
              testowych (pierwsze losowanie) - im bliżej przekątnej 1:1, tym
              lepsza estymacja; linie przerywane wyznaczają pas błędu jednego
              rzędu wielkości,
          (c) rozkład błędu RMSE ze wszystkich losowań - pokazuje, czy
              przewaga jednej metody jest stała, czy przypadkowa.
        """
        _styl_wykresu()
        fig, osie = plt.subplots(1, 3, figsize=(15, 4.8), dpi=110,
                                 gridspec_kw={'width_ratios': [1, 1, 0.9]})
        cls._rysuj_rozrzut_walidacji(osie[0], wynik, 'globalny')
        cls._rysuj_rozrzut_walidacji(osie[1], wynik, 'hu')
        cls._rysuj_bledy_walidacji(osie[2], wynik)
        fig.tight_layout()
        return formatuj_osie_pl(fig)

    # ------------------------------------------------------- lokalizacja odwiertów
    def znajdz_sweet_spots(self, liczba_odwiertow=1, promien_ochronny_komorki=3, h=1.0,
                           margines=1):
        """
        Wybiera komórki o największej pojemności przepływowej k*h, zachowując
        zadany odstęp między odwiertami. Margines odsuwa odwierty od granicy
        złoża (model nie dopuszcza odwiertu w komórce brzegowej).
        """
        self._sprawdz_dane()
        ocena = self.k_mD * float(h)
        robocza = ocena.copy()
        if margines > 0:
            robocza[:margines, :] = -np.inf
            robocza[-margines:, :] = -np.inf
            robocza[:, :margines] = -np.inf
            robocza[:, -margines:] = -np.inf

        znalezione = []
        r2 = promien_ochronny_komorki**2
        ii, jj = np.indices(robocza.shape)
        for _ in range(int(liczba_odwiertow)):
            if not np.isfinite(robocza).any():
                break
            i_opt, j_opt = np.unravel_index(np.argmax(robocza), robocza.shape)
            znalezione.append({'i': int(i_opt), 'j': int(j_opt),
                               'kh': float(ocena[i_opt, j_opt])})
            robocza[(ii - i_opt)**2 + (jj - j_opt)**2 <= r2] = -np.inf
        return znalezione

    # --------------------------------------------------------------- wykresy
    def generuj_tabele_rownan(self):
        if not self.modele_regresji:
            return None
        wiersze = []
        for hu_id in sorted(self.modele_regresji):
            mod = self.modele_regresji[hu_id]
            if mod['typ'] == 'exp':
                a, b = mod['popt']
                rownanie = (f"<i>k</i> = {liczba_pl(a, '.4g')}·e<sup>{liczba_pl(b)}"
                            f"<i>φ</i></sup>")
            elif mod['typ'] == 'pow':
                a, b = mod['popt']
                rownanie = f"<i>k</i> = {liczba_pl(a, '.4g')}·<i>φ</i><sup>{liczba_pl(b)}</sup>"
            else:
                rownanie = f"<i>k</i> = {liczba_pl(float(mod['popt'][0]), '.2f')} mD (wartość stała)"
            wiersze.append({
                "Jednostka HU": f"HU {hu_id}",
                "Liczba komórek": mod.get('n', ''),
                "Równanie korelacyjne": rownanie,
                "<i>R</i><sup>2</sup> (dla log <i>k</i>)": liczba_pl(mod['r2'], '.4f'),
                "RMSE (dla log <i>k</i>)": liczba_pl(mod['rmse_log'], '.4f'),
            })
        return pd.DataFrame(wiersze)

    def generuj_figure_diagnostyczna(self):
        """Wykres k-phi z podziałem na jednostki hydrauliczne i liniami korelacji."""
        if self.hu_grid is None or not self.modele_regresji:
            raise RuntimeError("Najpierw wyznacz HU i dopasuj korelacje.")
        _styl_wykresu()
        fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
        ax.set_yscale('log')
        phi_gladkie = np.linspace(self.phi.min(), self.phi.max(), 200)

        for hu_id in sorted(self.modele_regresji):
            maska = self.hu_grid == hu_id
            mod = self.modele_regresji[hu_id]
            kolor = KOLORY_HU[(hu_id - 1) % len(KOLORY_HU)]
            ax.scatter(self.phi[maska], self.k_mD_surowe[maska], color=kolor, alpha=0.7,
                       edgecolors='#333333', linewidths=0.5, s=35, zorder=3,
                       label=f"HU {hu_id} ($n$ = {mod.get('n', maska.sum())}, "
                             f"$R^2$ = {liczba_pl(mod['r2'])})")
            k_lin = przewidz_k(mod, phi_gladkie)
            widoczne = (k_lin > 1e-3) & (k_lin < 1e5)
            ax.plot(phi_gladkie[widoczne], k_lin[widoczne], color='black', linewidth=0.8, zorder=4)

        ax.set_xlabel(r'Porowatość efektywna, $\phi$ [-]', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_ylabel(r'Przepuszczalność absolutna, $k$ [mD]', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_title(r'Zależność $k(\phi)$ w obrębie jednostek hydraulicznych', fontsize=12,
                     fontweight='bold', pad=12)
        ax.grid(True, which='both', ls='--', color='#d0d0d0', alpha=0.6, zorder=1)
        for s in ax.spines.values():
            s.set_edgecolor('#222222')
            s.set_linewidth(1.2)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.,
                  frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9)
        fig.tight_layout()
        return formatuj_osie_pl(fig)

    def generuj_wykres_lokcia(self, tabela, miara='dystorsja'):
        """
        Klasyczny wykres łokcia w układzie stosowanym przez KElbowVisualizer
        z pakietu Yellowbrick: krzywa wybranej miary względem liczby klastrów
        i pionowa linia przerywana w wykrytym punkcie łokcia wraz z jego opisem.

        Miara domyślna to dystorsja, czyli średnia kwadratów odległości punktów
        od środków klastrów. Punkt łokcia wyznacza algorytm Kneedle
        (Satopää i in., 2011); patrz funkcja znajdz_lokiec.

        Metoda łokcia zawodzi, gdy dane nie są wyraźnie pogrupowane - krzywa
        jest wtedy gładka i bez załamania. W takim przypadku linia łokcia nie
        jest rysowana, a wykres opatrzony zostaje adnotacją.
        """
        nazwy = {'dystorsja': 'Dystorsja klastrów [-]',
                 'inercja': 'Inercja klastrów, WCSS [-]'}
        k = tabela['k'].to_numpy()
        wartosci = tabela[miara].to_numpy()

        _styl_wykresu()
        fig, ax1 = plt.subplots(figsize=(6.5, 4.0), dpi=150)
        ax1.plot(k, wartosci, marker='o', color='#1f77b4', linewidth=1.8, markersize=5,
                 label=('Dystorsja K-Means' if miara == 'dystorsja' else 'Inercja K-Means'),
                 zorder=3)

        k_lokcia, y_lokcia = self.lokiec_dystorsji(tabela, miara)
        if k_lokcia is not None:
            ax1.axvline(k_lokcia, color='#1f77b4', ls='--', linewidth=1.5, zorder=2,
                        label=f"Punkt łokcia: {sym('n', 'HU')} = {int(k_lokcia)}")
        else:
            ax1.annotate('Brak wyraźnego załamania krzywej -\nmetoda łokcia nie rozstrzyga',
                         xy=(0.5, 0.92), xycoords='axes fraction', ha='center',
                         fontsize=8.5, color='#d62728')

        ax1.set_xlabel(f"Liczba jednostek hydraulicznych, {sym('n', 'HU')} [-]", fontsize=10,
                       fontweight='bold')
        ax1.set_ylabel(nazwy.get(miara, miara), fontsize=10, fontweight='bold')
        ax1.set_xticks(list(k))
        ax1.tick_params(axis='both', labelsize=8)
        ax1.grid(True, ls='--', color='#e0e0e0', alpha=0.7)

        ax1.legend(fontsize=8, loc='best')
        ax1.set_title('Metoda łokcia dla podziału na jednostki hydrauliczne',
                      fontsize=11, fontweight='bold')
        for s in ax1.spines.values():
            s.set_edgecolor('#444444')
        fig.tight_layout()
        return formatuj_osie_pl(fig)

    def generuj_mape(self, pole, tytul, etykieta, skala_log=False, cmap='viridis',
                     dyskretne=False):
        """
        Mapa dowolnego pola w poprawnej orientacji (oś X pozioma, Y pionowa).
        Dla dyskretne=True (np. numery jednostek hydraulicznych) skala barwna
        dostaje po jednym kolorze na wartość i znaczniki w liczbach całkowitych.
        """
        from matplotlib.colors import LogNorm, BoundaryNorm
        _styl_wykresu()
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
        dane = np.asarray(pole, dtype=float)

        norma = None
        if dyskretne:
            wartosci = np.unique(dane).astype(int)
            granice = np.arange(wartosci.min() - 0.5, wartosci.max() + 1.5, 1.0)
            norma = BoundaryNorm(granice, len(granice) - 1)
            cmap = plt.get_cmap(cmap, len(granice) - 1)
        elif skala_log:
            norma = LogNorm(vmin=max(dane.min(), 1e-6), vmax=dane.max())

        obraz = ax.imshow(dane.T, origin='lower', cmap=cmap, norm=norma,
                          extent=[0, self.nx * self.dx, 0, self.ny * self.dy])
        if dyskretne:
            fig.colorbar(obraz, ax=ax, label=etykieta, shrink=0.85,
                         ticks=wartosci, boundaries=granice, spacing='proportional')
        else:
            fig.colorbar(obraz, ax=ax, label=etykieta, shrink=0.85)
        ax.set_xlabel(r'Współrzędna $x$ [m]')
        ax.set_ylabel(r'Współrzędna $y$ [m]')
        ax.set_title(tytul)
        fig.tight_layout()
        return formatuj_osie_pl(fig)

    def do_ramki(self):
        """Eksport pól statycznych do ramki danych (do załączników pracy)."""
        self._sprawdz_dane()
        ii, jj = np.indices((self.nx, self.ny))
        dane = {
            'I': ii.ravel(), 'J': jj.ravel(),
            'x': ((ii.ravel() + 0.5) * self.dx), 'y': ((jj.ravel() + 0.5) * self.dy),
            'Porowatosc': self.phi.ravel(),
            'Przepuszczalnosc': self.k_mD_surowe.ravel(),
            'Przepuszczalnosc_HU': self.k_mD.ravel(),
        }
        if self.fzi is not None:
            dane['RQI'] = self.rqi.ravel()
            dane['FZI'] = self.fzi.ravel()
            dane['HU'] = self.hu_grid.ravel()
        return pd.DataFrame(dane)