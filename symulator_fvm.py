"""
symulator_fvm.py - dynamiczny model przepływu gazu rzeczywistego w złożu 2D.

MODEL FIZYCZNY
Jednofazowy przepływ gazu w ośrodku porowatym ze stałym nasyceniem wodą
związaną S_wc, w warunkach izotermicznych, przy szczelnych granicach złoża
(warunek Neumanna dP/dn = 0 wynika wprost ze sformułowania bilansowego).

Bilans masy dla bloku i, w postaci całkowej (metoda objętości skończonych):

    d/dt [ V_p,i * a * (p/Z)_i ]  =  suma po sasiadach T_ij * (a/2) * (m_j - m_i)  -  q_i

    V_p,i = dx * dy * h * phi_i * (1 - S_wc)        objętość porowa gazu [m3]
    a     = T_sc / (p_sc * T)                        przeliczenie na warunki normalne [1/Pa]
    m(p)  = calka 2 p' / (mu(p') Z(p')) dp'          pseudociśnienie, Al-Hussainy i in. (1966)
    T_ij  = k_harm * A_ij / d_ij                     transmisyjność ściany [m3]

Jest to dwupunktowa aproksymacja strumienia (TPFA) opisana przez Lie (2016,
rozdz. 4.4.1, wzory 4.50-4.52, s. 131-134): transmisyjność ściany wynika ze
złożenia dwóch połówkowych transmisyjności, co dla siatki kartezjańskiej daje
średnią harmoniczną przepuszczalności sąsiadujących bloków. Strumień przez
ścianę jest jedną liczbą wspólną dla obu bloków, więc schemat zachowuje masę
także w ośrodku niejednorodnym.

Pseudociśnienie linearyzuje człon dyfuzyjny: zależność od ciśnienia, lepkości i
współczynnika ściśliwości jest zawarta w m(p), a ściśliwość gazu wchodzi przez
człon akumulacji d(p/Z)/dt, bez przybliżenia c_g = 1/p.

MODEL ODWIERTU
Odwiert jest członem źródłowym w bilansie bloku (nie warunkiem Dirichleta), więc
produkcja jest zgodna ze spadkiem ciśnienia - kontroluje to metoda
bilans_materialowy(). Wskaźnik produktywności wg Peacemana (1978, 1983):

    WI = 2 pi sqrt(kx ky) h / (ln(r_eq / r_w) + s + D q)
    r_eq = 0,28 sqrt( sqrt(ky/kx) dx^2 + sqrt(kx/ky) dy^2 ) / ( (ky/kx)^0,25 + (kx/ky)^0,25 )
    q_i  = WI * (a/2) * (m_i - m_wf)   gdy m_i > m_wf, w przeciwnym razie 0

Człon D*q (Forchheimer) liczony jest z wydatku z poprzedniego kroku czasowego.
Jednostka D to doba na tysiąc m3 std, czyli q podstawiane jest w tys. m3 std/d.

SCHEMAT NUMERYCZNY
Całkowanie niejawne (Euler wstecz) z metodą Newtona na zmiennej x = m(p)/m_ref,
zgodnie z podejściem opisanym przez Lie (2016, rozdz. 7.1-7.2, s. 201-211).
Schemat jest bezwarunkowo stabilny, więc krok czasowy ogranicza tylko dokładność,
a nie stabilność (schemat jawny wymagałby tu dt rzędu 0,5 doby).

LITERATURA
Al-Hussainy i in. (1966) - pseudociśnienie gazu rzeczywistego.
Peaceman (1978, 1983)    - ciśnienie w bloku z odwiertem, promień zastępczy.
Lie (2016)               - TPFA, schemat niejawny, model odwiertu.
Pełne opisy bibliograficzne w pliku BIBLIOGRAFIA.md.

JEDNOSTKI INTERFEJSU
ciśnienie [bar], k [mD], h [m], dx/dy [m], czas [doba], wydatek [m3 std/d],
D [d/tys. m3 std], zasoby i produkcja [m3 std].

UKŁAD TABLIC
Tablice mają kształt (nx, ny): pierwszy indeks to oś X, drugi oś Y.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

MD_NA_M2 = 9.869233e-16
BAR_NA_PA = 1.0e5
DOBA = 86400.0
P_SC = 101325.0        # [Pa]
T_SC = 288.15          # [K]
PROMIEN_ODWIERTU = 0.1524   # [m], 6 cali


class SymulatorFVM:
    def __init__(self, model_statyczny, model_pvt, p_poczatek_bar=170.0, odwierty=None,
                 lambda_anizo=1.0, h=20.0, s_wc=0.25, rw=PROMIEN_ODWIERTU,
                 tol_bar=1e-6, max_iter=30):
        self.statyka = model_statyczny
        self.pvt = model_pvt

        self.nx = int(model_statyczny.nx)
        self.ny = int(model_statyczny.ny)
        self.dx = float(model_statyczny.dx)
        self.dy = float(model_statyczny.dy)

        if self.nx == 0 or self.ny == 0 or model_statyczny.phi is None:
            raise ValueError("Model statyczny jest pusty - najpierw wczytaj dane złoża.")

        self.lambda_anizo = float(lambda_anizo)
        self.h = float(h)
        self.s_wc = float(s_wc)
        self.rw = float(rw)
        self.tol_bar = float(tol_bar)
        self.max_iter = int(max_iter)

        # Kopie pól statycznych: późniejsza zmiana obiektu złoża nie wpływa
        # na trwającą symulację.
        self.phi = np.array(model_statyczny.phi, dtype=float, copy=True)
        self.k_mD = np.array(model_statyczny.k_mD, dtype=float, copy=True)
        if self.phi.shape != (self.nx, self.ny) or self.k_mD.shape != (self.nx, self.ny):
            raise ValueError("Kształt pól phi i k nie zgadza się z rozmiarem siatki.")

        self.T_K = float(self.pvt.temp_zloza_K)
        self.a = T_SC / (P_SC * self.T_K)

        p_min, p_max = self.pvt.zakres_tabeli()
        if not (p_min < float(p_poczatek_bar) < p_max):
            raise ValueError(f"Ciśnienie początkowe poza zakresem tabeli PVT ({p_min}-{p_max} bar).")

        self.p = np.full((self.nx, self.ny), float(p_poczatek_bar), dtype=float)
        self._p0 = self.p.copy()

        self.odwierty = self._przygotuj_liste_odwiertow(odwierty)

        self._zbuduj_tabele_pvt()
        self._zbuduj_macierz()
        self._przelicz_objetosc()
        self._przygotuj_odwierty()

        self.calkowity_czas_dni = 0.0
        self.historia_czasu = []
        self.historia_wydatku = []                 # [m3 std/d] suma odwiertów
        self.historia_odwiertow = [[] for _ in self.odwierty]
        self.historia_p_wf = [[] for _ in self.odwierty]          # [bar]
        self.historia_trybu = [[] for _ in self.odwierty]
        self.produkcja_odwiertow = [0.0 for _ in self.odwierty]   # [m3 std]
        self.produkcja_skumulowana = 0.0           # [m3 std]
        self.historia_gp = [0.0]
        self.historia_pz = [self._pz_srednie()]    # [bar], ważone objętością porową
        self.historia_p_avg = [self._p_srednie()]
        self.historia_cisnien_3d = [self.p.copy()]
        self.aktualny_krok = 0
        self.interwal_zapisu = 1
        self.ostrzezenia = []

    # ------------------------------------------------------------ przygotowanie
    def _przygotuj_liste_odwiertow(self, odwierty):
        if odwierty is None:
            odwierty = [{'i': self.nx // 2, 'j': self.ny // 2}]
        lista = []
        zajete = set()
        for nr, otw in enumerate(odwierty, start=1):
            o = dict(otw)
            o['i'] = int(np.clip(int(o.get('i', self.nx // 2)), 0, self.nx - 1))
            o['j'] = int(np.clip(int(o.get('j', self.ny // 2)), 0, self.ny - 1))
            if (o['i'], o['j']) in zajete:
                raise ValueError(f"Dwa odwierty umieszczono w tej samej komórce ({o['i']}, {o['j']}).")
            zajete.add((o['i'], o['j']))
            o.setdefault('nazwa', f"O-{nr}")
            o.setdefault('skin', 0.0)
            o.setdefault('D', 0.0)
            o.setdefault('p_dna', 50.0)
            # Sterowanie: q_max to zadany wydatek (odpowiednik GRAT), p_dna to
            # minimalne ciśnienie denne (ograniczenie BHP), q_min to ekonomiczny
            # próg zamknięcia odwiertu (odpowiednik MinGRAT w WECON).
            o['q_max'] = float(o.get('q_max') or np.inf)
            o['q_min'] = float(o.get('q_min') or 0.0)
            if o['q_max'] <= 0:
                raise ValueError(f"Odwiert {o['nazwa']}: zadany wydatek musi być dodatni.")
            if o['q_min'] >= o['q_max']:
                raise ValueError(f"Odwiert {o['nazwa']}: próg ekonomiczny nie może być "
                                 "większy od zadanego wydatku.")
            o['q_last'] = float(o.get('q_last', 0.0))
            o['czynny'] = True
            o['tryb'] = 'wydatek' if np.isfinite(o['q_max']) else 'ciśnienie denne'
            o['p_wf'] = o['p_dna']
            o['czas_zamkniecia'] = None
            lista.append(o)
        return lista

    def _zbuduj_tabele_pvt(self):
        p_tab, m_si, pz_pa = self.pvt.tabela_pseudocisnienia_si()
        self._m_ref = float(m_si[-1])
        self._p_tab = p_tab
        self._x_tab = m_si / self._m_ref            # zmienna bezwymiarowa x = m/m_ref
        self._g_tab = pz_pa                         # p/Z [Pa]
        self._nachylenie = np.diff(self._g_tab) / np.diff(self._x_tab)

    def _zbuduj_macierz(self):
        """Macierz transmisyjności: strumień netto bloku = (L @ x)."""
        nx, ny = self.nx, self.ny
        kx = np.maximum(self.k_mD, 1e-3) * MD_NA_M2
        ky = kx * self.lambda_anizo
        self._kx, self._ky = kx, ky
        idx = np.arange(nx * ny).reshape(nx, ny)

        k1, k2 = kx[:-1, :], kx[1:, :]
        T_x = (2.0 * k1 * k2 / (k1 + k2)) * self.dy * self.h / self.dx
        a0, b0 = idx[:-1, :].ravel(), idx[1:, :].ravel()

        k1, k2 = ky[:, :-1], ky[:, 1:]
        T_y = (2.0 * k1 * k2 / (k1 + k2)) * self.dx * self.h / self.dy
        a1, b1 = idx[:, :-1].ravel(), idx[:, 1:].ravel()

        a = np.concatenate([a0, a1])
        b = np.concatenate([b0, b1])
        c = 0.5 * self.a * self._m_ref * np.concatenate([T_x.ravel(), T_y.ravel()])

        n = nx * ny
        self._L = sp.csr_matrix(
            (np.concatenate([c, c, -c, -c]),
             (np.concatenate([a, b, a, b]), np.concatenate([a, b, b, a]))),
            shape=(n, n))

    def _przelicz_objetosc(self):
        phi = np.maximum(self.phi, 1e-3)
        V = self.dx * self.dy * self.h
        self._Vgas = V * phi * (1.0 - self.s_wc)        # objętość porowa gazu [m3]
        self._Vgas_a = self._Vgas * self.a              # [m3 std / Pa(p/Z)]

    def _przygotuj_odwierty(self):
        i = np.array([o['i'] for o in self.odwierty], dtype=int)
        j = np.array([o['j'] for o in self.odwierty], dtype=int)
        self._iw = i * self.ny + j
        kx, ky = self._kx[i, j], self._ky[i, j]
        r = np.sqrt(ky / kx)
        req = 0.28 * np.sqrt(r * self.dx**2 + self.dy**2 / r) / (np.sqrt(r) + 1.0 / np.sqrt(r))
        self._ln_req = np.log(req / self.rw)
        self._req = req
        self._wi_licznik = 2.0 * np.pi * np.sqrt(kx * ky) * self.h

    # ------------------------------------------------------- funkcje pomocnicze
    def _g(self, x):
        """p/Z [Pa] jako funkcja x (interpolacja kawałkami liniowa) i jej pochodna."""
        xt = self._x_tab
        idx = np.clip(np.searchsorted(xt, x, side='right') - 1, 0, len(xt) - 2)
        nachylenie = self._nachylenie[idx]
        return self._g_tab[idx] + nachylenie * (x - xt[idx]), nachylenie

    def _p_na_x(self, p_bar):
        return np.interp(p_bar, self._p_tab, self._x_tab)

    def _x_na_p(self, x):
        return np.interp(x, self._x_tab, self._p_tab)

    def _zasoby(self, p_bar):
        """Zasoby gazu [m3 std] odpowiadające zadanemu polu ciśnień."""
        z = self.pvt.oblicz_z_factor(p_bar)
        return float(np.sum(self._Vgas_a * p_bar * BAR_NA_PA / z))

    def _pz_srednie(self):
        z = self.pvt.oblicz_z_factor(self.p)
        return float(np.sum(self._Vgas * self.p / z) / np.sum(self._Vgas))

    def _p_srednie(self):
        return float(np.sum(self._Vgas * self.p) / np.sum(self._Vgas))

    def _parametry_odwiertow(self):
        """
        Zwraca (cw, x_wf_min, q_cel, czynne):
          cw        - współczynnik taki, że wydatek przy sterowaniu ciśnieniem
                      dennym wynosi q = cw * (x_bloku - x_wf_min)  [m3 std/s]
          x_wf_min  - pseudociśnienie odpowiadające minimalnemu P_wf
          q_cel     - zadany wydatek (nieskończoność, gdy nie ograniczono) [m3 std/s]
          czynne    - które odwierty nie zostały zamknięte
        """
        skin = np.array([o['skin'] + o['D'] * (o['q_last'] / 1000.0) for o in self.odwierty])
        mianownik = np.maximum(self._ln_req + skin, 0.1)
        wi = self._wi_licznik / mianownik
        p_wf = np.clip([o['p_dna'] for o in self.odwierty], self._p_tab[0], self._p_tab[-1])
        q_cel = np.array([o['q_max'] for o in self.odwierty]) / DOBA
        czynne = np.array([o['czynny'] for o in self.odwierty])
        return wi * 0.5 * self.a * self._m_ref, self._p_na_x(p_wf), q_cel, czynne

    # ------------------------------------------------------------ krok czasowy
    def krok_czasowy(self, dt_dni):
        dt_dni = float(dt_dni)
        if dt_dni <= 0:
            raise ValueError("Krok czasowy musi być dodatni.")
        dt_s = dt_dni * DOBA
        n = self.nx * self.ny
        iw = self._iw
        cw, x_wf, q_cel, czynne = self._parametry_odwiertow()

        x_stare = self._p_na_x(self.p.ravel())
        g_stare, _ = self._g(x_stare)
        akumulacja = self._Vgas_a.ravel() / dt_s

        def wydatki(x):
            """
            Sterowanie odwiertem zgodnie z logiką symulatorów złożowych
            (Pettersen, 2006, rozdz. 8.1): trybem głównym jest zadany wydatek,
            a minimalne ciśnienie denne stanowi ograniczenie. Dopóki wydatek
            możliwy do uzyskania przy tym ciśnieniu przewyższa zadany, odwiert
            pracuje na zadanym wydatku; gdy złoże przestaje go dostarczać,
            sterowanie przechodzi samoczynnie na ciśnienie denne, a wydatek
            maleje. Formalnie q = min(q_zadany, q_przy_P_wf_min).
            """
            dostepny = np.where(x[iw] > x_wf, cw * (x[iw] - x_wf), 0.0)
            na_cisnieniu = dostepny <= q_cel
            q = np.where(na_cisnieniu, dostepny, q_cel)
            # Przy sterowaniu wydatkiem człon źródłowy nie zależy od ciśnienia,
            # więc jego pochodna znika - odwiert staje się stałym źródłem.
            dq = np.where(na_cisnieniu, cw, 0.0)
            return np.where(czynne, q, 0.0), np.where(czynne, dq, 0.0)

        x = x_stare.copy()
        zbiezny, dp = False, np.inf
        for _ in range(self.max_iter):
            g, dg = self._g(x)
            q_otw, dq_otw = wydatki(x)
            q = np.bincount(iw, weights=q_otw, minlength=n)
            dq = np.bincount(iw, weights=dq_otw, minlength=n)

            R = akumulacja * (g - g_stare) + self._L @ x + q
            J = (self._L + sp.diags(akumulacja * dg + dq)).tocsc()
            x_nowe = np.clip(x + spsolve(J, -R), 0.0, 1.0)

            dp = float(np.max(np.abs(self._x_na_p(x_nowe) - self._x_na_p(x))))
            x = x_nowe
            if dp < self.tol_bar:
                zbiezny = True
                break
        if not zbiezny:
            self.ostrzezenia.append(
                f"Brak zbieżności metody Newtona w t = {self.calkowity_czas_dni + dt_dni:.1f} d "
                f"(ostatnia poprawka {dp:.2e} bar). Zmniejsz krok czasowy.")

        q_otw, _ = wydatki(x)                    # [m3 std/s]
        q_na_dobe = q_otw * DOBA
        self.p = self._x_na_p(x).reshape(self.nx, self.ny)
        self.calkowity_czas_dni += dt_dni

        # Rzeczywiste ciśnienie denne: przy sterowaniu ciśnieniem równe zadanemu
        # minimum, przy sterowaniu wydatkiem wyższe, bo odwiert nie wykorzystuje
        # pełnej dostępnej depresji. Różnica między nimi pokazuje zapas odwiertu.
        x_bloku = x[iw]
        for nr, otw in enumerate(self.odwierty):
            if not otw['czynny']:
                otw.update(tryb='zamknięty', p_wf=float(self.p[otw['i'], otw['j']]), q_last=0.0)
            elif q_na_dobe[nr] <= 0.0:
                otw.update(tryb='bez dopływu', p_wf=otw['p_dna'], q_last=0.0)
            elif q_otw[nr] >= q_cel[nr] - 1e-12 and np.isfinite(q_cel[nr]):
                x_denne = max(float(x_bloku[nr] - q_cel[nr] / cw[nr]), float(self._x_tab[0]))
                otw.update(tryb='wydatek', p_wf=float(self._x_na_p(x_denne)),
                           q_last=float(q_na_dobe[nr]))
            else:
                otw.update(tryb='ciśnienie denne', p_wf=otw['p_dna'],
                           q_last=float(q_na_dobe[nr]))

        # Ograniczenie ekonomiczne: odwiert, którego wydatek spadł poniżej progu,
        # zostaje zamknięty na stałe (odpowiednik MinGRAT w słowie WECON).
        for nr, otw in enumerate(self.odwierty):
            if otw['czynny'] and otw['q_min'] > 0 and 0.0 <= q_na_dobe[nr] < otw['q_min']:
                otw['czynny'] = False
                otw['czas_zamkniecia'] = self.calkowity_czas_dni
                self.ostrzezenia.append(
                    f"Odwiert {otw['nazwa']} zamknięty w {self.calkowity_czas_dni:.0f} dniu: "
                    f"wydatek {q_na_dobe[nr]:.0f} m3/d spadł poniżej progu "
                    f"{otw['q_min']:.0f} m3/d.")

        wydatek = float(np.sum(q_na_dobe))
        self.produkcja_skumulowana += wydatek * dt_dni

        self.historia_czasu.append(self.calkowity_czas_dni)
        self.historia_wydatku.append(wydatek)
        for nr, qd in enumerate(q_na_dobe):
            self.historia_odwiertow[nr].append(float(qd))
            self.historia_p_wf[nr].append(float(self.odwierty[nr]['p_wf']))
            self.historia_trybu[nr].append(self.odwierty[nr]['tryb'])
            self.produkcja_odwiertow[nr] += float(qd) * dt_dni
        self.historia_gp.append(self.produkcja_skumulowana)
        self.historia_pz.append(self._pz_srednie())
        self.historia_p_avg.append(self._p_srednie())

        self.aktualny_krok += 1
        if self.aktualny_krok % self.interwal_zapisu == 0:
            self.historia_cisnien_3d.append(self.p.copy())
        return self.p

    def uruchom(self, czas_dni, dt_dni, callback=None):
        """Wykonuje serię kroków; callback(krok, liczba_krokow) służy paskowi postępu."""
        liczba_krokow = max(1, int(round(float(czas_dni) / float(dt_dni))))
        for krok in range(liczba_krokow):
            self.krok_czasowy(dt_dni)
            if callback is not None:
                callback(krok + 1, liczba_krokow)
        return self.p

    # -------------------------------------------------------- zasoby i bilans
    def oblicz_igip(self):
        """
        Zasoby pierwotne gazu [m3 std] w całym złożu, liczone z ciśnienia
        początkowego. Sumowane są wszystkie bloki siatki, zgodnie z tym, z
        których bloków pochodzi produkcja.
        """
        return self._zasoby(self._p0)

    def wspolczynnik_sczerpania(self):
        """Stopień sczerpania zasobów [%]."""
        igip = self.oblicz_igip()
        return 100.0 * self.produkcja_skumulowana / igip if igip > 0 else 0.0

    def bilans_materialowy(self):
        """
        Kontrola zachowania masy: produkcja skumulowana powinna być równa
        ubytkowi zasobów wynikającemu z obliczonego pola ciśnień. Błąd
        względny rzędu 1e-6 i mniejszy świadczy o poprawnym rozwiązaniu.
        """
        g0 = self._zasoby(self._p0)
        ubytek = g0 - self._zasoby(self.p)
        gp = self.produkcja_skumulowana
        return {'igip': g0, 'gp': gp, 'ubytek_zasobow': ubytek,
                'blad_wzgledny': (gp - ubytek) / ubytek if ubytek > 0 else 0.0}

    def diagnostyka_odwiertow(self):
        """Parametry odwiertów do tabeli w pracy."""
        wiersze = []
        for nr, otw in enumerate(self.odwierty):
            na_wydatku = [t == 'wydatek' for t in self.historia_trybu[nr]]
            wiersze.append({
                'Odwiert': otw['nazwa'],
                'I': otw['i'], 'J': otw['j'],
                'k [mD]': float(self.k_mD[otw['i'], otw['j']]),
                'r_eq [m]': float(self._req[nr]),
                'Zadany wydatek [m3/d]': ('bez ograniczenia' if not np.isfinite(otw['q_max'])
                                          else f"{otw['q_max']:.0f}"),
                'P_wf min [bar]': otw['p_dna'],
                'Skin': otw['skin'],
                'q końcowe [m3/d]': otw['q_last'],
                'P_wf końcowe [bar]': otw['p_wf'],
                'Tryb sterowania': otw['tryb'],
                'Czas na półce [dni]': (sum(na_wydatku) / len(na_wydatku) * self.calkowity_czas_dni
                                        if na_wydatku else 0.0),
                'Produkcja [mln m3]': self.produkcja_odwiertow[nr] / 1e6,
            })
        return wiersze

    def historia_do_ramki(self):
        """Historia produkcji i ciśnień jako tablica rekordów (do eksportu CSV)."""
        import pandas as pd
        dane = {
            'czas_dni': self.historia_czasu,
            'wydatek_m3_d': self.historia_wydatku,
            'produkcja_skum_m3': self.historia_gp[1:],
            'p_srednie_bar': self.historia_p_avg[1:],
            'pz_bar': self.historia_pz[1:],
        }
        for nr, otw in enumerate(self.odwierty):
            dane[f"q_{otw['nazwa']}_m3_d"] = self.historia_odwiertow[nr]
            dane[f"pwf_{otw['nazwa']}_bar"] = self.historia_p_wf[nr]
            dane[f"tryb_{otw['nazwa']}"] = self.historia_trybu[nr]
        return pd.DataFrame(dane)