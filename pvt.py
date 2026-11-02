"""
pvt.py - właściwości gazu rzeczywistego (moduł PVT).

Korelacje:
  * parametry pseudokrytyczne           - Standing (1977)
  * współczynnik ściśliwości Z          - Hall i Yarborough (1973)
  * lepkość dynamiczna                  - Lee, Gonzalez i Eakin (1966)
  * pseudociśnienie m(p)                - Al-Hussainy, Ramey i Crawford (1966)

Zakres ważności korelacji Lee-Gonzalez-Eakin: 0,7-55 MPa i 38-171 °C, dla gazów
o niskiej zawartości składników niewęglowodorowych. Poza tym zakresem wyniki
należy traktować jako przybliżone.

Jednostki interfejsu: ciśnienie w barach, temperatura w °C (konstruktor),
lepkość w cP, m(p) w bar^2/cP. Metody z przyrostkiem _si zwracają wielkości
w układzie SI i służą silnikowi przepływu.
"""
import numpy as np

BAR_NA_PA = 1.0e5
BAR_NA_PSI = 14.5038
P_SC_BAR = 1.01325      # ciśnienie normalne
T_SC_K = 288.15         # temperatura normalna (15 °C)

# Zakres i rozdzielczość tabeli pseudociśnienia
P_TAB_MIN_BAR = 0.5
P_TAB_MAX_BAR = 600.0
P_TAB_KROK_BAR = 0.2


class TermodynamikaGazu:
    def __init__(self, gamma_g=0.58, temp_zloza_C=60.0):
        self.gamma_g = float(gamma_g)
        self.temp_zloza_C = float(temp_zloza_C)
        self.temp_zloza_K = self.temp_zloza_C + 273.15
        self.temp_zloza_R = self.temp_zloza_K * 1.8
        self.M_w = 28.97 * self.gamma_g

        # Standing (1977) - parametry pseudokrytyczne gazu suchego
        P_pc_psi = 677.0 + 15.0 * self.gamma_g - 37.5 * self.gamma_g**2
        T_pc_R = 168.0 + 325.0 * self.gamma_g - 12.5 * self.gamma_g**2
        self.P_pc_bar = P_pc_psi / BAR_NA_PSI
        self.T_pc_K = T_pc_R / 1.8
        self.T_r = self.temp_zloza_K / self.T_pc_K

        self._zbuduj_tabele()

    # ------------------------------------------------------------- korelacje
    def oblicz_z_factor(self, p_bar):
        """Współczynnik ściśliwości Z metodą Halla-Yarborough (Newton-Raphson)."""
        p_bar = np.asarray(p_bar, dtype=float)
        P_r = np.clip(p_bar / self.P_pc_bar, 1e-3, 30.0)
        t = 1.0 / self.T_r

        A = 0.06125 * t * np.exp(-1.2 * (1.0 - t)**2)
        B = t * (14.76 - 9.76 * t + 4.58 * t**2)
        C = t * (90.7 - 242.2 * t + 42.4 * t**2)
        D = 2.18 + 2.82 * t

        y = np.clip(0.0125 * P_r * t * np.exp(-1.2 * (1.0 - t)**2), 1e-10, 0.99)
        for _ in range(40):
            f = -A * P_r + (y + y**2 + y**3 - y**4) / (1.0 - y)**3 - B * y**2 + C * y**D
            df = ((1.0 + 4.0 * y + 4.0 * y**2 - 2.0 * y**3 + y**4) / (1.0 - y)**4
                  - 2.0 * B * y + C * D * y**(D - 1.0))
            df = np.where(np.abs(df) < 1e-12, 1e-12, df)
            y_nowe = np.clip(y - f / df, 1e-10, 0.99)
            if np.max(np.abs(y_nowe - y)) < 1e-12:
                y = y_nowe
                break
            y = y_nowe
        return A * P_r / y

    def oblicz_lepkosc(self, p_bar, z=None):
        """Lepkość dynamiczna gazu [cP] wg Lee-Gonzalez-Eakin."""
        p_bar = np.asarray(p_bar, dtype=float)
        if z is None:
            z = self.oblicz_z_factor(p_bar)
        p_psi = p_bar * BAR_NA_PSI
        rho_g = (1.4935e-3 * p_psi * self.M_w) / (z * self.temp_zloza_R)   # g/cm3
        K = ((9.4 + 0.02 * self.M_w) * self.temp_zloza_R**1.5) / (209.0 + 19.0 * self.M_w + self.temp_zloza_R)
        X = 3.5 + 986.0 / self.temp_zloza_R + 0.01 * self.M_w
        Y = 2.4 - 0.2 * X
        return 1e-4 * K * np.exp(X * rho_g**Y)

    def oblicz_bg(self, p_bar, z=None):
        """Współczynnik objętościowy gazu Bg [m3/Nm3]."""
        p_bar = np.asarray(p_bar, dtype=float)
        if z is None:
            z = self.oblicz_z_factor(p_bar)
        return (P_SC_BAR / T_SC_K) * (self.temp_zloza_K / np.maximum(p_bar, 1e-3)) * z

    def oblicz_scisliwosc(self, p_bar):
        """Ściśliwość izotermiczna gazu c_g = 1/p - (1/Z)(dZ/dp) [1/bar]."""
        p_bar = np.asarray(p_bar, dtype=float)
        dp = 0.05
        z = self.oblicz_z_factor(p_bar)
        dz = (self.oblicz_z_factor(p_bar + dp) - self.oblicz_z_factor(np.maximum(p_bar - dp, 1e-3))) / (2 * dp)
        return 1.0 / np.maximum(p_bar, 1e-3) - dz / z

    # -------------------------------------------------- pseudociśnienie m(p)
    def _zbuduj_tabele(self):
        """
        Tablicuje m(p) raz, na stałej siatce ciśnień, metodą trapezów:
            m(p) = calka od p_ref do p z 2 p' / (mu(p') Z(p')) dp'
        Dzięki stałej siatce wynik nie zależy od tego, dla jakiego ciśnienia
        funkcja zostanie wywołana. Dolna granica całkowania jest stała, a
        w równaniach przepływu i tak występują wyłącznie różnice m(p).
        """
        n = int(round((P_TAB_MAX_BAR - P_TAB_MIN_BAR) / P_TAB_KROK_BAR)) + 1
        p_bar = np.linspace(P_TAB_MIN_BAR, P_TAB_MAX_BAR, n)
        z = self.oblicz_z_factor(p_bar)
        mu_cP = self.oblicz_lepkosc(p_bar, z)

        f_bar = 2.0 * p_bar / (mu_cP * z)                      # [bar/cP]
        m_bar = np.concatenate(([0.0], np.cumsum(0.5 * (f_bar[1:] + f_bar[:-1]) * np.diff(p_bar))))

        mu_pa_s = np.maximum(mu_cP * 1e-3, 1e-9)
        p_pa = p_bar * BAR_NA_PA
        f_si = 2.0 * p_pa / (mu_pa_s * z)                      # [Pa/s]
        m_si = np.concatenate(([0.0], np.cumsum(0.5 * (f_si[1:] + f_si[:-1]) * np.diff(p_pa))))

        self._p_tab = p_bar
        self._m_tab_bar = m_bar                                 # [bar^2/cP]
        self._m_tab_si = m_si                                   # [Pa/s]
        self._pz_tab_pa = p_pa / z                              # p/Z [Pa]
        self._z_tab = z

    def oblicz_pseudocisnienie(self, p_bar):
        """Pseudociśnienie m(p) [bar^2/cP] wg Al-Hussainy et al. (1966)."""
        return np.interp(np.asarray(p_bar, dtype=float), self._p_tab, self._m_tab_bar)

    def tabela_pseudocisnienia_si(self):
        """
        Tabele dla silnika przepływu: ciśnienie [bar], m(p) [Pa/s], p/Z [Pa].
        Zwracane są kopie, żeby rozmówca nie mógł zepsuć stanu obiektu.
        """
        return self._p_tab.copy(), self._m_tab_si.copy(), self._pz_tab_pa.copy()

    def zakres_tabeli(self):
        return float(self._p_tab[0]), float(self._p_tab[-1])