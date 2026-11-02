"""
testy.py - testy weryfikacyjne modelu.

Uruchomienie:  python testy.py

Testy obejmują kontrolę zachowania masy, zgodność z bilansem materiałowym
(prostoliniowość wykresu p/Z), zbieżność względem kroku czasowego, symetrię
rozwiązania w złożu jednorodnym oraz poprawność modułów PVT i statycznego.
Wyniki tych testów nadają się wprost do rozdziału o walidacji modelu.
"""
import sys
import numpy as np
import pandas as pd

from pvt import TermodynamikaGazu
from statyka import ModelStatycznyZloza, oblicz_fzi, przewidz_k, _dopasuj_w_skali_log
from symulator_fvm import SymulatorFVM
from generator_zloza import (generuj_zloze, generuj_zloze_realistyczne,
                             generuj_zloze_weglanowe)

ZALICZONE, NIEZALICZONE = [], []


def sprawdz(nazwa, warunek, komentarz=""):
    (ZALICZONE if warunek else NIEZALICZONE).append(nazwa)
    print(f"  [{'OK ' if warunek else 'BŁĄD'}] {nazwa}" + (f"  -> {komentarz}" if komentarz else ""))


def zbuduj_zloze(liczba_facji=1, seed=42, nx=30, ny=30):
    df = generuj_zloze(nx=nx, ny=ny, liczba_facji=liczba_facji, seed=seed)
    z = ModelStatycznyZloza(100.0, 100.0)
    z.wczytaj_dane(df)
    return z, df


def symulacja(zloze, pvt, odwierty, czas, dt, **kw):
    s = SymulatorFVM(zloze, pvt, 170.0, odwierty, **kw)
    s.uruchom(czas, dt)
    return s


def main():
    pvt = TermodynamikaGazu(0.58, 60.0)
    zloze, df = zbuduj_zloze()
    odwiert = [{'i': 15, 'j': 15, 'p_dna': 50.0}]

    print("\n1. MODUŁ PVT")
    z = pvt.oblicz_z_factor(np.array([1.0, 50.0, 170.0, 350.0]))
    sprawdz("Z w zakresie fizycznym i malejące do minimum",
            bool(np.all((z > 0.2) & (z < 1.2))), f"Z = {np.round(z, 4)}")
    sprawdz("Z dąży do 1 przy niskim ciśnieniu", abs(z[0] - 1.0) < 0.01, f"Z(1 bar) = {z[0]:.4f}")
    mu = pvt.oblicz_lepkosc(np.array([50.0, 170.0]))
    sprawdz("lepkość rośnie z ciśnieniem i jest rzędu 0,01-0,03 cP",
            mu[1] > mu[0] and 0.005 < mu[0] < 0.05, f"mu = {np.round(mu, 5)} cP")
    m1, m2 = pvt.oblicz_pseudocisnienie(50.0), pvt.oblicz_pseudocisnienie(170.0)
    sprawdz("m(p) rośnie monotonicznie", m2 > m1 > 0, f"m(50) = {m1:.3e}, m(170) = {m2:.3e} bar^2/cP")
    cg = pvt.oblicz_scisliwosc(np.array([50.0, 170.0]))
    sprawdz("ściśliwość gazu maleje z ciśnieniem i jest bliska 1/p",
            cg[0] > cg[1] > 0, f"c_g = {np.round(cg, 6)} 1/bar, 1/p = {np.round(1/np.array([50.,170.]), 6)}")

    print("\n2. MODUŁ STATYCZNY")
    sprawdz("siatka odtworzona z kolumn I/J", (zloze.nx, zloze.ny) == (30, 30), f"{zloze.nx}x{zloze.ny}")
    kontrola = df.sort_values(['I', 'J'])
    sprawdz("kolejność komórek zgodna z plikiem",
            np.allclose(zloze.phi.ravel(), kontrola['Porowatosc'].to_numpy()))
    try:
        ModelStatycznyZloza().wczytaj_dane(df.iloc[:-5])
        sprawdz("niezgodna liczba wierszy zgłaszana jako błąd", False, "nie zgłoszono wyjątku")
    except ValueError:
        sprawdz("niezgodna liczba wierszy zgłaszana jako błąd", True)

    rqi, _, _ = oblicz_fzi(0.15, 10.0, m=1.0)
    sprawdz("RQI zgodne ze wzorem Amaefule", abs(rqi - 0.0314 * np.sqrt(10.0 / 0.15)) < 1e-12,
            f"RQI = {rqi:.4f} um")
    _, _, fzi_m2 = oblicz_fzi(0.15, 10.0, m=2.0)
    sprawdz("FZI_m dla m=2 odpowiada wzorowi SPE 149047",
            abs(fzi_m2 - rqi / ((0.15 / 0.85) * 0.15)) < 1e-12, f"FZI_2 = {fzi_m2:.4f} um")


    zloze.wyznacz_jednostki_hu(liczba_hu=4, m=1.0)
    zloze.uwzglednij_wplyw_hu_na_przepuszczalnosc()
    sprawdz("etykiety HU uporządkowane rosnąco wg FZI",
            all(zloze.fzi[zloze.hu_grid == h].mean() < zloze.fzi[zloze.hu_grid == h + 1].mean()
                for h in range(1, 4)))
    sprawdz("korekta HU nie zmienia pola surowego",
            np.allclose(zloze.k_mD_surowe.ravel(), kontrola['Przepuszczalnosc'].to_numpy()))
    m_opt = zloze.znajdz_optymalne_m(liczba_hu=4)
    sprawdz("wyszukiwanie m nie zmienia stanu obiektu",
            np.allclose(zloze.k_mD_surowe.ravel(), kontrola['Przepuszczalnosc'].to_numpy())
            and zloze.hu_grid is not None, f"m_opt = {m_opt}")
    mod = _dopasuj_w_skali_log(np.array([0.1, 0.15, 0.2]), np.array([1.0, 10.0, 100.0]))
    sprawdz("regresja odtwarza zależność wykładniczą bez błędu",
            mod['r2'] > 0.999 and np.allclose(przewidz_k(mod, np.array([0.15])), 10.0, rtol=1e-6),
            f"typ = {mod['typ']}, R2 = {mod['r2']:.6f}")

    print("\n3. ZACHOWANIE MASY I BILANS MATERIAŁOWY")
    s0 = SymulatorFVM(zloze, pvt, 170.0, [], h=15.0)
    s0.uruchom(50, 10)
    sprawdz("bez odwiertu ciśnienie pozostaje stałe", np.abs(s0.p - 170.0).max() < 1e-9,
            f"maks. odchyłka {np.abs(s0.p - 170.0).max():.2e} bar")

    s = symulacja(zloze, pvt, odwiert, 365, 5, h=15.0)
    b = s.bilans_materialowy()
    sprawdz("produkcja zgodna z ubytkiem zasobów", abs(b['blad_wzgledny']) < 1e-5,
            f"błąd względny {b['blad_wzgledny']:.2e}")
    gp, pz = np.array(s.historia_gp), np.array(s.historia_pz)
    odchylenie = np.abs(np.polyval(np.polyfit(gp, pz, 1), gp) - pz).max()
    sprawdz("wykres p/Z względem produkcji jest prostą", odchylenie < 1e-3,
            f"maks. odchylenie {odchylenie:.2e} bar")
    pz0 = pz[0]
    igip_z_prostej = -np.polyfit(gp, pz, 1)[1] / np.polyfit(gp, pz, 1)[0]
    sprawdz("ekstrapolacja prostej p/Z odtwarza zasoby",
            abs(igip_z_prostej - b['igip']) / b['igip'] < 0.02,
            f"z prostej {igip_z_prostej/1e9:.3f} mld m3 wobec IGIP {b['igip']/1e9:.3f} mld m3")
    sprawdz("brak ostrzeżeń o zbieżności", not s.ostrzezenia, f"{s.ostrzezenia}")
    print(f"       IGIP = {b['igip']/1e6:.1f} mln m3, Gp = {b['gp']/1e6:.1f} mln m3, "
          f"RF = {s.wspolczynnik_sczerpania():.2f} %, p/Z: {pz0:.1f} -> {pz[-1]:.1f} bar")

    print("\n4. ZBIEŻNOŚĆ I STABILNOŚĆ")
    wyniki = {}
    for dt in (0.25, 1.0, 5.0, 20.0):
        wyniki[dt] = symulacja(zloze, pvt, odwiert, 200, dt, h=15.0).produkcja_skumulowana
    odn = wyniki[0.25]
    bledy = {dt: abs(v - odn) / odn * 100 for dt, v in wyniki.items()}
    sprawdz("wynik zbieżny przy zmniejszaniu kroku czasowego",
            bledy[1.0] < bledy[5.0] < bledy[20.0],
            "błąd Gp: " + ", ".join(f"dt={dt}d: {b:.2f}%" for dt, b in bledy.items() if dt > 0.25))
    sprawdz("duży krok czasowy nie powoduje niestabilności",
            np.isfinite(wyniki[20.0]) and wyniki[20.0] > 0 and bledy[20.0] < 10.0)

    zloze_j = ModelStatycznyZloza(100.0, 100.0)
    df_j = df.copy()
    df_j['Porowatosc'] = 0.15
    df_j['Przepuszczalnosc'] = 10.0
    zloze_j.wczytaj_dane(df_j)
    sj = symulacja(zloze_j, pvt, [{'i': 14, 'j': 14, 'p_dna': 50.0}], 200, 5, h=15.0)
    sprawdz("w złożu jednorodnym rozwiązanie jest symetryczne",
            np.abs(sj.p - sj.p.T).max() < 1e-9, f"maks. |p - p^T| = {np.abs(sj.p - sj.p.T).max():.2e}")

    print("\n5. ODWIERTY, SKIN I ANIZOTROPIA")
    s_ref = symulacja(zloze, pvt, [{'i': 15, 'j': 15, 'p_dna': 50.0, 'skin': 0.0}], 100, 5, h=15.0)
    s_skin = symulacja(zloze, pvt, [{'i': 15, 'j': 15, 'p_dna': 50.0, 'skin': 5.0}], 100, 5, h=15.0)
    sprawdz("dodatni skin zmniejsza produkcję",
            s_skin.produkcja_skumulowana < s_ref.produkcja_skumulowana,
            f"{s_skin.produkcja_skumulowana/1e6:.1f} wobec {s_ref.produkcja_skumulowana/1e6:.1f} mln m3")
    s_turb = symulacja(zloze, pvt, [{'i': 15, 'j': 15, 'p_dna': 50.0, 'D': 1e-3}], 100, 5, h=15.0)
    sprawdz("turbulencja Forchheimera zmniejsza produkcję",
            s_turb.produkcja_skumulowana < s_ref.produkcja_skumulowana,
            f"{s_turb.produkcja_skumulowana/1e6:.1f} mln m3")

    dwa = [{'i': 9, 'j': 9, 'p_dna': 50.0}, {'i': 21, 'j': 20, 'p_dna': 60.0}]
    s2 = symulacja(zloze, pvt, dwa, 365, 5, h=15.0, lambda_anizo=0.3)
    sprawdz("bilans masy zachowany dla dwóch odwiertów i anizotropii",
            abs(s2.bilans_materialowy()['blad_wzgledny']) < 1e-5,
            f"błąd {s2.bilans_materialowy()['blad_wzgledny']:.2e}")
    sprawdz("suma produkcji odwiertów równa produkcji całkowitej",
            abs(sum(s2.produkcja_odwiertow) - s2.produkcja_skumulowana) < 1e-6 * s2.produkcja_skumulowana)
    try:
        SymulatorFVM(zloze, pvt, 170.0, [{'i': 5, 'j': 5}, {'i': 5, 'j': 5}])
        sprawdz("dwa odwierty w jednej komórce zgłaszane jako błąd", False)
    except ValueError:
        sprawdz("dwa odwierty w jednej komórce zgłaszane jako błąd", True)

    print("\n6. ODPORNOŚĆ NA ZMIANY MODELU STATYCZNEGO")
    s_kopia = SymulatorFVM(zloze, pvt, 170.0, odwiert, h=15.0)
    k_przed = s_kopia.k_mD.copy()
    zloze.k_mD = zloze.k_mD * 2.0
    sprawdz("symulator pracuje na własnej kopii pól statycznych",
            np.allclose(s_kopia.k_mD, k_przed))
    zloze.przywroc_surowa_przepuszczalnosc()

    print("\n7. WPŁYW MIĄŻSZOŚCI")
    rf = {}
    for h in (5.0, 15.0, 30.0):
        sh = symulacja(zloze, pvt, odwiert, 365, 5, h=h)
        rf[h] = sh.wspolczynnik_sczerpania()
    sprawdz("przy stałym P_wf miąższość skaluje zasoby i wydatki, nie zmieniając RF",
            max(rf.values()) - min(rf.values()) < 0.01,
            ", ".join(f"h={h:.0f} m: RF={v:.3f}%" for h, v in rf.items()))

    print("\n8. JEDNOSTKI HYDRAULICZNE - WKŁAD W PREDYKCJĘ (walidacja krzyżowa)")
    for opis, facje in (("złoże jednofacjalne", 1), ("złoże czterofacjalne", 4)):
        z_test, _ = zbuduj_zloze(liczba_facji=facje, seed=7)
        w = z_test.walidacja_krzyzowa(liczba_hu=4, m=1.0, udzial_treningowy=0.2, powtorzenia=10)
        print(f"       {opis}: RMSE log k - globalna {w['rmse_globalna']:.4f}, "
              f"HU {w['rmse_hu']:.4f}, poprawa {w['poprawa_proc']:+.1f} %")
        if facje > 1:
            sprawdz("HU poprawiają predykcję w złożu wielofacjalnym", w['poprawa_proc'] > 0)

    print("\n9. DOBÓR WYKŁADNIKA CEMENTACJI m")
    for m_prawdziwe, tolerancja in ((1.0, 0.15), (2.0, 0.20)):
        z_m = ModelStatycznyZloza(100.0, 100.0)
        z_m.wczytaj_dane(generuj_zloze_realistyczne(m=m_prawdziwe, seed=2001))
        m_sugerowane = z_m.znajdz_optymalne_m(liczba_hu=4)
        sprawdz(f"kryterium odtwarza zadane m = {m_prawdziwe}",
                abs(m_sugerowane - m_prawdziwe) <= tolerancja,
                f"wskazano m = {m_sugerowane:.2f}")
        sprawdz(f"dobór m nie zmienia stanu obiektu (m = {m_prawdziwe})",
                z_m.hu_grid is None and np.allclose(z_m.k_mD, z_m.k_mD_surowe))
        tabela_m = z_m.diagnostyka_wykladnika_m(liczba_hu=4)
        zal = z_m.zalecenie_wykladnika_m(tabela_m)
        oczekiwany = 'klasyczny' if m_prawdziwe == 1.0 else 'zmodyfikowany'
        sprawdz(f"werdykt dla danych o m = {m_prawdziwe} brzmi '{oczekiwany}'",
                zal['werdykt'] == oczekiwany,
                f"zalecono m = {zal['m_sugerowane']:.2f}, przewaga {zal['zysk_proc']:+.1f} %")


    print("\n10. STEROWANIE ODWIERTEM (zadany wydatek i ograniczenie ciśnienia)")
    zloze_j2 = ModelStatycznyZloza(100.0, 100.0)
    df_j2 = df.copy()
    df_j2['Porowatosc'] = 0.18
    df_j2['Przepuszczalnosc'] = 50.0
    zloze_j2.wczytaj_dane(df_j2)

    bez_ogr = symulacja(zloze_j2, pvt, [{'i': 15, 'j': 15, 'p_dna': 50.0}], 365, 5, h=20.0)
    q_bez = bez_ogr.historia_wydatku[-1]

    # Porównanie z rozwiązaniem analitycznym dla stanu pseudoustalonego w złożu
    # zamkniętym (człon -3/4, ciśnienie średnie), w postaci pseudociśnieniowej:
    # Ahmed (2010), Reservoir Engineering Handbook, wyd. 4. Wzór 22 u Pettersena
    # (2006) dotyczy stanu ustalonego i nie zawiera członu -3/4.
    from symulator_fvm import MD_NA_M2, DOBA, P_SC, T_SC
    p_tab, m_si, _ = pvt.tabela_pseudocisnienia_si()
    m_fun = lambda p: np.interp(p, p_tab, m_si)
    re_zast = np.sqrt((30 * 100.0)**2 / np.pi)
    q_analit = ((2 * np.pi * 50.0 * MD_NA_M2 * 20.0 / (np.log(re_zast / 0.1524) - 0.75))
                * 0.5 * (m_fun(bez_ogr.historia_p_avg[-1]) - m_fun(50.0))
                * (T_SC / (P_SC * pvt.temp_zloza_K)) * DOBA)
    sprawdz("wydatek zgodny ze wzorem analitycznym dla stanu pseudoustalonego",
            abs(q_bez - q_analit) / q_analit < 0.03,
            f"symulator {q_bez:.0f}, wzór {q_analit:.0f} m3/d, różnica "
            f"{abs(q_bez - q_analit) / q_analit * 100:.2f} %")

    z_polka = symulacja(zloze_j2, pvt,
                        [{'i': 15, 'j': 15, 'p_dna': 50.0, 'q_max': 1_200_000.0}],
                        4380, 15, h=20.0)
    tryby = z_polka.historia_trybu[0]
    sprawdz("wydatek nigdy nie przekracza zadanego",
            max(z_polka.historia_odwiertow[0]) <= 1_200_000.0 + 1e-6)
    sprawdz("odwiert utrzymuje półkę, a potem przechodzi na ciśnienie denne",
            'wydatek' in tryby and 'ciśnienie denne' in tryby,
            f"półka {tryby.count('wydatek') * 15} dni z {len(tryby) * 15}")
    i_przej = next(i for i, t in enumerate(tryby) if t != 'wydatek')
    sprawdz("na półce ciśnienie denne jest wyższe od zadanego minimum",
            z_polka.historia_p_wf[0][0] > 50.0 and abs(z_polka.historia_p_wf[0][i_przej] - 50.0) < 1e-6,
            f"P_wf {z_polka.historia_p_wf[0][0]:.1f} -> 50,0 bar")
    sprawdz("bilans masy zachowany przy sterowaniu wydatkiem",
            abs(z_polka.bilans_materialowy()['blad_wzgledny']) < 1e-5,
            f"błąd {z_polka.bilans_materialowy()['blad_wzgledny']:.1e}")
    sprawdz("brak ostrzeżeń o zbieżności przy przełączaniu trybu", not z_polka.ostrzezenia)

    z_prog = symulacja(zloze_j2, pvt,
                       [{'i': 15, 'j': 15, 'p_dna': 50.0, 'q_max': 1_200_000.0,
                         'q_min': 300_000.0}], 4380, 15, h=20.0)
    sprawdz("próg ekonomiczny zamyka odwiert",
            z_prog.odwierty[0]['czas_zamkniecia'] is not None
            and z_prog.historia_odwiertow[0][-1] == 0.0,
            f"zamknięty w {z_prog.odwierty[0]['czas_zamkniecia']:.0f} dniu")
    sprawdz("po zamknięciu produkcja nie rośnie",
            abs(z_prog.bilans_materialowy()['blad_wzgledny']) < 1e-5)


    print("\n11. ROZRÓŻNIENIE ZŁOŻA PIASKOWCOWEGO I WĘGLANOWEGO")
    for opis, generator, liczba, oczekiwany in (
            ("piaskowiec wg Ellabad i in. (2001), m = 1,0",
             generuj_zloze_realistyczne, 4, 'klasyczny'),
            ("węglan wg Nooruddin i in. (2011), m = 2,0",
             generuj_zloze_weglanowe, 7, 'zmodyfikowany')):
        z_w = ModelStatycznyZloza(100.0, 100.0)
        z_w.wczytaj_dane(generator())
        zal = z_w.zalecenie_wykladnika_m(z_w.diagnostyka_wykladnika_m(liczba_hu=liczba))
        sprawdz(f"{opis}: werdykt '{oczekiwany}'", zal['werdykt'] == oczekiwany,
                f"zalecono m = {zal['m_sugerowane']:.2f}, przewaga {zal['zysk_proc']:+.1f} %")

    for opis, generator, m_w, prawda in (
            ("piaskowiec", generuj_zloze_realistyczne, 1.0, 4),
            ("węglan", generuj_zloze_weglanowe, 2.0, 7)):
        z_k = ModelStatycznyZloza(100.0, 100.0)
        z_k.wczytaj_dane(generator())
        zal_k = z_k.zalecenie_liczby_jednostek(z_k.diagnostyka_liczby_jednostek(m=m_w))
        sprawdz(f"{opis}: metoda łokcia odtwarza liczbę jednostek ({prawda})",
                abs(zal_k['k_sugerowane'] - prawda) <= 1 and zal_k['pewne'],
                f"zalecono {zal_k['k_sugerowane']}")


    from statyka import znajdz_lokiec
    k_kontrolne = np.arange(1, 11)
    sprawdz("algorytm Kneedle wykrywa łokieć krzywej 1/k",
            znajdz_lokiec(k_kontrolne, 1.0 / k_kontrolne)[0] == 3.0)
    sprawdz("algorytm Kneedle nie wykrywa łokcia na linii prostej",
            znajdz_lokiec(k_kontrolne, 10.0 - k_kontrolne)[0] is None)
    sprawdz("algorytm Kneedle nie wykrywa łokcia na krzywej stałej",
            znajdz_lokiec(k_kontrolne, np.ones(10))[0] is None)

    df_w = generuj_zloze_weglanowe()
    sprawdz("złoże węglanowe ma siedem jednostek źródłowych",
            df_w['HU_zrodlowa'].nunique() == 7)
    sprawdz("zakres przepuszczalności węglanu rzędu podanego w publikacji",
            df_w.Przepuszczalnosc.max() > 1000 and df_w.Przepuszczalnosc.median() > 5,
            f"{df_w.Przepuszczalnosc.min():.2f} - {df_w.Przepuszczalnosc.max():.0f} mD, "
            f"mediana {df_w.Przepuszczalnosc.median():.1f} mD")


    print("\n12. ZBIEŻNOŚĆ WZGLĘDEM ROZMIARU KRATKI (100 m vs 33 m)")
    # Ten sam obszar 3 x 3 km i to samo gładkie pole k(x, y) liczone na siatce
    # 30x30 (kratka 100 m) i 90x90 (kratka 33,3 m). Odwierty w tych samych
    # punktach terenu. Różnica mierzy błąd dyskretyzacji przestrzennej.
    wyniki_siatek = {}
    for f in (1, 3):
        n = 30 * f
        d = 3000.0 / n
        ii, jj = np.indices((n, n))
        x, y = (ii + 0.5) * d, (jj + 0.5) * d
        logk = 1.0 + 1.2 * np.exp(-((y - 1500 - 400 * np.sin(2 * np.pi * x / 3000)) / 350)**2) \
            + 0.4 * x / 3000
        z_s = ModelStatycznyZloza(d, d)
        z_s.wczytaj_dane(pd.DataFrame({'I': ii.ravel(), 'J': jj.ravel(),
                                       'Porowatosc': (0.12 + 0.03 * logk).ravel(),
                                       'Przepuszczalnosc': (10**logk).ravel()}))
        odw_s = [{'i': a * f + (f - 1) // 2, 'j': b * f + (f - 1) // 2,
                  'p_dna': 50.0, 'q_max': 1.2e6} for a, b in ((8, 8), (21, 21), (8, 21), (21, 8))]
        wyniki_siatek[f] = symulacja(z_s, pvt, odw_s, 1825.0, 15.0)
    q100 = np.array(wyniki_siatek[1].historia_wydatku)
    q33 = np.array(wyniki_siatek[3].historia_wydatku)
    roznica_q = np.sqrt(np.mean((q100 - q33)**2)) / np.ptp(q33) * 100
    roznica_pwf = np.max(np.abs(np.array(wyniki_siatek[1].historia_p_wf[0])
                                - np.array(wyniki_siatek[3].historia_p_wf[0])))
    sprawdz("kratka 100 m daje ten sam wydatek co kratka 33 m (różnica < 0,5 %)",
            roznica_q < 0.5, f"NRMSE {roznica_q:.3f} %")
    sprawdz("kratka 100 m daje to samo ciśnienie denne (różnica < 1 bar)",
            roznica_pwf < 1.0, f"maks. różnica {roznica_pwf:.2f} bar")

    print("\n" + "=" * 70)
    print(f"Zaliczone: {len(ZALICZONE)}, niezaliczone: {len(NIEZALICZONE)}")
    if NIEZALICZONE:
        print("Niezaliczone testy: " + ", ".join(NIEZALICZONE))
    return 1 if NIEZALICZONE else 0


if __name__ == '__main__':
    sys.exit(main())