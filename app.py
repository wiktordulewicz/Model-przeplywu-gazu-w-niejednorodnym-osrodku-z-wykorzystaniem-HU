"""
app.py - interfejs użytkownika symulatora (Streamlit).

Zasada podziału odpowiedzialności: ten plik wyłącznie zbiera parametry,
wywołuje warstwę obliczeniową i prezentuje wyniki. Cała fizyka i statystyka
znajduje się w modułach pvt.py, statyka.py i symulator_fvm.py, które nie
zależą od Streamlita i dają się uruchomić także ze zwykłego skryptu.

Uruchomienie:  streamlit run app.py
"""
import base64
from pathlib import Path
import io
import json
import uuid
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from statyka import (ModelStatycznyZloza, formatuj_osie_pl, sym, liczba_pl,
                     BLOKADA_RYSOWANIA)
from pvt import TermodynamikaGazu
from symulator_fvm import SymulatorFVM


def rysunek_png(rysuj, dpi=150, przycinaj=True):
    """
    Tworzy rysunek funkcją rysuj() i zwraca go jako PNG. Całość odbywa się
    pod wspólną blokadą: matplotlib nie jest bezpieczny wątkowo, a Streamlit
    przy szybkich zmianach widżetów może prowadzić dwa przebiegi skryptu
    naraz - bez blokady kończy się to błędem ParseException lub zawieszeniem.
    """
    with BLOKADA_RYSOWANIA:
        fig = rysuj()
        bufor = io.BytesIO()
        # przycinaj=False zachowuje pełny rozmiar rysunku - dwa wykresy obok
        # siebie mają wtedy identyczną wysokość
        fig.savefig(bufor, format='png', dpi=dpi, bbox_inches='tight' if przycinaj else None)
        plt.close(fig)
    return bufor.getvalue()


def osadz_png(png, opis="Wykres", max_szerokosc=None):
    """
    Wstawia obraz bezpośrednio w stronę (data URI). Nie korzysta z magazynu
    plików Streamlita, z którego obraz w zwiniętym panelu mógł zniknąć przed
    wyświetleniem (widać było wtedy tylko napis zastępczy '0').

    Obraz jest blokiem z dolnym odstępem: kontener markdown w Streamlicie ma
    ujemny margines dolny, przez co bez odstępu kolejny element nachodził na
    krawędź rysunku. max_szerokosc [px] pozwala wyśrodkować mniejszy rysunek.
    """
    dane = base64.b64encode(png).decode('ascii')
    ograniczenie = f"max-width:{max_szerokosc}px;" if max_szerokosc else ""
    st.markdown(f'<div style="padding-bottom:1.5rem">'
                f'<img src="data:image/png;base64,{dane}" alt="{opis}" '
                f'style="width:100%;{ograniczenie}display:block;margin:0 auto;"></div>',
                unsafe_allow_html=True)


def pokaz_rysunek(klucz, rysuj, opis="Wykres", dpi=150, wersja=None, max_szerokosc=None,
                  przycinaj=True):
    """
    Rysuje wykres tylko raz i zapamiętuje go w sesji pod podanym kluczem.
    Każda zmiana widżetu uruchamia skrypt od nowa; dzięki pamięci kolejne
    przebiegi wyświetlają gotowy obraz zamiast rysować wszystko ponownie, co
    wcześniej spowalniało aplikację przy każdej zmianie parametrów.
    Parametr wersja (np. numer kroku czasowego) wymusza ponowne narysowanie,
    gdy zmieni się zawartość, a pod kluczem przechowywany jest tylko jeden
    obraz - pamięć nie rośnie przy przewijaniu kroków czasowych.
    """
    pamiec = st.session_state.setdefault('obrazy', {})
    if klucz not in pamiec or pamiec[klucz][0] != wersja:
        pamiec[klucz] = (wersja, rysunek_png(rysuj, dpi, przycinaj))
    osadz_png(pamiec[klucz][1], opis, max_szerokosc)


# ------------------------------------------------------------------ animacja
def klatki_animacji(s, indeksy, czasy, p_min, p_max, dpi=90):
    """Klatki PNG mapy ciśnienia dla wybranych zapisów historii.

    Figura budowana jest raz, a w kolejnych klatkach podmieniane są tylko dane
    obrazu i tytuł, dlatego nawet kilkadziesiąt klatek powstaje w kilka sekund.
    Skala barw jest wspólna dla całej animacji (p_min, p_max), żeby kolory
    w różnych klatkach oznaczały to samo ciśnienie.
    """
    format_dnia = '.0f' if all(abs(t - round(t)) < 1e-9 for t in czasy) else '.1f'
    klatki = []
    with BLOKADA_RYSOWANIA:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
        obraz = ax.imshow(s.historia_cisnien_3d[indeksy[0]].T, cmap='plasma', origin='lower',
                          extent=[0, s.nx * s.dx, 0, s.ny * s.dy], vmin=p_min, vmax=p_max)
        fig.colorbar(obraz, ax=ax, label=r'Ciśnienie złożowe, $p$ [bar]')
        for otw in s.odwierty:
            ax.plot((otw['i'] + 0.5) * s.dx, (otw['j'] + 0.5) * s.dy, 'wo', markersize=7,
                    markeredgecolor='black')
            ax.annotate(otw['nazwa'], ((otw['i'] + 0.5) * s.dx, (otw['j'] + 0.5) * s.dy),
                        textcoords="offset points", xytext=(8, 5), color='white', fontsize=9)
        ax.set_xlabel(r'Współrzędna $x$ [m]')
        ax.set_ylabel(r'Współrzędna $y$ [m]')
        tytul = ax.set_title('')
        formatuj_osie_pl(fig)
        fig.tight_layout()
        for indeks, czas in zip(indeksy, czasy):
            obraz.set_data(s.historia_cisnien_3d[indeks].T)
            tytul.set_text(f"Rozkład ciśnienia złożowego, dzień {liczba_pl(czas, format_dnia)}")
            bufor = io.BytesIO()
            fig.savefig(bufor, format='png', dpi=dpi)
            klatki.append(bufor.getvalue())
        plt.close(fig)
    return klatki


def klatki_animacji_3d(s, indeksy, czasy, p_min, p_max, elewacja=30, azymut=-60, dpi=90):
    """Klatki PNG leja depresyjnego 3D.

    Zakres osi z i skala barw są stałe, więc lej widocznie pogłębia się
    i rozszerza z upływem czasu, zamiast być za każdym razem przeskalowany.
    Pionowe linie wskazują położenie odwiertów.
    """
    format_dnia = '.0f' if all(abs(t - round(t)) < 1e-9 for t in czasy) else '.1f'
    x_os = (np.arange(s.nx) + 0.5) * s.dx
    y_os = (np.arange(s.ny) + 0.5) * s.dy
    X, Y = np.meshgrid(x_os, y_os)
    zapas = 0.02 * max(p_max - p_min, 1e-6)
    z_dol, z_gora = p_min - zapas, p_max + zapas
    krok = max(1, int(np.ceil(max(s.nx, s.ny) / 60)))      # rzadsza siatka dla dużych złóż
    klatki = []
    with BLOKADA_RYSOWANIA:
        fig = plt.figure(figsize=(7, 5.5), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        mapowanie = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(p_min, p_max))
        fig.colorbar(mapowanie, ax=ax, shrink=0.6, pad=0.1, label=r'Ciśnienie złożowe, $p$ [bar]')
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        ax.set_zlabel(r'$p$ [bar]')
        ax.set_zlim(z_dol, z_gora)
        ax.view_init(elev=elewacja, azim=azymut)
        for otw in s.odwierty:
            xo, yo = (otw['i'] + 0.5) * s.dx, (otw['j'] + 0.5) * s.dy
            ax.plot([xo, xo], [yo, yo], [z_dol, z_gora], color='black', linewidth=1.5)
            ax.text(xo, yo, z_gora, otw['nazwa'], fontsize=8)
        tytul = ax.set_title('')
        formatuj_osie_pl(fig)
        powierzchnia = None
        for indeks, czas in zip(indeksy, czasy):
            if powierzchnia is not None:
                powierzchnia.remove()
            powierzchnia = ax.plot_surface(X, Y, s.historia_cisnien_3d[indeks].T, cmap='plasma',
                                           vmin=p_min, vmax=p_max, rstride=krok, cstride=krok,
                                           linewidth=0, antialiased=False)
            tytul.set_text(f"Lej depresyjny, dzień {liczba_pl(czas, format_dnia)}")
            bufor = io.BytesIO()
            fig.savefig(bufor, format='png', dpi=dpi)
            klatki.append(bufor.getvalue())
        plt.close(fig)
    return klatki


def odtwarzacz_html(klatki, opisy):
    """Prosty odtwarzacz klatek w HTML/JS: start, pauza, krokowanie, suwak, tempo, pętla.

    Całość działa w przeglądarce, więc odtwarzanie nie uruchamia ponownie
    skryptu Streamlita i nie obciąża serwera.
    """
    obrazy = json.dumps(["data:image/png;base64," + base64.b64encode(k).decode('ascii')
                         for k in klatki])
    podpisy = json.dumps(opisy, ensure_ascii=False)
    ostatnia = len(klatki) - 1
    return f"""
<div style="font-family: sans-serif; background: white; padding: 8px; border-radius: 6px;
            max-width: 640px; margin: 0 auto; color: #222;">
  <img id="obraz" style="width: 100%; display: block;">
  <input id="suwak" type="range" min="0" max="{ostatnia}" value="0" style="width: 100%;">
  <div style="display: flex; flex-wrap: wrap; gap: 6px; align-items: center; font-size: 14px;">
    <button id="pierwsza" title="Pierwsza klatka">&#9198;</button>
    <button id="wstecz" title="Poprzednia klatka">&#9664;</button>
    <button id="start">&#9654; Odtwórz</button>
    <button id="pauza">&#10073;&#10073; Pauza</button>
    <button id="dalej" title="Następna klatka">&#9654;</button>
    <button id="ostatnia" title="Ostatnia klatka">&#9197;</button>
    <label>Tempo
      <select id="szybkosc">
        <option value="2">wolno</option>
        <option value="6" selected>normalnie</option>
        <option value="12">szybko</option>
        <option value="25">bardzo szybko</option>
      </select>
    </label>
    <label><input id="petla" type="checkbox" checked> pętla</label>
    <span id="opis" style="margin-left: auto;"></span>
  </div>
</div>
<script>
  const obrazy = {obrazy};
  const podpisy = {podpisy};
  let biezaca = 0, licznik = null;
  const el = (id) => document.getElementById(id);
  function pokaz(n) {{
    biezaca = Math.max(0, Math.min(obrazy.length - 1, n));
    el('obraz').src = obrazy[biezaca];
    el('suwak').value = biezaca;
    el('opis').textContent = podpisy[biezaca] + ' (klatka ' + (biezaca + 1) + '/' + obrazy.length + ')';
  }}
  function zatrzymaj() {{ if (licznik) {{ clearInterval(licznik); licznik = null; }} }}
  function odtwarzaj() {{
    zatrzymaj();
    if (biezaca >= obrazy.length - 1) pokaz(0);
    licznik = setInterval(() => {{
      if (biezaca >= obrazy.length - 1) {{
        if (el('petla').checked) pokaz(0); else zatrzymaj();
      }} else pokaz(biezaca + 1);
    }}, 1000 / parseFloat(el('szybkosc').value));
  }}
  el('start').onclick = odtwarzaj;
  el('pauza').onclick = zatrzymaj;
  el('pierwsza').onclick = () => {{ zatrzymaj(); pokaz(0); }};
  el('ostatnia').onclick = () => {{ zatrzymaj(); pokaz(obrazy.length - 1); }};
  el('wstecz').onclick = () => {{ zatrzymaj(); pokaz(biezaca - 1); }};
  el('dalej').onclick = () => {{ zatrzymaj(); pokaz(biezaca + 1); }};
  el('suwak').oninput = (e) => {{ zatrzymaj(); pokaz(parseInt(e.target.value)); }};
  el('szybkosc').onchange = () => {{ if (licznik) odtwarzaj(); }};
  pokaz(0);
</script>
"""


def gif_z_klatek(klatki, fps=6):
    """Składa klatki PNG w zapętlony plik GIF do pobrania."""
    from PIL import Image
    obrazy = [Image.open(io.BytesIO(k)).convert('RGB').convert('P', palette=Image.ADAPTIVE)
              for k in klatki]
    bufor = io.BytesIO()
    obrazy[0].save(bufor, format='GIF', save_all=True, append_images=obrazy[1:],
                   duration=int(1000 / fps), loop=0)
    return bufor.getvalue()


st.set_page_config(page_title="Symulator przepływu gazu (FVM + HU)", layout="wide")

# Wykładnik cementacji w równaniu Kozeny-Carmana. Wartość 1 odpowiada
# klasycznemu wskaźnikowi FZI wg Amaefule i in. (1993) i jest przyjęta na stałe.
# Moduł statyczny obsługuje także wariant zmodyfikowany (Nooruddin i in., 2011)
# dla skał węglanowych - wystarczy przekazać inne m do metod klasy
# ModelStatycznyZloza; patrz diagnostyka_wykladnika_m i zalecenie_wykladnika_m.
WYKLADNIK_M = 1.0


# ---------------------------------------------------------------- konfiguracja
@dataclass(frozen=True)
class KonfiguracjaSymulacji:
    """Pełny zestaw parametrów jednego przebiegu - ułatwia zapis i porównania."""
    temp_zloza_C: float
    p_poczatkowe_bar: float
    gamma_g: float
    s_wc: float
    h_m: float
    anizotropia: float
    czas_dni: float
    dt_dni: float
    uzyj_hu: bool
    liczba_hu: int
    wykladnik_m: float

    def opis(self):
        return (f"{'HU k=' + str(self.liczba_hu) if self.uzyj_hu else 'dane surowe'}, "
                f"h={self.h_m:.0f} m, P_i={self.p_poczatkowe_bar:.0f} bar")


# ------------------------------------------------------- warstwa obliczeniowa
@st.cache_data(show_spinner=False)
def wczytaj_csv(zawartosc, nazwa):
    return pd.read_csv(io.BytesIO(zawartosc))


def zbuduj_zloze(df, nazwa_pliku, dx, dy):
    """
    Celowo bez pamięci podręcznej. Obiekt złoża jest instancją klasy
    z modułu statyka, więc gdyby trafił do cache, po każdej zmianie tego
    modułu w pamięci zostawałby obiekt starej klasy i wywołania nowych metod
    kończyłyby się błędem AttributeError aż do restartu serwera. Wczytanie
    siatki 30x30 trwa milisekundy, więc buforowanie i tak nic by nie dało.
    """
    zloze = ModelStatycznyZloza(dx=dx, dy=dy)
    zloze.wczytaj_dane(df)
    return zloze


@st.cache_data(show_spinner=False)
def kryteria_liczby_jednostek(_zloze, nazwa_pliku, m, max_k=10):
    return _zloze.diagnostyka_liczby_jednostek(max_k=max_k, m=m)


@st.cache_data(show_spinner=False)
def walidacja_hu(_zloze, nazwa_pliku, liczba_hu, m, powtorzenia=20, wersja=2):
    # 'wersja' należy do klucza pamięci podręcznej: po zmianie zawartości wyniku
    # walidacji zwiększenie numeru unieważnia stare, niepełne wyniki.
    return _zloze.walidacja_krzyzowa(liczba_hu=liczba_hu, m=m, powtorzenia=powtorzenia)


def przygotuj_zloze(zloze, cfg):
    """Ustawia pole przepuszczalności zgodnie z konfiguracją (z korektą HU albo bez)."""
    if cfg.uzyj_hu:
        zloze.wyznacz_jednostki_hu(liczba_hu=cfg.liczba_hu, m=cfg.wykladnik_m)
        zloze.uwzglednij_wplyw_hu_na_przepuszczalnosc()
    else:
        zloze.przywroc_surowa_przepuszczalnosc()
    return zloze


def nrmse(obserwowane, modelowane):
    obserwowane, modelowane = np.asarray(obserwowane), np.asarray(modelowane)
    zakres = np.ptp(obserwowane)
    return float(np.sqrt(np.mean((modelowane - obserwowane)**2)) / zakres) if zakres > 0 else np.nan


# --------------------------------------------------------------- wczytanie danych
st.title("Model przepływu gazu w niejednorodnym złożu z wykorzystaniem HU")
st.caption("Wiktor Dulewicz 2026")

st.subheader("1. Model statyczny złoża")
plik = st.file_uploader("Plik CSV z kolumnami Porowatość i Przepuszczalność "
                        "(opcjonalnie I, J lub x, y)", type=['csv'])


class PlikPrzykladowy:
    """Przykładowe złoże z repozytorium, udostępniane tak jak plik wgrany przez użytkownika."""
    name = "zloze_testowe.csv"

    def __init__(self, sciezka):
        self._dane = sciezka.read_bytes()

    def getvalue(self):
        return self._dane


SCIEZKA_PRZYKLADU = Path(__file__).parent / "zloze_testowe.csv"
if plik is None and SCIEZKA_PRZYKLADU.exists():
    if st.checkbox("Użyj przykładowego złoża testowego (30 × 30 bloków po 100 m)", value=True,
                   help="Dane syntetyczne o statystykach złoża opisanego przez Ellabad i in. (2001). "
                        "Wgranie własnego pliku zastępuje dane przykładowe."):
        plik = PlikPrzykladowy(SCIEZKA_PRZYKLADU)
if plik is None:
    st.info("Wczytaj plik ze złożem, aby odblokować parametry symulacji. "
            )
    st.stop()

try:
    df_dane = wczytaj_csv(plik.getvalue(), plik.name)
    zloze = zbuduj_zloze(df_dane, plik.name, 100.0, 100.0)
except Exception as blad:
    st.error(f"Nie udało się wczytać danych: {type(blad).__name__}: {blad}")
    st.stop()

for ostrzezenie in zloze.ostrzezenia:
    st.warning(ostrzezenie)

k1, k2, k3 = st.columns(3)
k1.metric("Siatka", f"{zloze.nx} x {zloze.ny}")
k2.metric("Rozmiar złoża", f"{zloze.nx * zloze.dx / 1000:.1f} x {zloze.ny * zloze.dy / 1000:.1f} km")
k3.metric("Przepuszczalność", f"{zloze.k_mD_surowe.min():.1f} - {zloze.k_mD_surowe.max():.0f} mD",
          help="Zakres wartości wejściowych w pliku.")

# ----------------------------------------------------------------- pasek boczny
sb = st.sidebar
sb.header("Parametry gazu i złoża")
temp_zloza = sb.slider("Temperatura złoża [°C]", 20.0, 120.0, 60.0, 1.0)
p_pocz = sb.slider("Ciśnienie początkowe [bar]", 50.0, 350.0, 170.0, 1.0)
gamma_g = sb.slider("Gęstość względna gazu [-]", 0.55, 0.80, 0.58, 0.01)
s_wc = sb.slider("Nasycenie wodą związaną S_wc [-]", 0.0, 0.5, 0.25, 0.01)
h_zloza = sb.slider("Miąższość złoża h [m]", 1.0, 100.0, 15.0, 1.0)
anizo = sb.slider("Anizotropia ky/kx [-]", 0.1, 1.0, 1.0, 0.1)

sb.header("Parametry symulacji")
czas_symulacji = sb.number_input("Czas eksploatacji [dni]", 1.0, 7300.0, 365.0, 1.0)
dt_dni = sb.slider("Krok czasowy dt [dni]", 0.1, 30.0, 1.0, 0.1,
                   help="Schemat niejawny jest bezwarunkowo stabilny; krok wpływa tylko na dokładność.")
liczba_krokow = max(1, int(round(czas_symulacji / dt_dni)))

sb.markdown("---")
sb.header("Jednostki hydrauliczne")
uzyj_hu = sb.checkbox("Zastosuj korektę przepuszczalności (model HU)", value=True)
liczba_hu = sb.slider("Liczba jednostek hydraulicznych", 1, 10, 4, 1)

with sb.expander("Dobór liczby jednostek"):
    tabela_k = kryteria_liczby_jednostek(zloze, plik.name, WYKLADNIK_M)
    zalecenie_k = zloze.zalecenie_liczby_jednostek(tabela_k)
    (st.success if zalecenie_k['pewne'] else st.warning)(
        f"**{zalecenie_k['k_sugerowane']} jednostek**  \n{zalecenie_k['zalecenie']}")
    pokaz_rysunek(('lokiec', plik.name, WYKLADNIK_M),
                  lambda: zloze.generuj_wykres_lokcia(tabela_k), "Wykres metody łokcia")
    st.caption(
        "Metoda łokcia: dystorsja, czyli średni kwadrat odległości punktów od środków "
        "klastrów, względem liczby jednostek. Punkt załamania wykrywa algorytm Kneedle "
        "(Satopää i in., 2011), liczony na logarytmie dystorsji z pominięciem k = 1. "
        "Gdy krzywa nie ma wyraźnego załamania, dane nie są wyraźnie pogrupowane "
        "i metoda nie rozstrzyga podziału.")

sb.markdown("---")
sb.header("Odwierty")
tryb = sb.radio("Sposób rozmieszczenia", ["Ręczny", "Automatyczny (maksimum k·h)"])
liczba_odwiertow = sb.number_input("Liczba odwiertów", 1, 10, 1, 1)

cfg = KonfiguracjaSymulacji(
    temp_zloza_C=temp_zloza, p_poczatkowe_bar=p_pocz, gamma_g=gamma_g, s_wc=s_wc,
    h_m=h_zloza, anizotropia=anizo, czas_dni=czas_symulacji, dt_dni=dt_dni,
    uzyj_hu=uzyj_hu, liczba_hu=liczba_hu, wykladnik_m=WYKLADNIK_M)

# Pole k przygotowywane jest tu raz, w jednym miejscu, zarówno dla doboru
# lokalizacji odwiertów, jak i dla samej symulacji.
przygotuj_zloze(zloze, cfg)

lista_odwiertow = []
sb.caption("Trybem głównym jest zadany wydatek, a minimalne ciśnienie denne stanowi "
           "ograniczenie: dopóki złoże dostarcza zadany wydatek, odwiert pracuje na półce, "
           "a gdy przestaje, sterowanie przechodzi samoczynnie na ciśnienie denne.")


def parametry_odwiertu(przedrostek, k_komorki=None):
    """Zbiera parametry jednego odwiertu; zwraca słownik bez współrzędnych."""
    if k_komorki is not None:
        st.caption(f"Pojemność przepływowa k·h = {k_komorki:.0f} mD·m")
    ogranicz = st.checkbox("Ogranicz wydatek (półka produkcyjna)", value=True,
                           key=f"{przedrostek}_ogr")
    q_max = None
    if ogranicz:
        q_max = st.number_input("Zadany wydatek [tys. m³/d]", 1.0, 500.0, 50.0, 1.0,
                                key=f"{przedrostek}_q") * 1000.0
    q_min = st.number_input("Próg ekonomiczny [tys. m³/d]", 0.0, 500.0, 20.0, 1.0,
                            key=f"{przedrostek}_qmin",
                            help="Gdy wydatek spadnie poniżej tej wartości, odwiert zostaje "
                                 "zamknięty na stałe. Zero wyłącza warunek.") * 1000.0
    p_dna = st.slider("Minimalne ciśnienie denne P_wf [bar]", 10.0, 300.0, 50.0, 5.0,
                      key=f"{przedrostek}_p")
    skin = st.slider("Skin mechaniczny [-]", -5.0, 15.0, 0.0, 0.5, key=f"{przedrostek}_s")
    d_turb = st.slider("Współczynnik Forchheimera D [d/tys. m³]", 0.0, 0.01, 0.0, 0.0005,
                       format="%.4f", key=f"{przedrostek}_d")
    return {'p_dna': p_dna, 'skin': skin, 'D': d_turb, 'q_max': q_max, 'q_min': q_min}


if tryb == "Ręczny":
    for nr in range(int(liczba_odwiertow)):
        with sb.expander(f"Odwiert O-{nr + 1}", expanded=(nr == 0)):
            w_i = st.slider("Współrzędna I (oś X)", 0, zloze.nx - 1, zloze.nx // 2, key=f"i_{nr}")
            w_j = st.slider("Współrzędna J (oś Y)", 0, zloze.ny - 1, zloze.ny // 2, key=f"j_{nr}")
            st.markdown("---")
            parametry = parametry_odwiertu(f"r{nr}")
        lista_odwiertow.append({'nazwa': f"O-{nr + 1}", 'i': w_i, 'j': w_j, **parametry})
else:
    promien = sb.slider("Minimalny odstęp między odwiertami [komórki]", 1, 10, 4, 1)
    znalezione = zloze.znajdz_sweet_spots(int(liczba_odwiertow), promien, h=cfg.h_m)
    for nr, opt in enumerate(znalezione):
        with sb.expander(f"Odwiert O-{nr + 1} (I={opt['i']}, J={opt['j']})", expanded=(nr == 0)):
            parametry = parametry_odwiertu(f"a{nr}", k_komorki=opt['kh'])
        lista_odwiertow.append({'nazwa': f"O-{nr + 1}", 'i': opt['i'], 'j': opt['j'], **parametry})
    if len(znalezione) < liczba_odwiertow:
        sb.warning(f"Przy odstępie {promien * 100:.0f} m zmieściło się tylko "
                   f"{len(znalezione)} odwiertów.")

# ------------------------------------------------------------------ symulacja
st.subheader("2. Symulacja")
if 'scenariusze' not in st.session_state:
    st.session_state.scenariusze = []

if st.button("Uruchom symulację", type="primary"):
    try:
        pvt = TermodynamikaGazu(gamma_g=cfg.gamma_g, temp_zloza_C=cfg.temp_zloza_C)
        symulator = SymulatorFVM(zloze, pvt, cfg.p_poczatkowe_bar, lista_odwiertow,
                                 lambda_anizo=cfg.anizotropia, h=cfg.h_m, s_wc=cfg.s_wc)
        pasek = st.progress(0.0, text="Rozwiązywanie układu równań...")

        def postep(krok, ile):
            # odświeżanie paska co 1 % - każda aktualizacja to komunikat do
            # przeglądarki, a przy tysiącach kroków spowalniało to obliczenia
            if krok == ile or krok % max(1, ile // 100) == 0:
                pasek.progress(krok / ile, text=f"Krok {krok} z {ile}")

        symulator.uruchom(cfg.czas_dni, cfg.dt_dni, callback=postep)
        pasek.empty()
        st.session_state.wynik = {'symulator': symulator, 'cfg': cfg,
                                  'hu_grid': zloze.hu_grid.copy() if (cfg.uzyj_hu and zloze.hu_grid is not None) else None,
                                  'tabela_hu': zloze.generuj_tabele_rownan() if cfg.uzyj_hu else None,
                                  'figura_hu': (rysunek_png(zloze.generuj_figure_diagnostyczna, dpi=110)
                                                if cfg.uzyj_hu else None),
                                  'id': uuid.uuid4().hex,
                                  'k_mD': zloze.k_mD.copy(), 'phi': zloze.phi.copy()}
        st.session_state.obrazy = {}           # rysunki poprzedniej symulacji
        st.success(f"Obliczono {len(symulator.historia_czasu)} kroków czasowych "
                   f"({cfg.czas_dni:.0f} dni eksploatacji).")
    except Exception as blad:
        st.error(f"Symulacja przerwana: {type(blad).__name__}: {blad}")

if 'wynik' not in st.session_state:
    st.stop()

wynik = st.session_state.wynik
s = wynik['symulator']
cfg_w = wynik['cfg']
wid = wynik.get('id', 'brak')
for ostrzezenie in s.ostrzezenia[:3]:
    st.warning(ostrzezenie)

# --------------------------------------------------------------------- wyniki
def format_objetosci(wartosc_m3):
    """Skraca zapis, żeby wartość mieściła się w kafelku."""
    if wartosc_m3 >= 1e9:
        return f"{wartosc_m3 / 1e9:.2f} mld m³"
    return f"{wartosc_m3 / 1e6:.0f} mln m³"


bilans = s.bilans_materialowy()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Zasoby pierwotne (IGIP)", format_objetosci(bilans['igip']))
m2.metric("Produkcja skumulowana", format_objetosci(bilans['gp']))
m3.metric("Stopień sczerpania", f"{s.wspolczynnik_sczerpania():.2f} %")
m4.metric("Błąd bilansu masy", f"{abs(bilans['blad_wzgledny']):.1e}",
          help="Względna różnica między produkcją a ubytkiem zasobów wynikającym z pola "
               "ciśnień. Wartości rzędu 1e-6 i mniejsze świadczą o poprawnym rozwiązaniu.")

# Nawigacja po wynikach. Zwykłe zakładki (st.tabs) po każdym kliknięciu
# przycisku, np. "Wykonaj walidację krzyżową", wracały do pierwszej karty.
# Wybór zapamiętany w stanie sesji zostaje na miejscu, a przy okazji rysowana
# jest tylko otwarta część wyników, co przyspiesza każde odświeżenie strony.
ZAKLADKI = ["Jednostki hydrauliczne", "Pole ciśnień", "Produkcja",
            "Bilans materiałowy", "Walidacja HU", "Eksport"]
zakladka = st.radio("Widok wyników", ZAKLADKI, horizontal=True, key="zakladka",
                    label_visibility="collapsed")
st.markdown("---")

if zakladka == ZAKLADKI[0]:
    if wynik['figura_hu'] is not None:
        osadz_png(wynik['figura_hu'], "Zależność k(φ) w jednostkach hydraulicznych",
                  max_szerokosc=1000)
        st.markdown("**Korelacje *k*(*φ*) w obrębie jednostek hydraulicznych**")
        st.write(wynik['tabela_hu'].to_html(escape=False, index=False, justify='left'),
                 unsafe_allow_html=True)
        pokaz_rysunek(('mapa_hu', wid), lambda: zloze.generuj_mape(
            wynik['hu_grid'], "Rozmieszczenie jednostek hydraulicznych",
            "Numer jednostki hydraulicznej (HU) [-]", cmap='tab10', dyskretne=True),
            "Mapa jednostek hydraulicznych", max_szerokosc=760)
    else:
        st.info("Symulację wykonano na danych surowych, bez podziału na jednostki hydrauliczne.")
    kol1, kol2 = st.columns(2)
    with kol1:
        pokaz_rysunek(('mapa_phi', wid), lambda: zloze.generuj_mape(
            wynik['phi'], "Porowatość efektywna", r"Porowatość, $\phi$ [-]"), "Mapa porowatości",
            przycinaj=False)
    with kol2:
        pokaz_rysunek(('mapa_k', wid), lambda: zloze.generuj_mape(
            wynik['k_mD'], "Przepuszczalność użyta w symulacji", r"Przepuszczalność, $k$ [mD]",
            skala_log=True), "Mapa przepuszczalności", przycinaj=False)

if zakladka == ZAKLADKI[1]:
    p_min_historii = float(min(x.min() for x in s.historia_cisnien_3d))
    p_max_historii = float(s.historia_cisnien_3d[0].max())

    # Czas każdego zapisu pola ciśnień; zapis 0 to stan początkowy (dzień 0).
    interwal = getattr(s, 'interwal_zapisu', 1)
    czasy_zapisu = [0.0] + [float(s.historia_czasu[n * interwal - 1])
                            for n in range(1, len(s.historia_cisnien_3d))]
    t_koniec = czasy_zapisu[-1]

    st.markdown("**Animacja zmian ciśnienia w czasie**")
    od_dnia, do_dnia = st.slider(
        "Zakres animacji [doba]", 0.0, float(t_koniec), (0.0, float(t_koniec)),
        step=float(cfg_w.dt_dni),
        help="Dzień 0 to stan początkowy, przed uruchomieniem odwiertów. "
             "Animacja obejmuje zapisy pola ciśnień z wybranego przedziału.")
    kol_a, kol_b = st.columns(2)
    liczba_klatek = int(kol_a.number_input(
        "Liczba klatek", 10, 200, 60, 10,
        help="Przy długiej symulacji klatki wybierane są równomiernie z zakresu. "
             "Więcej klatek to płynniejsza animacja, ale dłuższe przygotowanie."))
    rodzaj = kol_b.radio("Rodzaj animacji", ["Mapa ciśnienia (2D)", "Lej depresyjny (3D)"],
                         horizontal=True, key="rodzaj_animacji")
    animacja_3d = rodzaj == "Lej depresyjny (3D)"
    elewacja, azymut = 30, -60
    if animacja_3d:
        kol_c, kol_d = st.columns(2)
        elewacja = kol_c.slider("Wysokość kamery [°]", 5, 90, 30, 5,
                                help="90° to widok z góry, małe wartości - widok z boku.")
        azymut = kol_d.slider("Obrót kamery [°]", -180, 180, -60, 15)

    klucz_animacji = (wid, od_dnia, do_dnia, liczba_klatek, rodzaj, elewacja, azymut)
    if st.button("Utwórz animację"):
        w_zakresie = [n for n, t in enumerate(czasy_zapisu) if od_dnia - 1e-9 <= t <= do_dnia + 1e-9]
        if len(w_zakresie) < 2:
            st.warning("W wybranym zakresie są mniej niż dwa zapisy pola ciśnień - "
                       "rozszerz zakres animacji.")
        else:
            wybor = np.unique(np.round(np.linspace(
                0, len(w_zakresie) - 1, min(liczba_klatek, len(w_zakresie)))).astype(int))
            indeksy = [w_zakresie[n] for n in wybor]
            czasy = [czasy_zapisu[n] for n in indeksy]
            with st.spinner(f"Przygotowanie {len(indeksy)} klatek..."):
                if animacja_3d:
                    klatki = klatki_animacji_3d(s, indeksy, czasy, p_min_historii,
                                                p_max_historii, elewacja, azymut)
                else:
                    klatki = klatki_animacji(s, indeksy, czasy, p_min_historii, p_max_historii)
                format_dnia = '.0f' if all(abs(t - round(t)) < 1e-9 for t in czasy) else '.1f'
                opisy = [f"dzień {liczba_pl(t, format_dnia)}" for t in czasy]
                st.session_state.animacja = {
                    'klucz': klucz_animacji,
                    'html': odtwarzacz_html(klatki, opisy),
                    'gif': gif_z_klatek(klatki),
                    'plik': 'animacja_leja_3d.gif' if animacja_3d else 'animacja_cisnienia.gif'}

    animacja = st.session_state.get('animacja')
    if animacja is not None and animacja['klucz'][0] == wid:
        if animacja['klucz'] != klucz_animacji:
            st.caption("Ustawienia zmieniły się od utworzenia animacji - kliknij "
                       "„Utwórz animację”, aby ją odświeżyć.")
        components.html(animacja['html'], height=700 if animacja['plik'].endswith('3d.gif') else 650)
        st.download_button("Pobierz animację (GIF)", animacja['gif'], animacja['plik'], "image/gif")

    st.markdown("---")
    st.markdown("**Wybrany krok czasowy**")
    klatka = st.slider("Krok czasowy", 0, len(s.historia_cisnien_3d) - 1,
                       len(s.historia_cisnien_3d) - 1,
                       format="Krok %d")
    p_klatka = s.historia_cisnien_3d[klatka]
    czas_klatki = czasy_zapisu[klatka]
    st.caption(f"Czas: {liczba_pl(czas_klatki, '.1f')} dnia")

    def rysuj_mape_cisnien():
        # Transpozycja: pierwszy indeks tablicy to oś X, która ma być pozioma
        fig, ax = plt.subplots(figsize=(6, 5))
        obraz = ax.imshow(p_klatka.T, cmap='plasma', origin='lower',
                          extent=[0, s.nx * s.dx, 0, s.ny * s.dy],
                          vmin=p_min_historii, vmax=p_max_historii)
        fig.colorbar(obraz, ax=ax, label=r'Ciśnienie złożowe, $p$ [bar]')
        for otw in s.odwierty:
            ax.plot((otw['i'] + 0.5) * s.dx, (otw['j'] + 0.5) * s.dy, 'wo', markersize=7,
                    markeredgecolor='black')
            ax.annotate(otw['nazwa'], ((otw['i'] + 0.5) * s.dx, (otw['j'] + 0.5) * s.dy),
                        textcoords="offset points", xytext=(8, 5), color='white', fontsize=9)
        ax.set_xlabel(r'Współrzędna $x$ [m]')
        ax.set_ylabel(r'Współrzędna $y$ [m]')
        ax.set_title('Rozkład ciśnienia złożowego')
        return formatuj_osie_pl(fig)

    # mapa wyśrodkowana, lej depresyjny 3D pod nią na całą szerokość
    pokaz_rysunek(('cisnienie', wid), rysuj_mape_cisnien, "Mapa ciśnienia złożowego",
                  wersja=klatka, max_szerokosc=700)

    x_os = (np.arange(s.nx) + 0.5) * s.dx
    y_os = (np.arange(s.ny) + 0.5) * s.dy
    fig3d = go.Figure(data=[go.Surface(
        z=p_klatka.T, x=x_os, y=y_os, colorscale='Plasma',
        cmin=p_min_historii, cmax=p_max_historii)])
    fig3d.update_layout(
        separators=', ',            # przecinek dziesiętny, spacja jako separator tysięcy
        scene=dict(xaxis_title='<i>x</i> [m]', yaxis_title='<i>y</i> [m]',
                   zaxis_title='Ciśnienie, <i>p</i> [bar]',
                   aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.6)),
        height=650, margin=dict(l=0, r=0, b=0, t=30), title="Lej depresyjny")
    st.plotly_chart(fig3d, use_container_width=True)


def rysuj_wydatek():
    fig, ax = plt.subplots(figsize=(10, 4))
    for scenariusz in st.session_state.scenariusze:
        ax.plot(scenariusz['czas'], np.array(scenariusz['wydatek']) / 1e3, linestyle='--',
                label=scenariusz['opis'])
    ax.plot(s.historia_czasu, np.array(s.historia_wydatku) / 1e3, linewidth=2.5,
            label=f"Bieżąca: {cfg_w.opis()}")
    if len(s.odwierty) > 1:
        for nr, otw in enumerate(s.odwierty):
            ax.plot(s.historia_czasu, np.array(s.historia_odwiertow[nr]) / 1e3, linewidth=1,
                    alpha=0.7, label=otw['nazwa'])
    ax.set_xlabel(r"Czas, $t$ [d]")
    ax.set_ylabel(f"Wydatek gazu, {sym('q', 'g')} [tys. m³/d]")
    ax.set_ylim(bottom=0)        # od zera: stała półka nie udaje zmienności
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return formatuj_osie_pl(fig)


def rysuj_cisnienie_denne():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for nr, otw in enumerate(s.odwierty):
        ax.plot(s.historia_czasu, s.historia_p_wf[nr], linewidth=1.5, label=otw['nazwa'])
    ax.axhline(min(o['p_dna'] for o in s.odwierty), color='#d62728', ls='--', linewidth=1,
               label=f"Minimalne ciśnienie denne, {sym('p', 'wf,min')}")
    ax.set_xlabel(r"Czas, $t$ [d]")
    ax.set_ylabel(f"Ciśnienie denne, {sym('p', 'wf')} [bar]")
    p_gorna = max([float(s.historia_cisnien_3d[0].max())] + [max(h) for h in s.historia_p_wf if h])
    ax.set_ylim(0, p_gorna * 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return formatuj_osie_pl(fig)


gp = np.array(s.historia_gp) / 1e6
pz = np.array(s.historia_pz)
wspolczynniki = np.polyfit(gp, pz, 1)


def rysuj_pz():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gp, pz, 'o-', markersize=3, label="Wynik symulacji")
    gp_ekstrapolacja = np.linspace(0, -wspolczynniki[1] / wspolczynniki[0], 50)
    ax.plot(gp_ekstrapolacja, np.polyval(wspolczynniki, gp_ekstrapolacja), '--',
            linewidth=1, label="Ekstrapolacja liniowa")
    ax.set_xlabel(f"Produkcja skumulowana, {sym('G', 'p')} [mln m³]")
    ax.set_ylabel(r"Ciśnienie zredukowane, $p/Z$ [bar]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return formatuj_osie_pl(fig)


if zakladka == ZAKLADKI[2]:
    pokaz_rysunek(('wydatek', wid), rysuj_wydatek, "Wykres wydatku gazu",
                  wersja=st.session_state.get('wersja_scenariuszy', 0))

    caly_czas_na_polce = [nr for nr, tryby in enumerate(s.historia_trybu)
                          if tryby and all(t == 'wydatek' for t in tryby)]
    if caly_czas_na_polce:
        nazwy = ", ".join(s.odwierty[nr]['nazwa'] for nr in caly_czas_na_polce)
        kto = (f"Odwiert {nazwy} przez cały czas symulacji utrzymał"
               if len(caly_czas_na_polce) == 1
               else f"Odwierty {nazwy} przez cały czas symulacji utrzymały")
        st.info(
            f"{kto} zadany wydatek - "
            f"złoże dostarcza więcej, niż od niego żądano, więc wykres jest płaski. "
            f"Sczerpano dotąd {s.wspolczynnik_sczerpania():.1f} % zasobów. Aby zobaczyć "
            f"koniec półki i spadek wydobycia, zwiększ zadany wydatek albo wydłuż czas "
            f"eksploatacji.")

    st.markdown("**Ciśnienie denne i tryb sterowania**")
    pokaz_rysunek(('cisnienie_denne', wid), rysuj_cisnienie_denne, "Wykres ciśnienia dennego")
    st.caption("Dopóki ciśnienie denne jest wyższe od zadanego minimum, odwiert utrzymuje "
               "półkę produkcyjną. Zejście do linii przerywanej oznacza przejście na "
               "sterowanie ciśnieniem i początek spadku wydobycia.")

    st.dataframe(pd.DataFrame(s.diagnostyka_odwiertow()), use_container_width=True, hide_index=True)

    kol1, kol2 = st.columns(2)
    if kol1.button("Zapisz krzywą do porównania"):
        st.session_state.scenariusze.append({'czas': list(s.historia_czasu),
                                             'wydatek': list(s.historia_wydatku),
                                             'opis': cfg_w.opis()})
        st.session_state.wersja_scenariuszy = st.session_state.get('wersja_scenariuszy', 0) + 1
        st.rerun()
    if kol2.button("Wyczyść zapisane krzywe"):
        st.session_state.scenariusze = []
        st.session_state.wersja_scenariuszy = st.session_state.get('wersja_scenariuszy', 0) + 1
        st.rerun()

if zakladka == ZAKLADKI[3]:
    st.markdown("Wykres p/Z względem produkcji skumulowanej. Dla złoża zamkniętego "
                "o napędzie ekspansyjnym zależność ta jest prostoliniowa, a jej "
                "przecięcie z osią odciętych wskazuje zasoby pierwotne. Odchylenie od "
                "prostej świadczyłoby o błędzie w bilansie masy modelu.")
    pokaz_rysunek(('pz', wid), rysuj_pz, "Wykres p/Z")
    igip_z_wykresu = -wspolczynniki[1] / wspolczynniki[0] * 1e6
    kol1, kol2 = st.columns(2)
    kol1.metric("Zasoby z ekstrapolacji p/Z", f"{igip_z_wykresu / 1e6:.1f} mln m³")
    kol2.metric("Różnica względem IGIP objętościowego",
                f"{abs(igip_z_wykresu - bilans['igip']) / bilans['igip'] * 100:.2f} %")

if zakladka == ZAKLADKI[4]:
    st.markdown("""
Test sprawdza, czy podział na jednostki hydrauliczne poprawia przewidywanie
przepuszczalności w komórkach bez danych rdzeniowych. Korelacje dopasowywane są
wyłącznie na próbce treningowej, a ocena prowadzona na pozostałych komórkach.
Jednostka komórki testowej przewidywana jest z porowatości i położenia, nigdy
z jej własnej przepuszczalności - w przeciwnym razie wynik byłby obarczony
przeciekiem informacji, bo FZI jest funkcją k.
""")
    klucz_walidacji = (plik.name, cfg_w.liczba_hu, cfg_w.wykladnik_m)
    if st.button("Wykonaj walidację krzyżową"):
        with st.spinner("Liczenie..."):
            st.session_state.walidacja = {
                'klucz': klucz_walidacji,
                'wynik': walidacja_hu(zloze, plik.name, cfg_w.liczba_hu, cfg_w.wykladnik_m)}
    # Wynik zostaje w sesji, więc nie znika przy kolejnych odświeżeniach strony;
    # pokazywany jest tylko wtedy, gdy dotyczy bieżącego pliku i liczby jednostek.
    zapis_walidacji = st.session_state.get('walidacja')
    if zapis_walidacji is not None and zapis_walidacji['klucz'] == klucz_walidacji:
        w = zapis_walidacji['wynik']
        kol1, kol2, kol3 = st.columns(3)
        kol1.metric("RMSE (dla log k) - model globalny",
                    f"{liczba_pl(w['rmse_globalna'])} ± {liczba_pl(w['rmse_globalna_odch'])}",
                    help="Średnia z losowań ± odchylenie standardowe.")
        kol2.metric("RMSE (dla log k) - model HU",
                    f"{liczba_pl(w['rmse_hu'])} ± {liczba_pl(w['rmse_hu_odch'])}",
                    help="Średnia z losowań ± odchylenie standardowe.")
        kol3.metric("Poprawa dzięki HU", f"{liczba_pl(w['poprawa_proc'], '+.1f')} %",
                    help="Względne zmniejszenie błędu RMSE (dla log k) modelu HU względem "
                         "modelu globalnego: (RMSE glob. − RMSE HU) / RMSE glob. × 100 %.")
        st.caption(f"Trafność klasyfikacji jednostek: "
                   f"{liczba_pl(w['trafnosc_klasyfikacji'] * 100, '.1f')} %. "
                   f"Próbka treningowa: {w['n_treningowych']} komórek, "
                   f"{w['powtorzenia']} powtórzeń losowania.")
        with st.expander("Jak liczona jest poprawa?"):
            r_g, r_h = w['rmse_globalna'], w['rmse_hu']
            st.markdown(f"""
1. Komórki złoża dzielone są losowo na próbkę treningową (udające rdzenie,
   {w['n_treningowych']} komórek) i testową (pozostałe komórki).
2. Na próbce treningowej dopasowywane są dwa modele: **globalny** - jedna zależność
   *k*(*φ*) dla całego złoża - oraz **HU** - osobna zależność w każdej jednostce.
   Jednostkę komórki testowej przewiduje klasyfikator k-NN z porowatości i położenia.
3. Dla komórek testowych liczony jest błąd średniokwadratowy w skali logarytmicznej:
   RMSE = √( średnia z (log₁₀ *k*<sub>estymowane</sub> − log₁₀ *k*<sub>zmierzone</sub>)² ).
4. Losowanie powtarzane jest {w['powtorzenia']} razy, a wyniki uśredniane.
5. Poprawa = (RMSE<sub>glob.</sub> − RMSE<sub>HU</sub>) / RMSE<sub>glob.</sub> × 100 % =
   ({liczba_pl(r_g, '.3f')} − {liczba_pl(r_h, '.3f')}) / {liczba_pl(r_g, '.3f')} × 100 % =
   **{liczba_pl(w['poprawa_proc'], '+.1f')} %**.

**Interpretacja.** RMSE liczone jest na logarytmach, więc 10<sup>RMSE</sup> to typowy
mnożnik błędu estymacji przepuszczalności: model globalny myli się średnio
{liczba_pl(10 ** r_g, '.1f')}-krotnie, a model HU {liczba_pl(10 ** r_h, '.1f')}-krotnie.
Wartość dodatnia oznacza, że podział na jednostki hydrauliczne poprawia estymację,
ujemna - że ją pogarsza.
""", unsafe_allow_html=True)
        if 'przyklad' in w:
            klucz_rys = ('walidacja', plik.name, cfg_w.liczba_hu)
            kol_a, kol_b = st.columns(2)
            with kol_a:
                pokaz_rysunek(klucz_rys + ('a',), lambda: ModelStatycznyZloza.
                              generuj_wykres_rozrzutu_walidacji(w, 'globalny'),
                              "Walidacja - model globalny", przycinaj=False)
            with kol_b:
                pokaz_rysunek(klucz_rys + ('b',), lambda: ModelStatycznyZloza.
                              generuj_wykres_rozrzutu_walidacji(w, 'hu'),
                              "Walidacja - model HU", przycinaj=False)
            st.caption("(a), (b): przepuszczalność estymowana względem zmierzonej. Każdy "
                       "punkt to komórka testowa, dla której przepuszczalność oszacowano bez "
                       "znajomości jej wartości zmierzonej; linia ciągła to linia zgodności "
                       "1:1, a linie przerywane wyznaczają pas błędu jednego rzędu wielkości "
                       "(estymacja dziesięciokrotnie zawyżona lub zaniżona). Kolory w (b) "
                       "oznaczają przypisane jednostki hydrauliczne.")
            st.markdown("---")
            pokaz_rysunek(klucz_rys + ('c',), lambda: ModelStatycznyZloza.
                          generuj_wykres_bledu_walidacji(w),
                          "Walidacja - rozkład błędu", max_szerokosc=820)
            st.caption("(c): błąd średniokwadratowy w kolejnych losowaniach próbki "
                       "rdzeniowej. Każdy punkt to jedno losowanie, pudełko obejmuje środkowe "
                       "50 % wyników, czerwona kreska to mediana. Niższe wartości oznaczają "
                       "lepszą estymację, a rozdzielone pudełka - przewagę niezależną od "
                       "losowania.")
        if w['poprawa_proc'] <= 0:
            st.info("W tym złożu podział na jednostki nie poprawia predykcji. Jest to wynik "
                    "oczekiwany, gdy wszystkie komórki opisuje jedna zależność *k*(*φ*): dzielenie "
                    "jej na kilka korelacji dopasowuje się wtedy do szumu.")

    st.markdown("---")
    st.markdown("**Dopasowanie do historii produkcji**")
    plik_hist = st.file_uploader("CSV z kolumnami 'czas' i 'wydatek'", type=['csv'], key="hist")
    if plik_hist is not None:
        df_hist = pd.read_csv(plik_hist)
        if {'czas', 'wydatek'}.issubset(df_hist.columns):
            obserwowane = np.interp(s.historia_czasu, df_hist['czas'], df_hist['wydatek'])
            blad = nrmse(obserwowane, s.historia_wydatku)
            kol1, kol2 = st.columns(2)
            kol1.metric("NRMSE", f"{blad * 100:.2f} %",
                        help="Pierwiastek błędu średniokwadratowego odniesiony do rozpiętości "
                             "danych pomiarowych. Miara odporniejsza niż R² dla krzywych "
                             "o silnym trendzie.")
            kol2.metric("Średni błąd względny",
                        f"{np.mean(np.abs(np.array(s.historia_wydatku) - obserwowane) / np.maximum(obserwowane, 1e-9)) * 100:.2f} %")
            def rysuj_dopasowanie():
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(df_hist['czas'], np.asarray(df_hist['wydatek']) / 1e3, 'o',
                        markersize=4, label="Dane historyczne")
                ax.plot(s.historia_czasu, np.array(s.historia_wydatku) / 1e3, linewidth=2,
                        label="Model")
                ax.set_xlabel(r"Czas, $t$ [d]")
                ax.set_ylabel(f"Wydatek gazu, {sym('q', 'g')} [tys. m³/d]")
                ax.grid(True, alpha=0.3)
                ax.legend()
                return formatuj_osie_pl(fig)

            pokaz_rysunek(('historia', wid, plik_hist.name, plik_hist.size),
                          rysuj_dopasowanie, "Dopasowanie do historii produkcji")
            st.caption("Ocena dopasowania należy do użytkownika: niski błąd nie dowodzi "
                       "poprawności modelu, ponieważ różne zestawy parametrów mogą dawać "
                       "podobne krzywe produkcji.")
        else:
            st.error("Plik musi zawierać kolumny 'czas' oraz 'wydatek'.")

if zakladka == ZAKLADKI[5]:
    st.markdown("Pliki do załączników pracy i do dalszej analizy.")
    kol1, kol2, kol3 = st.columns(3)
    kol1.download_button("Model statyczny (CSV)", zloze.do_ramki().to_csv(index=False).encode('utf-8'),
                         "model_statyczny.csv", "text/csv")
    kol2.download_button("Historia produkcji (CSV)",
                         s.historia_do_ramki().to_csv(index=False).encode('utf-8'),
                         "historia_produkcji.csv", "text/csv")
    kol3.download_button("Końcowe pole ciśnień (CSV)",
                         pd.DataFrame(s.p).to_csv(index=False).encode('utf-8'),
                         "pole_cisnien.csv", "text/csv")
    parametry = {**asdict(cfg_w),
                 'igip_m3': bilans['igip'],
                 'produkcja_skumulowana_m3': bilans['gp'],
                 'stopien_sczerpania_proc': s.wspolczynnik_sczerpania(),
                 'blad_bilansu_masy': bilans['blad_wzgledny'],
                 'odwierty': [{k: v for k, v in o.items() if k != 'q_last'} for o in s.odwierty]}
    st.download_button("Parametry i wyniki (JSON)",
                       json.dumps(parametry, ensure_ascii=False, indent=2,
                                  default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)),
                       "parametry_symulacji.json", "application/json")
    with st.expander("Podgląd parametrów symulacji"):
        st.json(json.loads(json.dumps(parametry, default=lambda o: o.tolist()
                                      if hasattr(o, 'tolist') else str(o))))