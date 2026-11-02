# Symulator przepływu gazu w złożu niejednorodnym z jednostkami hydraulicznymi (HU)

Dwuwymiarowy, dynamiczny model przepływu gazu ziemnego w niejednorodnym złożu,
w którym rozkład przepustowości odtwarzany jest na podstawie jednostek hydraulicznych
(HU) wyznaczanych metodą FZI. Aplikacja z interfejsem w przeglądarce (Streamlit).

Program powstał w ramach pracy inżynierskiej *„Budowa dynamicznego modelu przepływu gazu
w niejednorodnym złożu z wykorzystaniem hydraulicznych jednostek (HU) udostępnianych skał
zbiornikowych”* (AGH, Wydział Wiertnictwa, Nafty i Gazu, kierunek Inżynieria Naftowa i Gazownicza).

**Aplikacja online:** *(adres zostanie uzupełniony po wdrożeniu)*

## Co robi program

- **Model statyczny i jednostki hydrauliczne** – wskaźniki RQI i FZI (Amaefule i in., 1993),
  podział na jednostki metodą k-średnich na log FZI, dobór liczby jednostek metodą
  „łokcia” (Kneedle), estymacja przepuszczalności z porowatości w każdej jednostce.
- **Właściwości gazu (PVT)** – współczynnik ściśliwości Z (Hall–Yarborough),
  lepkość (Lee–Gonzalez–Eakin), współczynnik objętościowy B<sub>g</sub>, pseudociśnienie m(p).
- **Model dynamiczny** – bilans masy w objętościach skończonych na siatce kartezjańskiej
  (przewodności międzyblokowe ze średniej harmonicznej), schemat w pełni niejawny
  (Euler wstecz) rozwiązywany metodą Newtona w pseudociśnieniu.
- **Odwierty** – model Peacemana, sterowanie zadanym wydatkiem z ograniczeniem minimalnego
  ciśnienia dennego i wyłączaniem odwiertu poniżej wydatku minimalnego.
- **Kontrola wyników** – zasoby geologiczne metodą objętościową, bilans materiałowy
  (p/Z względem wydobycia skumulowanego), porównanie z rozwiązaniem analitycznym.

## Uruchomienie lokalne

Wymagany Python 3.10–3.12.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplikacja otworzy się w przeglądarce pod adresem `http://localhost:8501`.
Po starcie domyślnie wczytywane jest przykładowe złoże `zloze_testowe.csv`;
można je zastąpić własnym plikiem.

## Format danych wejściowych

Plik CSV, w którym każdy wiersz opisuje jeden blok siatki:

| Kolumna | Opis | Wymagana |
|---|---|---|
| `Porowatosc` | porowatość efektywna [-] | tak |
| `Przepuszczalnosc` | przepuszczalność [mD] | tak |
| `I`, `J` lub `x`, `y` | położenie bloku w siatce | nie |

Przykład: `zloze_testowe.csv` – syntetyczne złoże 30 × 30 bloków o statystykach
czterech jednostek hydraulicznych triasowego złoża klastycznego opisanego przez
Ellabad, Corbetta i Strauba (2001).

## Pliki

| Plik | Zawartość |
|---|---|
| `app.py` | interfejs aplikacji (Streamlit) |
| `statyka.py` | model statyczny: FZI, jednostki hydrauliczne, metoda łokcia, mapy |
| `pvt.py` | właściwości gazu ziemnego |
| `symulator_fvm.py` | silnik symulacji dynamicznej (metoda objętości skończonych, ang. *finite volume method*, FVM) i model odwiertów |
| `generator_zloza.py` | generatory syntetycznych złóż testowych |
| `badanie_hu.py` | badanie wpływu podziału na jednostki hydrauliczne na wyniki symulacji |
| `testy.py` | testy weryfikacyjne (PVT, bilans masy, zbieżność, odwierty, jednostki hydrauliczne) |
| `zloze_testowe.csv` | przykładowe dane wejściowe |

Testy uruchamia się poleceniem `python testy.py`.

## Autor

Wiktor Dulewicz, AGH w Krakowie
