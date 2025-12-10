# Konspekt projektu zaliczeniowego

**Przedmiot:** Text Mining

---

# Lingwistyczne profile ugrupowań politycznych w Sejmie X kadencji

## Analiza interpelacji poselskich metodami eksploracji tekstu

| | |
|---|---|
| **Autor:** | Krzysztof Stawarz |
| **Kierunek:** | Informatyka Społeczna, II stopień |
| **Specjalność:** | Sztuczna inteligencja i data mining |
| **Prowadzący:** | dr hab. inż. Maciej Wielgosz |
| **Semestr:** | zimowy 2024/2025 |

---

## 1. Kontekst i zainteresowanie badawcze

Komunikacja polityczna w erze cyfrowej podlega intensywnym przemianom. Interpelacje poselskie, jako sformalizowany instrument kontroli parlamentarnej, stanowią bogate źródło danych tekstowych, które pozwalają na ilościową analizę dyskursu politycznego. W przeciwieństwie do wypowiedzi medialnych czy postów w mediach społecznościowych, interpelacje mają charakter ustandaryzowany i są w całości dostępne publicznie poprzez API Sejmu RP.

X kadencja Sejmu (od listopada 2023) charakteryzuje się specyficzną konfiguracją polityczną: po raz pierwszy od 2015 roku koalicja rządowa obejmuje partie centrolewicowe i liberalne, podczas gdy Prawo i Sprawiedliwość przeszło do opozycji. Ta zmiana ról (rządzący ↔ opozycja) stwarza unikalną okazję do zbadania, jak pozycja polityczna wpływa na język używany przez posłów.

## 2. Problem badawczy

Główne pytanie badawcze brzmi: **Czy i w jaki sposób ugrupowania polityczne w Sejmie X kadencji różnią się pod względem profilu lingwistycznego swoich interpelacji?**

Pytania szczegółowe:

- Jakie słowa i frazy są charakterystyczne (wyróżniające) dla poszczególnych klubów parlamentarnych?
- Czy istnieją rozpoznawalne „słowniki partyjne" — zestawy terminów używanych niemal wyłącznie przez dane ugrupowanie?
- Jakie tematy dominują w interpelacjach poszczególnych partii (analiza tematyczna)?
- Czy partie opozycyjne i rządzące różnią się stylem komunikacji (np. długość interpelacji, złożoność składniowa)?
- Jak rozkłada się aktywność interpelacyjna w czasie i w relacji do adresatów (ministerstw)?

## 3. Cel projektu

Celem projektu jest stworzenie kompleksowej analizy porównawczej języka politycznego polskich ugrupowań parlamentarnych, zwieńczonej:

1. **Raportem analitycznym** — dokument zawierający wyniki analiz statystycznych i wizualizacje wraz z interpretacją.
2. **Interaktywną aplikacją Shiny** — narzędzie umożliwiające eksplorację danych: filtrowanie po partii, czasie, temacie; generowanie wykresów porównawczych; wyszukiwanie słów kluczowych.

## 4. Dane wejściowe (input)

### 4.1. Źródło danych

Oficjalne API Sejmu RP (`api.sejm.gov.pl`) — endpoint `/sejm/term10/interpellations`.

### 4.2. Zakres danych

Wszystkie interpelacje poselskie X kadencji (od 13.11.2023 do momentu pobrania danych). Szacunkowa liczba: 5 000–10 000 dokumentów.

### 4.3. Struktura pojedynczego rekordu

| Pole | Opis |
|------|------|
| `title` | Tytuł interpelacji |
| `bodyHTML` | Treść interpelacji (HTML → czysty tekst) |
| `from` | Numer legitymacji posła (mapowanie na klub) |
| `to` | Adresat (ministerstwo/urząd) |
| `sentDate` | Data wysłania |
| `receiptDate` | Data wpływu odpowiedzi |

### 4.4. Analizowane ugrupowania

| Klub/koło | Pozycja | Orientacja |
|-----------|---------|------------|
| Prawo i Sprawiedliwość (PiS) | Opozycja | Prawica |
| Koalicja Obywatelska (KO) | Koalicja rządząca | Centrum/centrolewica |
| Polska 2050 — Trzecia Droga | Koalicja rządząca | Centrum |
| Polskie Stronnictwo Ludowe (PSL) | Koalicja rządząca | Centrum/agraryzm |
| Konfederacja | Opozycja | Prawica wolnościowa |
| Lewica | Koalicja rządząca | Lewica |
| Razem | Koalicja rządząca* | Lewica |

*Razem wyodrębniło się z klubu Lewicy w trakcie kadencji.

## 5. Metodologia i narzędzia

### 5.1. Pipeline przetwarzania

1. **Pozyskanie danych**: skrypt R/Python odpytujący API Sejmu, zapis do formatu CSV/RDS.
2. **Preprocessing**: usunięcie znaczników HTML, tokenizacja, lematyzacja (pakiet `udpipe` z modelem dla języka polskiego), usunięcie stopwords.
3. **Wzbogacenie**: mapowanie posłów na kluby parlamentarne (osobne API `/MP`).
4. **Analiza ilościowa**: obliczenie metryk tf-idf, budowa macierzy dokument-termin.
5. **Modelowanie tematyczne**: algorytm LDA (Latent Dirichlet Allocation) do identyfikacji ukrytych tematów.
6. **Wizualizacja**: pakiet `ggplot2` do wykresów statycznych.
7. **Aplikacja**: framework `Shiny` do interaktywnej eksploracji.

### 5.2. Narzędzia techniczne

| Zadanie | Pakiety R |
|---------|-----------|
| Pobieranie danych | `httr2`, `jsonlite` |
| Przetwarzanie tekstu | `tidytext`, `quanteda`, `udpipe` |
| Modelowanie tematyczne | `topicmodels`, `stm` |
| Wizualizacja | `ggplot2`, `ggwordcloud`, `plotly` |
| Aplikacja interaktywna | `shiny`, `shinydashboard`, `DT` |

## 6. Planowane wizualizacje

1. **Chmury słów porównawcze** — osobna chmura dla każdej partii, wielkość słowa proporcjonalna do tf-idf (nie surowej częstotliwości).

2. **Wykres radarowy profili tematycznych** — osie odpowiadają tematom z LDA (np. „ekonomia", „bezpieczeństwo", „UE", „zdrowie"), wartości to proporcja interpelacji danej partii w danym temacie.

3. **Heatmapa partia × ministerstwo** — macierz pokazująca, które partie najczęściej interpelują które resorty.

4. **Timeline aktywności** — wykres liniowy liczby interpelacji w czasie, z podziałem na partie; możliwość nałożenia wydarzeń politycznych.

5. **Scatter plot podobieństwa lingwistycznego** — redukcja wymiarowości (PCA lub t-SNE) wektorowych reprezentacji partii; odległość = różnica w języku.

6. **Bar chart słów wyróżniających** — top 10 słów o najwyższym tf-idf dla każdej partii (faceted plot).

## 7. Potencjalne wnioski (hipotezy do weryfikacji)

Na podstawie wiedzy o polskiej scenie politycznej można sformułować hipotezy, które projekt pozwoli zweryfikować:

- **H1**: Partie opozycyjne (PiS, Konfederacja) składają więcej interpelacji niż partie rządzące (efekt asymetrii opozycja–rząd).

- **H2**: PiS najczęściej używa słów z pola semantycznego „bezpieczeństwo" i „suwerenność"; KO — „praworządność" i „demokracja".

- **H3**: Konfederacja ma najbardziej wyróżniający się profil lingwistyczny (największa odległość od centroidu wszystkich partii).

- **H4**: Partia Razem jako jedyna regularnie używa terminologii ekonomii heterodoksyjnej (np. „redystrybucja", „nierówności", „kapitał").

- **H5**: Istnieje korelacja między pozycją na osi lewica–prawica a profilem tematycznym interpelacji.

## 8. Rezultaty projektu (output)

1. **Repozytorium kodu** (GitHub) — skrypty do replikacji analizy.
2. **Raport PDF** — dokument ok. 15–20 stron z opisem metodologii, wynikami i wnioskami.
3. **Aplikacja Shiny** — wdrożona na `shinyapps.io` lub prezentowana lokalnie.
4. **Zbiór danych** — oczyszczone interpelacje w formacie CSV (potencjalnie do dalszych badań).

## 9. Harmonogram realizacji

| Tydzień | Zadania |
|---------|---------|
| 1 (16–22.12) | Pobranie danych z API, eksploracja struktury, mapowanie posłów |
| 2 (23–29.12) | Preprocessing tekstu, lematyzacja, usunięcie stopwords |
| 3 (30.12–05.01) | Analiza tf-idf, pierwsze wizualizacje porównawcze |
| 4 (06–12.01) | Modelowanie tematyczne LDA, zaawansowane wizualizacje |
| 5 (13–19.01) | Budowa aplikacji Shiny, integracja wykresów |
| 6 (20–26.01) | Redakcja raportu, dokumentacja, testy, poprawki |

---

**Uwaga**: Projekt realizowany jest wyłącznie w celach edukacyjnych. Analiza opiera się na publicznie dostępnych danych udostępnianych przez Kancelarię Sejmu RP.
