------------------------------------------------------------------------

# Lingwistyczne profile ugrupowań politycznych w Sejmie X kadencji

## Analiza interpelacji poselskich i stenogramów sejmowych metodami eksploracji tekstu

### Konspekt projektu zaliczeniowego

|||
|---|---|
| **Autor:** | Krzysztof Stawarz |
| **Kierunek:** | Informatyka Społeczna, II stopień |
| **Specjalność:** | Sztuczna inteligencja i data mining |
| **Przedmiot:**   | Text mining                         |
| **Prowadzący:**  | dr hab. inż. Maciej Wielgosz        |
| **Semestr:**     | zimowy 2024/2025                    |

------------------------------------------------------------------------

## 1. Kontekst i zainteresowanie badawcze

Komunikacja polityczna w erze cyfrowej podlega intensywnym przemianom. Parlament stanowi centralne miejsce dyskursu politycznego, a generowane w nim teksty — zarówno pisemne (interpelacje, druki), jak i mówione (wystąpienia plenarne) — tworzą bogaty korpus do analizy ilościowej. W przeciwieństwie do wypowiedzi w mediach społecznościowych, dokumenty parlamentarne mają charakter ustandaryzowany, są kompletne (obejmują wszystkich posłów) i w całości dostępne publicznie poprzez API Sejmu RP.

X kadencja Sejmu (od listopada 2023) charakteryzuje się specyficzną konfiguracją polityczną: po raz pierwszy od 2015 roku koalicja rządowa obejmuje partie centrolewicowe i liberalne, podczas gdy Prawo i Sprawiedliwość przeszło do opozycji. Ta zmiana ról (rządzący ↔ opozycja) stwarza unikalną okazję do zbadania, jak pozycja polityczna wpływa na język używany przez posłów — zarówno w komunikacji formalnej (interpelacje), jak i w spontanicznych wystąpieniach na sali plenarnej (stenogramy).

## 2. Problem badawczy

Główne pytanie badawcze brzmi: **Czy i w jaki sposób ugrupowania polityczne w Sejmie X kadencji różnią się pod względem profilu lingwistycznego?**

Pytania szczegółowe:

-   Jakie słowa i frazy są charakterystyczne (wyróżniające) dla poszczególnych klubów parlamentarnych?
-   Czy istnieją rozpoznawalne „słowniki partyjne" — zestawy terminów używanych niemal wyłącznie przez dane ugrupowanie?
-   Jakie tematy dominują w komunikacji poszczególnych partii (analiza tematyczna)?
-   Czy partie opozycyjne i rządzące różnią się stylem komunikacji (np. długość wypowiedzi, złożoność składniowa, emocjonalność)?
-   Jak rozkłada się aktywność komunikacyjna w czasie i w relacji do adresatów/tematów?
-   **Czy profil lingwistyczny partii różni się między komunikacją formalną (interpelacje) a spontaniczną (wystąpienia plenarne)?**

## 3. Cel projektu

Celem projektu jest stworzenie kompleksowej analizy porównawczej języka politycznego polskich ugrupowań parlamentarnych, zwieńczonej:

1.  **Raportem analitycznym** — dokument zawierający wyniki analiz statystycznych i wizualizacje wraz z interpretacją.
2.  **Interaktywną aplikacją Shiny** — narzędzie umożliwiające eksplorację danych: filtrowanie po partii, czasie, temacie, typie źródła; generowanie wykresów porównawczych; wyszukiwanie słów kluczowych.

## 4. Dane wejściowe (input)

### 4.1. Źródło danych

Oficjalne API Sejmu RP (`api.sejm.gov.pl`) — w pełni otwarte, bezpłatne, bez limitów zapytań.

Projekt wykorzystuje **dwa komplementarne źródła tekstowe**:

| Źródło | Endpoint API | Charakter | Funkcja w projekcie |
|----------------|----------------|----------------|-------------------------|
| **Interpelacje poselskie** | `/sejm/term10/interpellations` | Tekst pisany, formalny, przemyślany | Analiza priorytetów tematycznych, „agendy" partii |
| **Stenogramy posiedzeń** | `/sejm/term10/proceedings` | Tekst mówiony, spontaniczny, emocjonalny | Analiza retoryki, sentymentu, stylu komunikacji |

### 4.2. Źródło 1: Interpelacje poselskie

**Czym są interpelacje?**

Interpelacja to formalny, pisemny instrument kontroli parlamentarnej. Poseł kieruje do premiera lub ministra zapytanie dotyczące spraw o „zasadniczym charakterze", przedstawiając stan faktyczny i formułując pytania. Adresat ma 21 dni na pisemną odpowiedź (art. 192 Regulaminu Sejmu).

**Charakterystyka lingwistyczna:** - Tekst pisany, redagowany, często z udziałem asystentów - Język formalny, urzędowy - Struktura: opis problemu → pytania - Brak spontaniczności, niska emocjonalność - Tematyka: sprawy resortowe, lokalne, branżowe

**Wartość analityczna:** - Pokazuje **priorytety tematyczne** partii (o co pytają?) - Ujawnia **relacje z resortami** (które ministerstwa są „atakowane"?) - Pozwala badać **asymetrię opozycja–rząd** (kto pyta więcej?)

**Zakres danych:** \~5 000–10 000 dokumentów (X kadencja).

**Struktura rekordu:**

| Pole          | Opis                                        |
|---------------|---------------------------------------------|
| `title`       | Tytuł interpelacji                          |
| `bodyHTML`    | Treść interpelacji (HTML → czysty tekst)    |
| `from`        | Numer legitymacji posła (mapowanie na klub) |
| `to`          | Adresat (ministerstwo/urząd)                |
| `sentDate`    | Data wysłania                               |
| `receiptDate` | Data wpływu odpowiedzi                      |

### 4.3. Źródło 2: Stenogramy posiedzeń plenarnych

**Czym są stenogramy?**

Stenogramy to pełne transkrypcje wystąpień na posiedzeniach plenarnych Sejmu. Obejmują: przemówienia posłów, wypowiedzi ministrów, oświadczenia, repliki, a także adnotacje (np. „Oklaski", „Głos z sali", „Wesołość na sali").

**Charakterystyka lingwistyczna:** - Tekst mówiony, transkrybowany - Język zróżnicowany: od formalnego (expose) po potoczny (polemiki) - Spontaniczność, emocje, retoryka perswazyjna - Interakcje: przerywniki, riposty, reakcje sali - Tematyka: bieżąca agenda legislacyjna, wydarzenia polityczne

**Wartość analityczna:** - Pokazuje **styl retoryczny** partii (jak mówią?) - Umożliwia **analizę sentymentu** (emocje, ton) - Ujawnia **dynamikę debaty** (kto z kim polemizuje?) - Pozwala badać **różnice rejestrów** (język oficjalny vs. spontaniczny)

**Zakres danych:** Dziesiątki godzin nagrań/transkrypcji na posiedzenie, \~50+ posiedzeń w kadencji.

**Struktura rekordu:**

| Pole             | Opis                                             |
|------------------|--------------------------------------------------|
| `number`         | Numer posiedzenia                                |
| `date`           | Data posiedzenia                                 |
| `transcriptHTML` | Pełna transkrypcja (HTML)                        |
| `points`         | Lista punktów obrad z przypisanymi wystąpieniami |

### 4.4. Analizowane ugrupowania

-   Prawo i Sprawiedliwość
-   Koalicja Obywatelska
-   Konfederacja
-   Brauniści
-   Trzecia Droga
    -   Polska2050
    -   Polskie Stronnictwo Ludowe
-   Nowa Lewica
-   Razem

------------------------------------------------------------------------

## 5. Uzasadnienie metodologiczne: dlaczego te źródła?

### 5.1. Dlaczego API Sejmu, a nie media społecznościowe?

Potencjalnym źródłem danych o języku politycznym są media społecznościowe (Twitter/X, Facebook). Poniższa tabela uzasadnia wybór API Sejmu:

| Kryterium | API Sejmu | Twitter/X | Facebook |
|------------------|------------------|------------------|------------------|
| **Dostępność API** | Bezpłatne, otwarte, bez limitów | Płatne od 2023 (\~\$100+/mies.), restrykcyjne limity | Brak publicznego API dla treści postów |
| **Legalność** | Dane publiczne, pełna legalność | Scraping łamie ToS, ryzyko prawne | Scraping łamie ToS |
| **Kompletność** | 100% posłów, 100% dokumentów | Tylko posłowie z aktywnymi kontami (\~60-70%) | Niekompletne, zróżnicowana aktywność |
| **Reprezentatywność** | Oficjalna komunikacja parlamentarna | Komunikacja PR, często przez zespoły | Głównie content marketingowy |
| **Stabilność źródła** | Gwarantowana przez Kancelarię Sejmu | API zmieniało się wielokrotnie (2023: zamknięcie darmowego dostępu) | Ciągłe zmiany polityki dostępu |
| **Kontekst instytucjonalny** | Jasny: interpelacje, wystąpienia plenarne | Rozmyty: kampania, komentarze, memy | Rozmyty |
| **Porównywalność** | Standaryzowany format, te same reguły dla wszystkich | Różne style, długości, multimedia | Różne formaty |
| **Atrybucja autorstwa** | Pewna (legitymacja posła) | Często konta prowadzone przez zespoły | Często konta prowadzone przez zespoły |

**Wniosek:** API Sejmu oferuje dane **kompletne, legalne, stabilne i porównywalne** — cechy kluczowe dla rygorystycznej analizy ilościowej. Media społecznościowe, mimo atrakcyjności (spontaniczność, emocje), wprowadzają zbyt wiele zmiennych zakłócających i problemów prawno-technicznych.

### 5.2. Dlaczego dwa źródła (interpelacje + stenogramy)?

Wykorzystanie dwóch źródeł pozwala na **triangulację metodologiczną** i uchwycenie różnych wymiarów komunikacji politycznej:

| Wymiar | Interpelacje | Stenogramy |
|------------------|-----------------------------|-------------------------|
| **Rejestr językowy** | Formalny, pisany | Zróżnicowany, mówiony |
| **Spontaniczność** | Niska (tekst redagowany) | Wysoka (odpowiedzi ad hoc) |
| **Emocjonalność** | Niska | Wysoka |
| **Co mierzy** | Priorytety, agendę | Retorykę, styl |
| **Porównywalność** | Wysoka (standaryzowany format) | Średnia (różne konteksty wypowiedzi) |
| **Objętość na posła** | Średnia | Zróżnicowana (zależy od funkcji) |

**Synergia źródeł:**

1.  **Weryfikacja krzyżowa**: Czy partia mówi o tym samym w interpelacjach i na mównicy?
2.  **Analiza rejestrów**: Jak zmienia się język partii w zależności od kontekstu (formalny vs. spontaniczny)?
3.  **Pełniejszy obraz**: Interpelacje pokazują „co" (tematy), stenogramy — „jak" (retoryka).

### 5.3. Ograniczenia i zagrożenia dla trafności

| Ograniczenie | Wpływ | Mitygacja |
|-------------------------------|------------------|------------------------|
| **Interpelacje często pisane przez asystentów** | Język może nie odzwierciedlać indywidualnego stylu posła | Agregacja na poziomie partii, nie posła |
| **Stenogramy wymagają segmentacji** | Trudność przypisania wypowiedzi do posła | Wykorzystanie znaczników HTML z API |
| **Nierówna aktywność posłów** | Niektórzy mówią więcej | Normalizacja (tf-idf, proporcje) |
| **Kadencja trwa \~1 rok** | Ograniczona możliwość analizy trendów czasowych | Skupienie na porównaniach między partiami |
| **Brak danych z mediów społecznościowych** | Utrata wymiaru „nieformalnego" | Stenogramy częściowo kompensują (spontaniczność) |

### 5.4. Alternatywne źródła — dlaczego odrzucone?

| Źródło | Powód odrzucenia |
|-----------------------|-------------------------------------------------|
| **Twitter/X** | Płatne API, niekompletność, problemy prawne |
| **Facebook** | Brak API, scraping nielegalny |
| **Wywiady medialne** | Brak ustrukturyzowanego źródła, trudność agregacji |
| **Programy partyjne** | Za mało danych (1 dokument/partia/wybory), brak dynamiki czasowej |
| **Druki sejmowe** | Język prawniczy, często identyczny między partiami (projekty rządowe) |

------------------------------------------------------------------------

## 6. Metodologia i narzędzia

### 6.1. Pipeline przetwarzania

```         
┌─────────────────────────────────────────────────────────────────┐
│ 1. POZYSKANIE DANYCH                                            │
│    API Sejmu → JSON → DataFrame                                 │
│    - /interpellations (interpelacje)                            │
│    - /proceedings (stenogramy)                                  │
│    - /MP (mapowanie posłów na kluby)                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING                                                │
│    - Usunięcie znaczników HTML                                  │
│    - Segmentacja stenogramów na wypowiedzi                      │
│    - Tokenizacja → Lematyzacja (udpipe, model pl)               │
│    - Usunięcie stopwords (lista polska + custom)                │
│    - Tagowanie POS (części mowy)                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. ANALIZA ILOŚCIOWA                                            │
│    - Częstotliwość słów (raw, normalized)                       │
│    - tf-idf per partia                                          │
│    - Macierz dokument-termin (DTM)                              │
│    - N-gramy (bigramy, trigramy)                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. MODELOWANIE                                                  │
│    - LDA (Latent Dirichlet Allocation) → tematy                 │
│    - Analiza sentymentu (słownik + ML)                          │
│    - Word embeddings (opcjonalnie: fastText pl)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. WIZUALIZACJA I RAPORTOWANIE                                  │
│    - ggplot2 (wykresy statyczne)                                │
│    - Shiny (aplikacja interaktywna)                             │
│    - R Markdown (raport PDF)                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2. Narzędzia techniczne

| Zadanie                | Pakiety R                                      |
|------------------------|------------------------------------------------|
| Pobieranie danych      | `httr2`, `jsonlite`                            |
| Przetwarzanie tekstu   | `tidytext`, `quanteda`, `udpipe`, `stringr`    |
| Modelowanie tematyczne | `topicmodels`, `stm`, `LDAvis`                 |
| Analiza sentymentu     | `sentimentr`, `syuzhet` (z polskim słownikiem) |
| Wizualizacja           | `ggplot2`, `ggwordcloud`, `ggraph`, `plotly`   |
| Aplikacja interaktywna | `shiny`, `shinydashboard`, `DT`, `bslib`       |

------------------------------------------------------------------------

## 7. Planowane wizualizacje

### 7.1. Wizualizacje porównawcze (interpelacje + stenogramy)

| \# | Wizualizacja | Opis | Źródło danych |
|--------------|----------------------|--------------|-----------------------|
| 1 | **Chmury słów porównawcze** | Osobna chmura dla każdej partii, wielkość ∝ tf-idf | Oba |
| 2 | **Wykres radarowy profili tematycznych** | Osie = tematy LDA, wartości = proporcja dokumentów | Oba |
| 3 | **Heatmapa partia × ministerstwo** | Intensywność interpelowania resortów | Interpelacje |
| 4 | **Timeline aktywności** | Liczba dokumentów w czasie, z wydarzeniami politycznymi | Oba |
| 5 | **Scatter plot podobieństwa (PCA/t-SNE)** | Partie jako punkty, odległość = różnica lingwistyczna | Oba |
| 6 | **Bar chart słów wyróżniających** | Top 10 słów o najwyższym tf-idf per partia | Oba |
| 7 | **Boxplot sentymentu** | Rozkład sentymentu wypowiedzi per partia | Stenogramy |
| 8 | **Sieć współwystępowania słów** | Graf: wierzchołki = słowa, krawędzie = współwystępowanie | Stenogramy |
| 9 | **Faceted comparison: formalny vs. spontaniczny** | Te same metryki dla interpelacji i stenogramów | Oba |

### 7.2. Funkcjonalności aplikacji Shiny

-   Wybór partii do porównania (checkboxy)
-   Filtry czasowe (date range picker)
-   Wybór źródła (interpelacje / stenogramy / oba)
-   Wyszukiwarka słów kluczowych z kontekstem (KWIC)
-   Eksport wykresów (PNG, SVG)
-   Tabela z surowymi danymi (sortowalna, filtrowalna)

------------------------------------------------------------------------

## 8. Potencjalne wnioski (hipotezy do weryfikacji)

### 8.1. Hipotezy dotyczące różnic między partiami

-   **H1**: Partie opozycyjne (PiS, Konfederacja) składają więcej interpelacji niż partie rządzące (efekt asymetrii opozycja–rząd).

-   **H2**: PiS najczęściej używa słów z pola semantycznego „bezpieczeństwo", „granica", „suwerenność"; KO — „praworządność", „demokracja", „Europa".

-   **H3**: Konfederacja ma najbardziej wyróżniający się profil lingwistyczny (największa odległość od centroidu wszystkich partii w przestrzeni tf-idf).

-   **H4**: Partia Razem jako jedyna regularnie używa terminologii ekonomii heterodoksyjnej (np. „redystrybucja", „nierówności", „kapitał", „praca").

-   **H5**: Istnieje korelacja między pozycją na osi lewica–prawica a profilem tematycznym (np. lewica → tematy socjalne; prawica → bezpieczeństwo).

### 8.2. Hipotezy dotyczące różnic między źródłami

-   **H6**: Sentyment wypowiedzi w stenogramach jest bardziej zróżnicowany (wyższa wariancja) niż w interpelacjach.

-   **H7**: Słownictwo w interpelacjach jest bardziej techniczne/urzędowe, w stenogramach — bardziej potoczne i emocjonalne.

-   **H8**: Partie opozycyjne wykazują wyższy poziom negatywnego sentymentu w stenogramach niż partie rządzące.

------------------------------------------------------------------------

## 9. Rezultaty projektu (output)

1.  **Repozytorium kodu** (GitHub)
    -   Skrypty do pobierania danych
    -   Pipeline preprocessingu
    -   Kod analiz i wizualizacji
    -   Dokumentacja (README)
2.  **Raport PDF** (\~15–20 stron)
    -   Opis metodologii
    -   Wyniki analiz z wizualizacjami
    -   Interpretacja i wnioski
    -   Dyskusja ograniczeń
3.  **Aplikacja Shiny**
    -   Wdrożona na `shinyapps.io` lub prezentowana lokalnie
    -   Interaktywna eksploracja danych
4.  **Zbiór danych** (CSV/RDS)
    -   Oczyszczone interpelacje i stenogramy
    -   Potencjalnie do dalszych badań (open data)

------------------------------------------------------------------------

## 10. Harmonogram realizacji

| Tydzień | Daty | Zadania |
|---------------------------|------------------|---------------------------|
| 1 | 16–22.12 | Pobranie danych z API (interpelacje + stenogramy), eksploracja struktury, mapowanie posłów→partie |
| 2 | 23–29.12 | Preprocessing tekstu, lematyzacja, segmentacja stenogramów |
| 3 | 30.12–05.01 | Analiza tf-idf, pierwsze wizualizacje porównawcze |
| 4 | 06–12.01 | Modelowanie tematyczne (LDA), analiza sentymentu, zaawansowane wizualizacje |
| 5 | 13–19.01 | Budowa aplikacji Shiny, integracja wykresów |
| 6 | 20–26.01 | Redakcja raportu, dokumentacja, testy, poprawki |

------------------------------------------------------------------------