# Historical Identity Crosswalk — SEC EDGAR (v0.4)

## Σκοπός και αυστηροί περιορισμοί

Το ιστορικό S&P500 είναι **διαφορετικό** από το baseline των 24 μετοχών και το forward-only snapshot των 165 μετοχών.

Η SEC παρέχει **CIK, νομικά ονόματα εταιρειών και καταχωρήσεις filings**, όχι αξιόπιστη ιστορική λίστα ticker→CIK για κάθε περίοδο. Το `master.idx` έχει `CIK | Company Name | Form Type | Date Filed | Filename`, ενώ το `submissions/CIK...json` έχει `name`, `formerNames`, **σημερινά** `tickers` και filings. Καμία από αυτές τις πηγές μόνη της δεν τεκμηριώνει ότι το παλιό σύμβολο `AGN` αντιστοιχεί αυτόματα σε συγκεκριμένο CIK ή σε σταθερό security ID. Συνεπώς τα αυτόματα matches είναι **υποψήφια** (`candidate_unverified`) και δεν γίνονται ποτέ αυτόματα `verified`.

Επίσημες πηγές: [SEC indexing](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data), [SEC submissions API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [fair access](https://www.sec.gov/about/privacy-information). Χρησιμοποίησε πραγματικό SEC_USER_AGENT με ονοματεπώνυμο και email. Μην τρέχεις παράλληλους collectors ή πολλά SEC requests/sec.

## Βήμα 1 — ενημέρωση κώδικα και πλήρης inventory unresolved tickers

Από το root του GitHub repository σε PowerShell:

```powershell
 git pull origin main
 python -m pip install -r requirements-data.txt
 python -m pytest -q
 $env:SEC_USER_AGENT="Your Real Name real-email@example.com"
 python 12_build_historical_crosswalk.py --skip-indexes
```

Χρειάζονται τα **πλήρη** `results/free_source_coverage/annual_coverage.csv` **και** `per_security_coverage.csv` από την εκτέλεση του script 11. Αν έχεις μόνο το πρώτο, ξανατρέξε `11_collect_sec_price_audit.py --max-symbols 20` (ή χρησιμοποίησε τα ήδη υπάρχοντα τοπικά outputs). Ο script 12 ελέγχει ότι για **κάθε** έτος 2015–2026 οι ανά-ticker εγγραφές είναι ίσες με το πλήθος `memberships`: απαγορεύει την ατελή λίστα. Το `--skip-indexes` δημιουργεί **όλη** τη λίστα unresolved χωρίς downloads.

Παραγωγή:

- `results/historical_crosswalk/unresolved_review_queue.csv`: όλα τα μοναδικά unresolved symbols και τα έτη στα οποία εμφανίζονται.
- `results/historical_crosswalk/candidate_crosswalk.csv`: προτάσεις ή `unresolved` ανά ιστορικό διάστημα συμμετοχής.
- `results/historical_crosswalk/crosswalk_audit.json`: ποσότητες, SEC calls, σφάλματα.

## Βήμα 2 — αυτόματα υποψήφια matches από SEC indexes

```powershell
 python 12_build_historical_crosswalk.py --index-years 2015,2018,2021,2025 --index-quarters 1
```

Τα αρχικά **unverified company-name search hints** βρίσκονται στο `data/universe/historical/ticker_name_hints.csv` για AABA, ABMD, ADS, AET, AGN, ALXN, ANDV, ANTM, ARG, ARNC, BCR. Πρόκειται **μόνο για ερωτήματα αναζήτησης**, όχι για επιβεβαιωμένα CIKs ή εταιρικά γεγονότα. Προσθέτεις νέα `historical_ticker, company_name_hint, cik_hint, hint_source` για όσα άλλα `unresolved_review_queue.csv` δεν βγάλουν προτάσεις.

Το script συγκρίνει ονόματα στους SEC `master.idx`, βρίσκει υποψήφια CIKs και κατεβάζει `submissions` **μόνο** για αυτά τα CIKs. Τα αρχεία αποθηκεύονται τοπικά υπό `data/cache/sec/` (`master_indexes/YYYYQn.idx.gz`, `submissions/CIK....json`). Μπορείς να το ξανατρέξεις χωρίς να ξανακατεβάσεις όσα αρχεία υπάρχουν στην cache.

Για μόνο λίγα unresolved symbols ή δοκιμή από cache:

```powershell
 python 12_build_historical_crosswalk.py --max-symbols 10 --offset 0
 python 12_build_historical_crosswalk.py --max-symbols 10 --offset 10
 python 12_build_historical_crosswalk.py --offline
```

Το `--max-symbols` περιορίζει μόνο το expensive SEC lookup, **όχι** την πλήρη unresolved review queue. Οι υποψήφιοι δεν αποδεικνύουν ticker identity και δεν είναι κατάλληλοι για training.

## Βήμα 3 — evidence και verified_crosswalk

Αντιγράφεις το `data/universe/historical/verified_crosswalk_TEMPLATE.csv` στο **τοπικό** `data/universe/historical/verified_crosswalk.csv` και εγγράφεις ΜΟΝΟ mappings που έλεγξες από τεκμήρια του ιστορικού ticker και ξεχωριστής εμπορεύσιμης share class.

Απαραίτητες στήλες:

`security_id,cik,historical_ticker,valid_from,valid_to,company_name,corporate_action,mapping_source,verification_status`

- `security_id`: σταθερό αναγνωριστικό **μετοχής/share class**, όχι μόνο CIK ούτε ticker. Κράτα ξεχωριστά τις συνδεδεμένες οντότητες σε merger.
- `cik`: εκδότης SEC (αριθμός, αποθηκεύεται normalized integer/10-digit στη μεταφορά).
- `valid_from` inclusive / `valid_to` exclusive, ημερομηνίες ISO, κενό `valid_to` επιτρέπεται.
- `mapping_source`: συγκεκριμένο πλήρες `https://...` URL στοιχείου που στηρίζει την αντιστοίχιση. **Ο έλεγχος του κώδικα απαιτεί URL αλλά δεν μπορεί να αποδείξει ότι είναι επαρκές — χρειάζεται ανεξάρτητη κρίση.**
- `verification_status`: ακριβώς `verified` **μόνο μετά από έλεγχο**. Μην αντιγράφεις `candidate_unverified` ως `verified` χωρίς ανεξάρτητο τεκμήριο.

Ο parser απορρίπτει υποψήφια statuses, άκυρες ημερομηνίες, κενό/απλοϊκό security ID, CIK χωρίς αριθμό και overlap intervals για το ίδιο symbol. Το τοπικό αρχείο `verified_crosswalk.csv` είναι **gitignored**.

## Βήμα 4 — resumed SEC CompanyFacts & targeted price probe

```powershell
 python 11_collect_sec_price_audit.py --max-symbols 20 --offset 0 --prices
 python 11_collect_sec_price_audit.py --max-symbols 20 --offset 20 --prices
 python 11_collect_sec_price_audit.py --max-symbols 20 --offset 40 --prices
```

Χωρίς `--refresh`, οι καταχωρήσεις SEC CompanyFacts αποθηκεύονται **ανά CIK** σε `data/cache/sec/companyfacts/CIK##########.json` και τα Yahoo presence probes ανά symbol σε `data/cache/yfinance/`. Η επανάληψη μιας εντολής μετά από διακοπή επαναχρησιμοποιεί τα επιτυχημένα caches. Τα `annual_coverage.csv` και `per_security_coverage.csv` παράγονται εκ νέου και **συμπεριλαμβάνουν τα cached αποτελέσματα των προηγούμενων παρτίδων**, όχι μόνο όσα επιλέχθηκαν στην τελευταία εκτέλεση. Χρησιμοποίησε `--refresh` μόνο αν θέλεις εκ νέου δοκιμή (σβήνει/ανανεώνει caches του επιλεγμένου batch).

**Προσοχή:** Ακόμη και `verified_crosswalk` δεν εγγυάται terminal returns στο Yahoo. Το `no_bars_unverified` δεν ισοδυναμεί με delisted. Το `raw_*_present=True` σημαίνει μόνο διαθεσιμότητα tags, όχι ότι έχουμε πέντε σωστά PIT χαρακτηριστικά. Δεν αλλάζουμε ML experiments ή το αρχικό baseline σε αυτή τη φάση.

## Status / κριτήρια ολοκλήρωσης

- [x] Ασφαλής inventory όλων των unresolved ανά έτος (2015–2026)
- [x] Αυτόματα SEC name→CIK candidates και SEC formerNames corroboration
- [x] Χωρίς αυτόματο CIK approval · temporal verified precedences · conflict checks
- [x] Cache ανά CIK και resumable batches
- [ ] Χειροκίνητη επαλήθευση των ιστορικών mappings
- [ ] Πραγματικός price & delisting audit / corporate-action reconciliation
- [ ] Πλήρης point-in-time 5-feature coverage ανά GICS sector
- [ ] EODHD μόνο αφού μετρηθούν πραγματικά delisting-related gaps