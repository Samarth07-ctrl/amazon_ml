# Dataset Blockers — Business Entity Resolution

## Purpose

This document lists **only the data characteristics that can block or materially degrade entity matching** in the provided dataset. It does not propose solutions or preprocessing methods.

Raw `business_name` and `business_address` values must be treated as source data; these observations describe the problems present in those values.

---

## 1. Noisy Business Names

Business names contain characters and formatting that can obscure the underlying identity.

Observed examples include:

- Leading or trailing symbols: `@`, `<<`, `>>`, `+`, `*`, `--`
- Embedded non-semantic symbols
- Leading junk such as `...`
- Domain-style names ending in `.com`
- Punctuation and formatting differences between sources

Example:
- `Sweetbarbershop.Com`

### Why this blocks matching

Two records representing the same business may have substantially different raw strings even when their meaningful name content is almost identical. Exact matching therefore becomes unreliable, while naive normalization can also create false matches.

---

## 2. Multilingual and Mixed-Script Names

The dataset contains genuine non-Latin and multilingual business names.

Observed scripts include:

- Gujarati
- Kannada
- Malayalam
- Hindi / Devanagari
- Tamil
- Latin
- Mixed-script records

Examples include:

- `આકૃષ્ટ ફાઉન્ડેશન પ્રાઇવેટ લિમિટેડ`
- `ಬ್ಯಾಂಕ್ ಇಂಜಿನિયરಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್`
- `കൃഷ്ണ ഇന്റർനാഷണൽ ഹാർഡ്‌വെയർ പ്രൈവറ്റ് ലിമിറ്റഡ്`

A single record can also contain multiple scripts, such as a native-script business name combined with a Latin country or state suffix.

### Why this blocks matching

Representations designed only for English/Latin text can fail to recognize equivalent names across scripts. Converting everything to ASCII can also destroy information needed to distinguish businesses.

---

## 3. Accented-Character Noise

Some names contain accented characters that appear inconsistent with the surrounding language/context.

Observed examples include:

- `Smith Próduce of Spokane`
- `Horizon Regional Réal`
- `Twisted Cóffee`
- `Janeta's Select Mórtgage`

These appear to be character-level noise rather than encoding corruption.

### Why this blocks matching

A name such as `Produce` and a noisy variant such as `Próduce` can represent the same business while having different raw character sequences. Similarity methods that are sensitive to character substitutions can therefore lose matching signal.

---

## 4. Typos and Character-Level Errors

The data contains ordinary spelling errors involving character substitutions and transpositions.

A documented example is:

- `Indai` instead of `India`

### Why this blocks matching

A small edit can significantly change exact strings and token representations. At the same time, treating every small edit as evidence of identity can introduce false matches between genuinely different businesses.

---

## 5. Legal-Suffix Variation

Business names differ in their use, spelling, and placement of legal entity suffixes.

Examples of relevant forms include:

- `Pvt`
- `Ltd`
- `LLC`
- `Inc`
- `Corp`

The suffix may be:

- Present in one source and absent in another
- Written differently
- Placed differently
- Combined with different punctuation

### Why this blocks matching

The same business can appear to have different names because of legal-form differences. Conversely, legal suffixes are common across unrelated businesses, so treating them as strong identity evidence can create false matches.

---

## 6. Word-Order Variation

The same business name can contain the same words in different orders across sources.

### Why this blocks matching

Exact string comparison is sensitive to token order. Two equivalent names can therefore appear dissimilar despite sharing the same meaningful tokens.

---

## 7. Missing Addresses

`business_address` is genuinely missing for some records.

There are two distinct forms:

1. A genuinely null/empty address field.
2. An address containing the literal text `NULL`.

Example:

`GJ, Ahmedabad, 209, NULL`

### Why this blocks matching

Address information is unavailable or partially unavailable for some records. Missing address information therefore cannot safely be interpreted as evidence that two businesses are different.

The literal string `NULL` can also be mistaken for an actual address token if it is not recognized as a missing-value sentinel.

---

## 8. Incomplete Address Information

Addresses can contain different subsets of their components across sources.

Some records may contain:

- City but no street
- Street but no locality
- Partial postal information
- Missing address components

### Why this blocks matching

The same physical business may have substantially different address strings because one source contains only a subset of the available information.

---

## 9. Address Abbreviation Differences

Address components may be abbreviated differently across records.

### Why this blocks matching

Equivalent addresses can have different textual representations. Direct string comparison can therefore treat the same location as different.

---

## 10. Address Reordering

Address components can occur in different orders across sources.

### Why this blocks matching

The same address can produce different raw strings even when the underlying components are identical. Order-sensitive comparison can therefore underestimate similarity.

---

## 11. Numbers in Names and Addresses

Business names and addresses can contain meaningful numeric information.

Examples of possible patterns include:

- `3M`
- `7-Eleven`
- `Studio 54`
- Building or street numbers

### Why this blocks matching

Numbers can be important identity/location signals. Removing them indiscriminately can collapse distinct businesses into similar representations.

---

## 12. Cross-Script / Transliteration Variation

Equivalent businesses may potentially appear using different writing systems or transliterated forms.

This is particularly relevant because the dataset contains multiple Indian scripts and the test data includes a country not represented in training.

### Why this blocks matching

Two representations of the same business may share little or no literal character sequence when written in different scripts. A representation based only on one script can therefore lose candidate matches.

---

## 13. Unseen Geographic Variation

Training data contains US and India records, while France is present in the test data.

### Why this blocks matching

Address conventions, abbreviations, ordering, and naming patterns can differ by country. Patterns learned too specifically from the training countries may not transfer directly to unseen geographic formats.

---

## 14. Combined Noise

The most difficult records may contain several blockers simultaneously.

For example, a single record can combine:

- Multilingual text
- Transliteration
- Legal-suffix variation
- Typos
- Punctuation noise
- Missing address components
- Address abbreviation
- Address reordering

### Why this blocks matching

Individual similarity signals can become unreliable when multiple forms of noise occur at once. A pipeline that depends heavily on one representation or one exact similarity measure can therefore miss genuine matches or produce false matches.

---

## Important Non-Blockers

The following observed phenomena should **not** be treated as genuine data corruption based on the supplied investigation:

### Excel `#NAME?`

`#NAME?` observed in spreadsheet views is a viewer artifact caused by Excel interpreting certain leading characters as formula triggers. It is not evidence that the underlying TSV contains `#NAME?`.

### Mojibake such as `à¸...` / `à¤...`

The observed garbled text is an encoding/viewer artifact from interpreting UTF-8 data using an incompatible encoding. The underlying TSV contains correctly encoded native-script text.

These should therefore **not be added as matching-data noise without first checking the raw UTF-8 TSV**.

---

## Summary of Core Blockers

| Blocker | Primary affected field |
|---|---|
| Symbol/punctuation noise | Name |
| Multilingual / mixed scripts | Name |
| Accented-character noise | Name |
| Typos / character substitutions | Name |
| Legal-suffix variation | Name |
| Word-order variation | Name |
| Missing / `NULL` addresses | Address |
| Incomplete addresses | Address |
| Address abbreviations | Address |
| Address reordering | Address |
| Meaningful numbers | Name / Address |
| Cross-script / transliteration variation | Name |
| Unseen country/address conventions | Address / Name |
| Multiple simultaneous noise types | Name / Address |

**Key observation:** the central difficulty is not simply corrupted text. The same business can have multiple valid-but-different textual representations across sources, while unrelated businesses can share common normalized components. This creates a precision-vs-recall problem rather than a simple text-cleaning problem.
