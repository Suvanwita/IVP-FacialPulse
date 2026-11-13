# Synthetic evaluation summary

This file validates the pipeline logic with synthetic input values. It does not validate against real patient data.

- Total cases: 50
- Output structure valid: 50/50
- Predicted score inside the synthetic expected band: 46/50
- Average PHQ-8 score: 9.66

| Session | Input band | Audio | PHQ-8 | Severity | 95% CI | In expected band | Output valid |
|---|---|---:|---:|---|---|---:|---:|
| T01 | 0-4 | false | 3 | Minimal / None | 0.0 - 6.9 | True | True |
| T02 | 0-4 | false | 3 | Minimal / None | 0.0 - 6.9 | True | True |
| T03 | 0-4 | false | 3 | Minimal / None | 0.0 - 6.7 | True | True |
| T04 | 0-4 | false | 3 | Minimal / None | 0.0 - 6.7 | True | True |
| T05 | 0-4 | false | 3 | Minimal / None | 0.0 - 6.9 | True | True |
| T06 | 0-4 | false | 4 | Minimal / None | 0.4 - 7.3 | True | True |
| T07 | 0-4 | false | 4 | Minimal / None | 0.2 - 7.1 | True | True |
| T08 | 0-4 | false | 4 | Minimal / None | 0.1 - 7.0 | True | True |
| T09 | 0-4 | false | 3 | Minimal / None | 0.0 - 6.9 | True | True |
| T10 | 0-4 | false | 4 | Minimal / None | 0.1 - 7.1 | True | True |
| T11 | 0-4 | false | 4 | Minimal / None | 0.2 - 7.1 | True | True |
| T12 | 0-4 | false | 3 | Minimal / None | 0.0 - 6.7 | True | True |
| T13 | 0-4 | false | 3 | Minimal / None | 0.0 - 6.9 | True | True |
| T14 | 0-4 | false | 3 | Minimal / None | 0.0 - 6.7 | True | True |
| T15 | 5-9 | false | 6 | Mild | 2.7 - 9.6 | True | True |
| T16 | 5-9 | false | 7 | Mild | 3.1 - 10.0 | True | True |
| T17 | 5-9 | false | 6 | Mild | 2.6 - 9.6 | True | True |
| T18 | 5-9 | false | 6 | Mild | 2.8 - 9.7 | True | True |
| T19 | 5-9 | false | 7 | Mild | 3.1 - 10.1 | True | True |
| T20 | 5-9 | false | 6 | Mild | 2.6 - 9.5 | True | True |
| T21 | 5-9 | false | 6 | Mild | 3.0 - 9.9 | True | True |
| T22 | 5-9 | false | 5 | Mild | 1.1 - 8.0 | True | True |
| T23 | 5-9 | false | 5 | Mild | 1.1 - 8.0 | True | True |
| T24 | 5-9 | false | 6 | Mild | 2.7 - 9.6 | True | True |
| T25 | 5-9 | false | 5 | Mild | 1.1 - 8.0 | True | True |
| T26 | 5-9 | false | 4 | Minimal / None | 0.9 - 7.8 | False | True |
| T27 | 10-14 | true | 12 | Moderate | 9.5 - 14.0 | True | True |
| T28 | 10-14 | true | 12 | Moderate | 9.8 - 14.2 | True | True |
| T29 | 10-14 | true | 12 | Moderate | 10.2 - 14.6 | True | True |
| T30 | 10-14 | true | 12 | Moderate | 9.6 - 14.0 | True | True |
| T31 | 10-14 | true | 15 | Moderately Severe | 13.0 - 17.4 | False | True |
| T32 | 10-14 | true | 12 | Moderate | 9.7 - 14.1 | True | True |
| T33 | 10-14 | true | 12 | Moderate | 9.6 - 14.0 | True | True |
| T34 | 10-14 | true | 12 | Moderate | 10.1 - 14.5 | True | True |
| T35 | 10-14 | true | 9 | Mild | 6.8 - 11.2 | False | True |
| T36 | 10-14 | true | 12 | Moderate | 10.1 - 14.5 | True | True |
| T37 | 10-14 | true | 12 | Moderate | 10.1 - 14.5 | True | True |
| T38 | 10-14 | true | 15 | Moderately Severe | 12.6 - 17.0 | False | True |
| T39 | 15-24 | true | 19 | Moderately Severe | 16.8 - 21.3 | True | True |
| T40 | 15-24 | true | 18 | Moderately Severe | 16.0 - 20.4 | True | True |
| T41 | 15-24 | true | 18 | Moderately Severe | 15.3 - 19.8 | True | True |
| T42 | 15-24 | true | 19 | Moderately Severe | 16.6 - 21.0 | True | True |
| T43 | 15-24 | true | 19 | Moderately Severe | 16.3 - 20.7 | True | True |
| T44 | 15-24 | true | 19 | Moderately Severe | 17.0 - 21.4 | True | True |
| T45 | 15-24 | true | 18 | Moderately Severe | 15.3 - 19.8 | True | True |
| T46 | 15-24 | true | 18 | Moderately Severe | 15.4 - 19.8 | True | True |
| T47 | 15-24 | true | 17 | Moderately Severe | 15.2 - 19.7 | True | True |
| T48 | 15-24 | true | 17 | Moderately Severe | 15.2 - 19.6 | True | True |
| T49 | 15-24 | true | 19 | Moderately Severe | 17.2 - 21.7 | True | True |
| T50 | 15-24 | true | 19 | Moderately Severe | 17.1 - 21.6 | True | True |

Important:
- These are synthetic engineering tests only.
- They are not medical or clinical validation.
- They only check if the scoring pipeline produces a valid PHQ-8-like output and range.
