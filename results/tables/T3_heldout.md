| episode    | episode_label           |   n_real_crisis |   real |    lo |     hi | generator            | name                        | family_tag        |     gen |   gen_sd | inside   |
|:-----------|:------------------------|----------------:|-------:|------:|-------:|:---------------------|:----------------------------|:------------------|--------:|---------:|:---------|
| covid_2020 | COVID crash, 2020       |              72 | 101.1  | 65.11 | 140.1  | iid_gaussian         | i.i.d. Gaussian             | static            |   2.262 |   0.5991 | False    |
| covid_2020 | COVID crash, 2020       |              72 | 101.1  | 65.11 | 140.1  | t_ppca               | Static t-PPCA               | static            |   2.512 |   1.431  | False    |
| covid_2020 | COVID crash, 2020       |              72 | 101.1  | 65.11 | 140.1  | stationary_bootstrap | Stationary bootstrap        | resampling        |   8.627 |   0.4632 | False    |
| covid_2020 | COVID crash, 2020       |              72 | 101.1  | 65.11 | 140.1  | garch_fhs            | GARCH-t FHS                 | dynamic-classical |  10.98  |   0.9141 | False    |
| covid_2020 | COVID crash, 2020       |              72 | 101.1  | 65.11 | 140.1  | dcc_garch            | DCC-GARCH-t                 | dynamic-classical |   8.477 |   1.172  | False    |
| covid_2020 | COVID crash, 2020       |              72 | 101.1  | 65.11 | 140.1  | regime_gaussian      | Regime Gaussian             | regime            |   5.888 |   0.7672 | False    |
| covid_2020 | COVID crash, 2020       |              72 | 101.1  | 65.11 | 140.1  | hmm_gaussian         | Gaussian HMM                | regime            |   3.355 |   0.9329 | False    |
| covid_2020 | COVID crash, 2020       |              72 | 101.1  | 65.11 | 140.1  | timevae              | TimeVAE                     | neural            | nan     | nan      | False    |
| covid_2020 | COVID crash, 2020       |              72 | 101.1  | 65.11 | 140.1  | quantgan             | QuantGAN                    | neural            |   4.087 |   0.8262 | False    |
| covid_2020 | COVID crash, 2020       |              72 | 101.1  | 65.11 | 140.1  | timegan              | TimeGAN                     | neural            |   1.248 |   1.431  | False    |
| covid_2020 | COVID crash, 2020       |              72 | 101.1  | 65.11 | 140.1  | cgnsde               | CG-NSDE v1 (preregistered)  | neural-ours       |   2.907 |   0.9104 | False    |
| covid_2020 | COVID crash, 2020       |              72 | 101.1  | 65.11 | 140.1  | cgnsde2              | CG-NSDE v2 (ours, post hoc) | neural-ours       |  22.15  |   2.076  | False    |
| shock_2022 | 2022 rate and war shock |              23 |  49.42 | 31.44 |  71.67 | iid_gaussian         | i.i.d. Gaussian             | static            |   2.262 |   0.5991 | False    |
| shock_2022 | 2022 rate and war shock |              23 |  49.42 | 31.44 |  71.67 | t_ppca               | Static t-PPCA               | static            |   2.512 |   1.431  | False    |
| shock_2022 | 2022 rate and war shock |              23 |  49.42 | 31.44 |  71.67 | stationary_bootstrap | Stationary bootstrap        | resampling        |   8.627 |   0.4632 | False    |
| shock_2022 | 2022 rate and war shock |              23 |  49.42 | 31.44 |  71.67 | garch_fhs            | GARCH-t FHS                 | dynamic-classical |  10.98  |   0.9141 | False    |
| shock_2022 | 2022 rate and war shock |              23 |  49.42 | 31.44 |  71.67 | dcc_garch            | DCC-GARCH-t                 | dynamic-classical |   8.477 |   1.172  | False    |
| shock_2022 | 2022 rate and war shock |              23 |  49.42 | 31.44 |  71.67 | regime_gaussian      | Regime Gaussian             | regime            |   5.888 |   0.7672 | False    |
| shock_2022 | 2022 rate and war shock |              23 |  49.42 | 31.44 |  71.67 | hmm_gaussian         | Gaussian HMM                | regime            |   3.355 |   0.9329 | False    |
| shock_2022 | 2022 rate and war shock |              23 |  49.42 | 31.44 |  71.67 | timevae              | TimeVAE                     | neural            | nan     | nan      | False    |
| shock_2022 | 2022 rate and war shock |              23 |  49.42 | 31.44 |  71.67 | quantgan             | QuantGAN                    | neural            |   4.087 |   0.8262 | False    |
| shock_2022 | 2022 rate and war shock |              23 |  49.42 | 31.44 |  71.67 | timegan              | TimeGAN                     | neural            |   1.248 |   1.431  | False    |
| shock_2022 | 2022 rate and war shock |              23 |  49.42 | 31.44 |  71.67 | cgnsde               | CG-NSDE v1 (preregistered)  | neural-ours       |   2.907 |   0.9104 | False    |
| shock_2022 | 2022 rate and war shock |              23 |  49.42 | 31.44 |  71.67 | cgnsde2              | CG-NSDE v2 (ours, post hoc) | neural-ours       |  22.15  |   2.076  | False    |