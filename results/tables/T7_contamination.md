| panel                | metric        |   real_clean |   real_contaminated |   pair_flips |   pairs | example                          |
|:---------------------|:--------------|-------------:|--------------------:|-------------:|--------:|:---------------------------------|
| one bad tick (-76%)  | std_ratio     |    1         |           1         |            2 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| one bad tick (-76%)  | kurt_med      |    6.081     |           6.081     |            0 |      21 |                                  |
| one bad tick (-76%)  | acf1_ret      |   -3.444e-05 |           0.000576  |            5 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| one bad tick (-76%)  | acf1_sq       |    0.1419    |           0.1376    |            0 |      21 |                                  |
| one bad tick (-76%)  | acf5_sq_mean  |    0.112     |           0.1097    |            0 |      21 |                                  |
| one bad tick (-76%)  | skew_ew       |   -1.194     |          -1.196     |            1 |      21 | DCC-GARCH-t vs i.i.d. Gaussian   |
| one bad tick (-76%)  | corr_fro      |    2.594     |           2.614     |            1 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| one bad tick (-76%)  | corr_mae      |    0.05025   |           0.04999   |            1 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| one bad tick (-76%)  | eig_dist      |    0.3329    |           0.3332    |            2 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| one bad tick (-76%)  | pc1_share     |    0.2923    |           0.2911    |            1 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| one bad tick (-76%)  | crisis_freq   |    0.1501    |           0.1501    |            3 |      21 | Gaussian HMM vs Regime Gaussian  |
| one bad tick (-76%)  | boost_pct     |   29.33      |          30.86      |            1 |      21 | i.i.d. Gaussian vs Static t-PPCA |
| one bad tick (-76%)  | tail_coexc    |    0.2055    |           0.2056    |            2 |      21 | GARCH-t FHS vs Static t-PPCA     |
| one bad tick (-76%)  | cvar_ratio    |    2.2       |           2.267     |            0 |      21 |                                  |
| one bad tick (-76%)  | pc1_share_gap |    0.07377   |           0.0776    |            1 |      21 | i.i.d. Gaussian vs Static t-PPCA |
| one bad tick (-76%)  | connect_gap   |    0.1136    |           0.1195    |            0 |      21 |                                  |
| one bad tick (-76%)  | disc_feat     |    0.5764    |           0.5826    |            3 |      21 | Gaussian HMM vs Regime Gaussian  |
| one bad tick (-76%)  | disc_raw      |    0.499     |           0.4881    |            1 |      21 | i.i.d. Gaussian vs Static t-PPCA |
| 1% bad ticks         | std_ratio     |    1         |           1         |            5 |      21 | Gaussian HMM vs i.i.d. Gaussian  |
| 1% bad ticks         | kurt_med      |    6.081     |          78.75      |            0 |      21 |                                  |
| 1% bad ticks         | acf1_ret      |   -3.444e-05 |           0.0004001 |            8 |      21 | DCC-GARCH-t vs Gaussian HMM      |
| 1% bad ticks         | acf1_sq       |    0.1419    |           0.005038  |            4 |      21 | Gaussian HMM vs Regime Gaussian  |
| 1% bad ticks         | acf5_sq_mean  |    0.112     |           0.001349  |            8 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| 1% bad ticks         | skew_ew       |   -1.194     |          -0.6911    |            2 |      21 | DCC-GARCH-t vs i.i.d. Gaussian   |
| 1% bad ticks         | corr_fro      |    2.594     |           1.056     |            4 |      21 | DCC-GARCH-t vs Static t-PPCA     |
| 1% bad ticks         | corr_mae      |    0.05025   |           0.01711   |            7 |      21 | DCC-GARCH-t vs Static t-PPCA     |
| 1% bad ticks         | eig_dist      |    0.3329    |           0.1015    |            3 |      21 | DCC-GARCH-t vs Static t-PPCA     |
| 1% bad ticks         | pc1_share     |    0.2923    |           0.06366   |            5 |      21 | DCC-GARCH-t vs Static t-PPCA     |
| 1% bad ticks         | crisis_freq   |    0.1501    |           0.1501    |           12 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| 1% bad ticks         | boost_pct     |   29.33      |          11.68      |            7 |      21 | DCC-GARCH-t vs Gaussian HMM      |
| 1% bad ticks         | tail_coexc    |    0.2055    |           0.1903    |           11 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| 1% bad ticks         | cvar_ratio    |    2.2       |           1.834     |            7 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| 1% bad ticks         | pc1_share_gap |    0.07377   |           0.02241   |            6 |      21 | DCC-GARCH-t vs Gaussian HMM      |
| 1% bad ticks         | connect_gap   |    0.1136    |           0.02145   |            4 |      21 | DCC-GARCH-t vs Gaussian HMM      |
| 1% bad ticks         | disc_feat     |    0.5764    |           0.6255    |            1 |      21 | Gaussian HMM vs Static t-PPCA    |
| 1% bad ticks         | disc_raw      |    0.499     |           0.5011    |           11 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| one unadjusted split | std_ratio     |    1         |           1         |            1 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| one unadjusted split | kurt_med      |    6.081     |           6.081     |            0 |      21 |                                  |
| one unadjusted split | acf1_ret      |   -3.444e-05 |           7.925e-05 |            3 |      21 | DCC-GARCH-t vs i.i.d. Gaussian   |
| one unadjusted split | acf1_sq       |    0.1419    |           0.1385    |            0 |      21 |                                  |
| one unadjusted split | acf5_sq_mean  |    0.112     |           0.1082    |            0 |      21 |                                  |
| one unadjusted split | skew_ew       |   -1.194     |          -1.192     |            1 |      21 | DCC-GARCH-t vs i.i.d. Gaussian   |
| one unadjusted split | corr_fro      |    2.594     |           2.595     |            1 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| one unadjusted split | corr_mae      |    0.05025   |           0.04952   |            1 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| one unadjusted split | eig_dist      |    0.3329    |           0.3289    |            2 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| one unadjusted split | pc1_share     |    0.2923    |           0.2899    |            1 |      21 | DCC-GARCH-t vs GARCH-t FHS       |
| one unadjusted split | crisis_freq   |    0.1501    |           0.1501    |            3 |      21 | GARCH-t FHS vs i.i.d. Gaussian   |
| one unadjusted split | boost_pct     |   29.33      |          28.53      |            1 |      21 | i.i.d. Gaussian vs Static t-PPCA |
| one unadjusted split | tail_coexc    |    0.2055    |           0.2055    |            1 |      21 | GARCH-t FHS vs Static t-PPCA     |
| one unadjusted split | cvar_ratio    |    2.2       |           2.195     |            0 |      21 |                                  |
| one unadjusted split | pc1_share_gap |    0.07377   |           0.07148   |            1 |      21 | i.i.d. Gaussian vs Static t-PPCA |
| one unadjusted split | connect_gap   |    0.1136    |           0.1124    |            0 |      21 |                                  |
| one unadjusted split | disc_feat     |    0.5764    |           0.5691    |            2 |      21 | Gaussian HMM vs Regime Gaussian  |
| one unadjusted split | disc_raw      |    0.499     |           0.4912    |            0 |      21 |                                  |