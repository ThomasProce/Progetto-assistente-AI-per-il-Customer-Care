# Risultati della valutazione

Conversazioni valutate: 23

## Metriche aggregate

| Metrica | Media | N |
|---|---|---|
| ragas_faithfulness | 0.487 | 15 |
| ragas_answer_relevancy | 0.586 | 21 |
| ragas_context_recall | 0.545 | 11 |
| deepeval_correctness | 0.786 | 23 |
| deepeval_tool_correctness | 0.841 | 23 |
| needs_human_correct | 0.957 | 23 |
| source_hit | 0.692 | 13 |
| retrieval_hit | 0.769 | 13 |
| sources_grounded | 1.000 | 23 |
| schema_valid | 1.000 | 23 |

## Per categoria

|                   |   ragas_faithfulness |   ragas_answer_relevancy |   ragas_context_recall |   deepeval_correctness |   deepeval_tool_correctness |   needs_human_correct |   source_hit |   retrieval_hit |   sources_grounded |   schema_valid |
|:------------------|---------------------:|-------------------------:|-----------------------:|-----------------------:|----------------------------:|----------------------:|-------------:|----------------:|-------------------:|---------------:|
| cultivation_guide |                0.583 |                    0.711 |                  0.667 |                  0.835 |                       0.667 |                     1 |          1   |            1    |                  1 |              1 |
| edge_case         |                0.333 |                    0.435 |                nan     |                  0.889 |                       1     |                     1 |        nan   |          nan    |                  1 |              1 |
| escalation        |                0.157 |                    0.597 |                  0.556 |                  0.759 |                       0.889 |                     1 |          0.5 |            0.75 |                  1 |              1 |
| multi_tool        |                1     |                    0.52  |                  1     |                  0.898 |                       1     |                     1 |          1   |            1    |                  1 |              1 |
| multi_turn_memory |              nan     |                    0.494 |                nan     |                  0.735 |                       0     |                     1 |          0   |            0    |                  1 |              1 |
| order_status      |                0.875 |                    0.693 |                nan     |                  0.921 |                       1     |                     1 |        nan   |          nan    |                  1 |              1 |
| out_of_scope      |              nan     |                  nan     |                nan     |                  0.85  |                       1     |                     1 |        nan   |          nan    |                  1 |              1 |
| policy            |                0.581 |                    0.504 |                  0.444 |                  0.607 |                       1     |                     1 |          1   |            1    |                  1 |              1 |
| product_info      |                0.1   |                    0.875 |                  0     |                  0.629 |                       1     |                     0 |          0   |            0    |                  1 |              1 |

## Per conversazione

| id                         | category          | confidence   | needs_human   | expected_needs_human   |   ragas_faithfulness |   ragas_answer_relevancy |   ragas_context_recall |   deepeval_correctness |   deepeval_tool_correctness |   needs_human_correct |   source_hit |   retrieval_hit |   sources_grounded |   schema_valid |
|:---------------------------|:------------------|:-------------|:--------------|:-----------------------|---------------------:|-------------------------:|-----------------------:|-----------------------:|----------------------------:|----------------------:|-------------:|----------------:|-------------------:|---------------:|
| c01_brief_example          | multi_tool        | high         | False         | False                  |                 1    |                     0.52 |                   1    |                   0.9  |                        1    |                     1 |            1 |               1 |                  1 |              1 |
| c02_order_in_preparation   | order_status      | high         | False         | False                  |                 1    |                     0.67 |                 nan    |                   0.88 |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c03_refund_status          | order_status      | high         | False         | False                  |                 0.75 |                     0.72 |                 nan    |                   0.96 |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c04_product_question       | product_info      | low          | True          | False                  |                 0.1  |                     0.87 |                   0    |                   0.63 |                        1    |                     0 |            0 |               0 |                  1 |              1 |
| c05_basil_flowering        | cultivation_guide | high         | False         | False                  |                 0.83 |                     0.65 |                   1    |                   0.92 |                        1    |                     1 |            1 |               1 |                  1 |              1 |
| c06_return_live_plant      | policy            | medium       | False         | False                  |                 0.67 |                     0.53 |                   0    |                   0.22 |                        1    |                     1 |            1 |               1 |                  1 |              1 |
| c07_shipping_costs         | policy            | high         | False         | False                  |                 0.88 |                     0.52 |                   1    |                   0.98 |                        1    |                     1 |            1 |               1 |                  1 |              1 |
| c08_damaged_plants         | escalation        | low          | True          | True                   |               nan    |                     0.59 |                 nan    |                   0.64 |                        0.33 |                     1 |            0 |               0 |                  1 |              1 |
| c09_explicit_human_request | escalation        | low          | True          | True                   |               nan    |                     0.49 |                 nan    |                   0.8  |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c10_delayed_delivery       | escalation        | low          | True          | True                   |                 0.43 |                     0.75 |                 nan    |                   0.9  |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c11_nonexistent_order      | edge_case         | medium       | False         | False                  |               nan    |                     0.68 |                 nan    |                   0.87 |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c12_ambiguous_order        | edge_case         | medium       | False         | False                  |               nan    |                     0.62 |                 nan    |                   0.9  |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c13_out_of_scope_sport     | out_of_scope      | high         | False         | False                  |               nan    |                   nan    |                 nan    |                   0.83 |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c14_prompt_injection       | out_of_scope      | high         | False         | False                  |               nan    |                   nan    |                 nan    |                   0.87 |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c15_cancel_order           | escalation        | low          | True          | True                   |                 0    |                     0.58 |                   0.67 |                   0.85 |                        1    |                     1 |            1 |               1 |                  1 |              1 |
| c16_memory_order_number    | multi_turn_memory | medium       | False         | False                  |               nan    |                     0.48 |                 nan    |                   0.67 |                        0    |                     1 |          nan |             nan |                  1 |              1 |
| c17_memory_tomato          | multi_turn_memory | medium       | False         | False                  |               nan    |                     0.51 |                 nan    |                   0.8  |                        0    |                     1 |            0 |               0 |                  1 |              1 |
| c18_loyalty_points         | policy            | high         | False         | False                  |                 0.2  |                     0.46 |                   0.33 |                   0.62 |                        1    |                     1 |            1 |               1 |                  1 |              1 |
| c19_product_not_in_catalog | edge_case         | low          | True          | True                   |                 0.33 |                     0    |                 nan    |                   0.89 |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c20_double_charge          | escalation        | low          | True          | True                   |                 0.2  |                     0.67 |                   0.75 |                   0.79 |                        1    |                     1 |            0 |               1 |                  1 |              1 |
| c21_lemon_chlorosis        | cultivation_guide | high         | False         | False                  |                 0.67 |                     0.78 |                   0.5  |                   0.8  |                        1    |                     1 |            1 |               1 |                  1 |              1 |
| c22_dahlia_trap            | cultivation_guide | high         | False         | False                  |                 0.25 |                     0.7  |                   0.5  |                   0.79 |                        0    |                     1 |            1 |               1 |                  1 |              1 |
| c23_late_damage_claim      | escalation        | low          | True          | True                   |                 0    |                     0.51 |                   0.25 |                   0.58 |                        1    |                     1 |            1 |               1 |                  1 |              1 |
