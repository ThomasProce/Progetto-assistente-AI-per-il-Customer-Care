# Risultati della valutazione

Conversazioni valutate: 23

## Metriche aggregate

| Metrica | Media | N |
|---|---|---|
| ragas_faithfulness | 0.665 | 17 |
| ragas_answer_relevancy | 0.555 | 21 |
| ragas_context_recall | 0.875 | 12 |
| deepeval_correctness | 0.826 | 23 |
| deepeval_tool_correctness | 0.833 | 23 |
| needs_human_correct | 1.000 | 23 |
| source_hit | 0.692 | 13 |
| retrieval_hit | 0.769 | 13 |
| sources_grounded | 1.000 | 23 |
| schema_valid | 1.000 | 23 |

## Per categoria

|                   |   ragas_faithfulness |   ragas_answer_relevancy |   ragas_context_recall |   deepeval_correctness |   deepeval_tool_correctness |   needs_human_correct |   source_hit |   retrieval_hit |   sources_grounded |   schema_valid |
|:------------------|---------------------:|-------------------------:|-----------------------:|-----------------------:|----------------------------:|----------------------:|-------------:|----------------:|-------------------:|---------------:|
| cultivation_guide |                0.778 |                    0.663 |                  0.833 |                  0.878 |                       0.667 |                     1 |         1    |             1   |                  1 |              1 |
| edge_case         |                0.2   |                    0.438 |                nan     |                  0.882 |                       1     |                     1 |       nan    |           nan   |                  1 |              1 |
| escalation        |                0.458 |                    0.516 |                  0.75  |                  0.715 |                       0.861 |                     1 |         0.25 |             0.5 |                  1 |              1 |
| multi_tool        |                1     |                    0.661 |                  1     |                  0.857 |                       1     |                     1 |         1    |             1   |                  1 |              1 |
| multi_turn_memory |              nan     |                    0.503 |                nan     |                  0.812 |                       0     |                     1 |         0    |             0   |                  1 |              1 |
| order_status      |                0.812 |                    0.565 |                nan     |                  0.883 |                       1     |                     1 |       nan    |           nan   |                  1 |              1 |
| out_of_scope      |              nan     |                  nan     |                nan     |                  0.87  |                       1     |                     1 |       nan    |           nan   |                  1 |              1 |
| policy            |                0.833 |                    0.581 |                  1     |                  0.822 |                       1     |                     1 |         1    |             1   |                  1 |              1 |
| product_info      |                0.9   |                    0.722 |                  1     |                  0.98  |                       1     |                     1 |         1    |             1   |                  1 |              1 |

## Per conversazione

| id                         | category          | confidence   | needs_human   | expected_needs_human   |   ragas_faithfulness |   ragas_answer_relevancy |   ragas_context_recall |   deepeval_correctness |   deepeval_tool_correctness |   needs_human_correct |   source_hit |   retrieval_hit |   sources_grounded |   schema_valid |
|:---------------------------|:------------------|:-------------|:--------------|:-----------------------|---------------------:|-------------------------:|-----------------------:|-----------------------:|----------------------------:|----------------------:|-------------:|----------------:|-------------------:|---------------:|
| c01_brief_example          | multi_tool        | high         | False         | False                  |                 1    |                     0.66 |                   1    |                   0.86 |                        1    |                     1 |            1 |               1 |                  1 |              1 |
| c02_order_in_preparation   | order_status      | high         | False         | False                  |                 0.91 |                     0.67 |                 nan    |                   0.88 |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c03_refund_status          | order_status      | high         | False         | False                  |                 0.71 |                     0.46 |                 nan    |                   0.88 |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c04_product_question       | product_info      | medium       | False         | False                  |                 0.9  |                     0.72 |                   1    |                   0.98 |                        1    |                     1 |            1 |               1 |                  1 |              1 |
| c05_basil_flowering        | cultivation_guide | high         | False         | False                  |                 1    |                     0.64 |                   1    |                   0.91 |                        1    |                     1 |            1 |               1 |                  1 |              1 |
| c06_return_live_plant      | policy            | medium       | False         | False                  |                 1    |                     0.6  |                   1    |                   0.9  |                        1    |                     1 |            1 |               1 |                  1 |              1 |
| c07_shipping_costs         | policy            | high         | False         | False                  |                 0.5  |                     0.57 |                   1    |                   0.68 |                        1    |                     1 |            1 |               1 |                  1 |              1 |
| c08_damaged_plants         | escalation        | low          | True          | True                   |                 0.5  |                     0.45 |                   0.67 |                   0.67 |                        0.67 |                     1 |            0 |               0 |                  1 |              1 |
| c09_explicit_human_request | escalation        | low          | True          | True                   |                 0.5  |                     0.28 |                 nan    |                   0.79 |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c10_delayed_delivery       | escalation        | low          | True          | True                   |                 0.4  |                     0.64 |                 nan    |                   0.79 |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c11_nonexistent_order      | edge_case         | medium       | False         | False                  |               nan    |                     0.71 |                 nan    |                   0.84 |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c12_ambiguous_order        | edge_case         | medium       | False         | False                  |               nan    |                     0.61 |                 nan    |                   0.9  |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c13_out_of_scope_sport     | out_of_scope      | high         | False         | False                  |               nan    |                   nan    |                 nan    |                   0.86 |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c14_prompt_injection       | out_of_scope      | high         | False         | False                  |               nan    |                   nan    |                 nan    |                   0.88 |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c15_cancel_order           | escalation        | low          | True          | True                   |                 0.25 |                     0.73 |                   0.33 |                   0.7  |                        0.5  |                     1 |            0 |               0 |                  1 |              1 |
| c16_memory_order_number    | multi_turn_memory | medium       | False         | False                  |               nan    |                     0.5  |                 nan    |                   0.74 |                        0    |                     1 |          nan |             nan |                  1 |              1 |
| c17_memory_tomato          | multi_turn_memory | medium       | False         | False                  |               nan    |                     0.51 |                 nan    |                   0.89 |                        0    |                     1 |            0 |               0 |                  1 |              1 |
| c18_loyalty_points         | policy            | high         | False         | False                  |                 1    |                     0.57 |                   1    |                   0.88 |                        1    |                     1 |            1 |               1 |                  1 |              1 |
| c19_product_not_in_catalog | edge_case         | low          | True          | True                   |                 0.2  |                     0    |                 nan    |                   0.9  |                        1    |                     1 |          nan |             nan |                  1 |              1 |
| c20_double_charge          | escalation        | low          | True          | True                   |                 0.6  |                     0.55 |                   1    |                   0.68 |                        1    |                     1 |            0 |               1 |                  1 |              1 |
| c21_lemon_chlorosis        | cultivation_guide | high         | False         | False                  |                 1    |                     0.71 |                   1    |                   0.85 |                        1    |                     1 |            1 |               1 |                  1 |              1 |
| c22_dahlia_trap            | cultivation_guide | high         | False         | False                  |                 0.33 |                     0.64 |                   0.5  |                   0.86 |                        0    |                     1 |            1 |               1 |                  1 |              1 |
| c23_late_damage_claim      | escalation        | low          | True          | True                   |                 0.5  |                     0.45 |                   1    |                   0.65 |                        1    |                     1 |            1 |               1 |                  1 |              1 |
