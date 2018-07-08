import pandas as pd
import numpy as np
import sys
import scipy.stats as ss

OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value: {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value: {more_searches_p:.3g}\n'
    '"Did more/less instructors use the search feature?" p-value: {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value: {more_instr_searches_p:.3g}'
)


def main():
    filename = sys.argv[1]
    data = pd.read_json(filename, orient='records', lines='True')

    new_data = data[data['uid'] % 2 == 1].reset_index().drop(['index'], axis=1)
    old_data = data[data['uid'] % 2 == 0].reset_index().drop(['index'], axis=1)

    data['is_new'] = pd.Series(data['uid'] % 2 == 1, index=data.index)
    data['search_once'] = pd.Series(data['search_count'] > 0, index=data.index)

    table = pd.crosstab(data.is_new, data.search_once)
    _, more_users_p_value, _, _ = ss.chi2_contingency(table)
    _, more_searches_p_value = ss.mannwhitneyu(old_data.search_count, new_data.search_count)

    # print(data)
    ins_data = data[data['is_instructor'] == True].reset_index().drop(['index'], axis=1)
    ins_new_data = ins_data[ins_data['is_new'] == True].reset_index().drop(['index'], axis=1)
    ins_old_data = ins_data[ins_data['is_new'] == False].reset_index().drop(['index'], axis=1)

    ins_table = pd.crosstab(ins_data.is_new, ins_data.search_once)
    _, more_instr_p_value, _, _ = ss.chi2_contingency(ins_table)
    _, more_instr_searches_p_value = ss.mannwhitneyu(ins_old_data.search_count, ins_new_data.search_count)

    print(OUTPUT_TEMPLATE.format(
        more_users_p=more_users_p_value,
        more_searches_p=more_searches_p_value,
        more_instr_p=more_instr_p_value,
        more_instr_searches_p=more_instr_searches_p_value,
    ))


if __name__ == '__main__':
    main()












