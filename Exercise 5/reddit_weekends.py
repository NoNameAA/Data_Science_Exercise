import pandas as pd
import numpy as np
import sys
from datetime import date
import scipy.stats as ss
import matplotlib.pyplot as plt


OUTPUT_TEMPLATE = (
    "Initial (invalid) T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann–Whitney U-test p-value: {utest_p:.3g}"
)


def main():
    reddit_counts = sys.argv[1]

    input_filename = sys.argv[1]
    counts = pd.read_json(sys.argv[1], lines=True)
    counts = counts[counts['date'].between('2012-01-01', '2013-12-31')]
    counts = counts[counts['subreddit'] == 'canada'].reset_index()
    counts = counts.drop(['index'], axis=1)
    counts['weekday'] = pd.Series(1, index=counts.index)
    counts['weekday'] = counts['date'].apply(lambda x: x.weekday())

    weekday_data = counts[counts['weekday'] < 5]
    weekend_data = counts[counts['weekday'] >= 5]

    weekday_data = weekday_data.reset_index().drop(['index'], axis=1)
    weekend_data = weekend_data.reset_index().drop(['index'], axis=1)

    (_, p_value) = ss.ttest_ind(weekday_data['comment_count'], weekend_data['comment_count'])
    (_, weekday_data_normal_p_value) = ss.normaltest(weekday_data['comment_count'])
    (_, weekend_data_normal_p_value) = ss.normaltest(weekend_data['comment_count'])
    (_, levene_p_value) = ss.levene(weekday_data['comment_count'], weekend_data['comment_count'])

    weekly_weekday = weekday_data.copy()
    weekly_weekend = weekend_data.copy()

    # Fix 1

    weekday_data['comment_count_fix1'] = weekday_data['comment_count']
    weekend_data['comment_count_fix1'] = weekend_data['comment_count']
    weekday_data['comment_count_fix1'] = weekday_data['comment_count_fix1'].apply(lambda x: np.sqrt(x))
    weekend_data['comment_count_fix1'] = weekend_data['comment_count_fix1'].apply(lambda x: np.sqrt(x))

    (_, weekday_data_normal_p_value_fix1) = ss.normaltest(weekday_data['comment_count_fix1'])
    (_, weekend_data_normal_p_value_fix1) = ss.normaltest(weekend_data['comment_count_fix1'])
    (_, levene_p_value_fix1) = ss.levene(weekday_data['comment_count_fix1'], weekend_data['comment_count_fix1'])

    # Fix 2

    weekly_weekday['week'] = pd.Series(1, index=weekly_weekday.index)
    weekly_weekday['year'] = pd.Series(1, index=weekly_weekday.index)
    weekly_weekend['week'] = pd.Series(1, index=weekly_weekend.index)
    weekly_weekend['year'] = pd.Series(1, index=weekly_weekend.index)
    weekly_weekday['year'] = weekly_weekday['date'].apply(lambda x: x.isocalendar()[0])
    weekly_weekday['week'] = weekly_weekday['date'].apply(lambda x: x.isocalendar()[1])
    weekly_weekend['year'] = weekly_weekend['date'].apply(lambda x: x.isocalendar()[0])
    weekly_weekend['week'] = weekly_weekend['date'].apply(lambda x: x.isocalendar()[1])

    weekly_weekday = weekly_weekday.groupby(['week', 'year'])['comment_count'].mean().reset_index()
    weekly_weekend = weekly_weekend.groupby(['week', 'year'])['comment_count'].mean().reset_index()

    (_, weekly_weekday_normal_p_value) = ss.normaltest(weekly_weekday['comment_count'])
    (_, weekly_weekend_normal_p_value) = ss.normaltest(weekly_weekend['comment_count'])
    (_, weekly_levene_p_value) = ss.levene(weekly_weekday['comment_count'], weekly_weekend['comment_count'])
    (_, weekly_p_value) = ss.ttest_ind(weekly_weekday['comment_count'], weekly_weekend['comment_count'])

    # Fix 3

    (_, u_test_p_value) = ss.mannwhitneyu(weekday_data['comment_count'], weekend_data['comment_count'])

    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=p_value,
        initial_weekday_normality_p=weekday_data_normal_p_value,
        initial_weekend_normality_p=weekend_data_normal_p_value,
        initial_levene_p=levene_p_value,
        transformed_weekday_normality_p=weekday_data_normal_p_value_fix1,
        transformed_weekend_normality_p=weekend_data_normal_p_value_fix1,
        transformed_levene_p=levene_p_value_fix1,
        weekly_weekday_normality_p=weekly_weekday_normal_p_value,
        weekly_weekend_normality_p=weekly_weekend_normal_p_value,
        weekly_levene_p=weekly_levene_p_value,
        weekly_ttest_p=weekly_p_value,
        utest_p=u_test_p_value,
    ))


if __name__ == '__main__':
    main()