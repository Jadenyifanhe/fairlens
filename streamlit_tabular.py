import streamlit as st
import pandas as pd
import src.fairlens as fl
from scipy.stats import skew

# Set page title
st.title('SynthEthic Tabular Pipeline')

# File uploader
file = st.file_uploader("Choose a CSV file", type="csv")
if file:
    df = pd.read_csv(file)
    if df is not None:
        st.write('### Preview of Dataset')
        st.dataframe(df.head())

        sensitive_attributes = ['Sex', 'Ethnicity', 'MaritalStatus']  # Checklist from the column names
        target_attributes = 'DecileScore'

        class Auditor:
            def __init__(self, df, sensitive_attributes, target_attribute):
                self.df = df
                self.sensitive_attributes = sorted(sensitive_attributes, key=lambda x: x[0])
                self.target_attribute = target_attribute
                self.fscorer = fl.FairnessScorer(
                    df,
                    target_attr=target_attribute,
                    sensitive_attrs=sensitive_attributes
                )
                self.fairness_df = []
                self.max_comb = len(self.sensitive_attributes)

            def get_fairness_df(self, max_comb=None):
                if not max_comb:
                    max_comb = self.max_comb
                if len(self.fairness_df) == 0:
                    score_df = self.fscorer.distribution_score(max_comb=max_comb, method='dist_to_rest', p_value=True)
                    score_df_ = score_df[score_df['P-Value'] < 0.05]
                    if len(score_df_) == 0:
                        self.fairness_df = score_df
                        return score_df
                    self.fairness_df = score_df_
                return self.fairness_df

            def split_column(self, df):
                df[self.sensitive_attributes] = df['Group'].str.split(',', expand=True, n=(self.max_comb - 1))
                return df

            def get_audit_report(self):
                # Everything that is printed must be part of audit dashboard
                for attr in self.sensitive_attributes:
                    st.subheader(f"Audit Report for Attribute: {attr}")
                    attr_fscorer = fl.FairnessScorer(
                        self.df,
                        target_attr=self.target_attribute,
                        sensitive_attrs=[attr]
                    )
                    st.write("Distances from distributions:")
                    score_df = attr_fscorer.distribution_score(max_comb=1, method='dist_to_rest', p_value=True)
                    score_df = score_df.sort_values(by='Distance', ascending=False)
                    skews = []
                    medians = []

                    for index, row_data in score_df.iterrows():
                        eq_df = self.df[self.df[attr] == row_data['Group'].strip()][self.target_attribute]
                        neq_df = self.df[self.df[attr] != row_data['Group'].strip()][self.target_attribute]

                        diff_median = neq_df.median() - eq_df.median()
                        skew_diff = skew(neq_df) - skew(eq_df)
                        if diff_median == 0:
                            medians.append(1)
                        else:
                            medians.append(diff_median)
                        skews.append(skew_diff)
                    score_df['medians'] = medians
                    score_df['skews'] = skews

                    distance_score = [medians[i] * score_df['Distance'][i] for i in range(len(medians))]
                    score_df['Bias_Magnitude'] = distance_score
                    st.dataframe(score_df[['Group', 'Bias_Magnitude']])

                    st.write("Underrepresented communities:")
                    props = len(self.df) / len(score_df) * 0.2
                    underep_df = score_df[score_df['Counts'] < props].sort_values(by='Counts', ascending=True)
                    if len(underep_df) == 0:
                        min_proportion_index = score_df['Counts'].idxmin()
                        underep_df = score_df.loc[min_proportion_index]
                    st.write(underep_df['Group'])

            def get_worst_combos(self):
                fair_df = self.get_fairness_df()
                filtered_df = fair_df[fair_df['Group'].str.count(',') >= (self.max_comb - 1)]
                return self.split_column(filtered_df)

        # Instantiate the Auditor class and display results
        a = Auditor(df, sensitive_attributes, target_attributes)
        a_df = a.get_worst_combos().sort_values(by='P-Value', ascending=True)
        st.subheader('Worst Combination of Sensitive Attributes')
        st.dataframe(a_df)

        st.subheader('Audit Report')
        a.get_audit_report()
else:
    st.info('Please upload a CSV file to start the analysis.')
