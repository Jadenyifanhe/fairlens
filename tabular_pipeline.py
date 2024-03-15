import pandas as pd
import src.fairlens as fl
import matplotlib.pyplot as plt
from ctgan import CTGAN
from sdv.datasets.local import load_csvs
from sdv.sampling import Condition
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from scipy.stats import skew

df = pd.read_csv("datasets/compas.csv")
print(df.head())

sensitive_attributes = ['Sex', 'Ethnicity', 'MaritalStatus'] #Checklist from the column names
target_attributes = 'DecileScore'

class Auditor:
  def __init__(self, df, sensitive_attributes, target_attribute):
    self.df = df
    self.sensitive_attributes = sorted(sensitive_attributes, key=lambda x: x[0])
    self.target_attribute = target_attribute
    self.fscorer = fl.FairnessScorer(
        df,
        target_attr= target_attribute,
        sensitive_attrs=sensitive_attributes
      )
    self.fairness_df = []
    self.max_comb = len(self.sensitive_attributes)

  def get_fairness_df(self, max_comb = None):
    if not max_comb:
      max_comb = self.max_comb
    if len(self.fairness_df)==0:
      score_df = self.fscorer.distribution_score(max_comb=max_comb, method = 'dist_to_rest', p_value = True)
      score_df_ = score_df[score_df['P-Value']<0.05]
      if len(score_df_)==0:
        self.fairness_df = score_df
        return score_df
      #print(score_df)
      self.fairness_df = score_df_
    return self.fairness_df

  def split_column(self, df):
      #print(df['Group'])
      df[self.sensitive_attributes] = df['Group'].str.split(',', expand=True, n=(self.max_comb-1))
      return df

  def get_audit_report(self):
    #Everything that is printed must be part of audit dashboard
    for attr in self.sensitive_attributes:
      print("Attribute: ", attr)
      attr_fscorer = fl.FairnessScorer(
        df,
        target_attr= self.target_attribute,
        sensitive_attrs=[attr]
      )
      print("Distances from distributions: ")
      score_df = attr_fscorer.distribution_score(max_comb=1, method = 'dist_to_rest', p_value = True)
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

      distance_score = [medians[i]* score_df['Distance'][i] for i in range(len(medians))]
      score_df['Bias_Magnitude'] = distance_score
      score_df[['Group', 'Bias_Magnitude']].to_csv('Results'+attr+'.csv')
      print(score_df[['Group', 'Bias_Magnitude']])

      print("Underrepresented communities: ")
      props = len(self.df)/len(score_df)*0.2
      #print(props)
      underep_df = score_df[score_df['Counts'] < props].sort_values(by='Counts', ascending=True)
      if len(underep_df)==0:
         min_proportion_index = score_df['Counts'].idxmin()
         underep_df = score_df.loc[min_proportion_index]
         #print(underep_df)

      print(underep_df['Group'])

      #break


  def get_worst_combos(self):
    fair_df = self.get_fairness_df()
    filtered_df = fair_df[fair_df['Group'].str.count(',') >= (self.max_comb-1)]

    return self.split_column(filtered_df)

a = Auditor(df, sensitive_attributes, target_attributes)
a_df = a.get_worst_combos().sort_values(by='P-Value', ascending=True)
print(a_df)

print(a.get_audit_report())