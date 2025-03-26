import pandas as pd
import json
import numpy as np

col_info = {'education': ['educational_institution_name',
            'degree_names',
            'passing_years',
            'educational_results',
            'result_types',
            'major_field_of_studies'],
            
'experience': ['professional_company_names',
            'company_urls',
            'start_dates',
            'end_dates',
            'related_skils_in_job',
            'positions',
            'locations',
            'responsibilities'],

'extracurriculars': ['extra_curricular_activity_types',
            'extra_curricular_organization_names',
            'extra_curricular_organization_links',
            'role_positions'],
            'languages': ['languages',
                          'proficiency_levels'],
            'certifications': [
            'certification_providers',
            'certification_skills',
            'online_links',
            'issue_dates',
            'expiry_dates'
                               ]}

df = pd.read_csv('resume_data.csv')
col_to = {}
for d in col_info:
    for c in col_info[d]:
        col_to[c] = d
not_covered = []
for c in df.columns:
    if c not in col_to:
        col_to[c] = None
jli = []
for i in range(df.shape[0]):
    r = {}
    for c in df.columns:
        if col_to[c] is None:
            try:
                value = eval(df.loc[i, c])
            except:
                value = df.loc[i, c]
            try:
                if pd.isna(value):
                    r[c] = None
                else:
                    r[c] = value
            except ValueError:
                r[c] = value
        else:
            if col_to[c] not in r:
                r[col_to[c]] = []
            try:
                dlic = eval(df.loc[i, c])
            except:
                dlic = []
            for k in range(len(dlic)):
                while len(r[col_to[c]]) <= k:
                    r[col_to[c]].append({})
                value = dlic[k]
                try:
                    if pd.isna(value):
                        r[col_to[c]][k][c] = None
                    else:
                        r[col_to[c]][k][c] = value
                except ValueError:
                    r[col_to[c]][k][c] = value
    jli.append(r)
with open('resume_data.json', 'w') as f:
    json.dump(jli, f, indent=4)

df = pd.read_csv('job_data_merged_1.csv')
jli = []
for i in range(df.shape[0]):
    r = {}
    for c in df.columns:
        if c != 'Unnamed: 0':
            try:
                if pd.isna(df.loc[i, c]):
                    r[c] = None
                else:
                    r[c] = df.loc[i, c]
            except ValueError:
                r[c] = df.loc[i, c]
    jli.append(r)
with open('job_data_merged_1.json', 'w') as f:
    json.dump(jli, f, indent=4)

eval_cols = ['Benefits', 'Company Profile']
int_cols = ['Job ID']
df = pd.read_csv('job_descriptions.csv')
jli = []
for i in range(df.shape[0]):
    r = {}
    for c in df.columns:
        if c == 'Unnamed: 0':
            continue
        try:
            if c in eval_cols:
                value = eval(df.loc[i, c])
            else:
                value = df.loc[i, c]
            try:
                if pd.isna(value):
                    value = None
            except ValueError:
                pass
            if isinstance(value, np.int64):
                value = int(value)
            elif isinstance(value, np.float64):
                value = float(value)
            elif type(value) is set:
                value = list(value)
            r[c] = value
        except TypeError:
            pass
        except SyntaxError:
            print('Syntax error!')
    jli.append(r)
with open('job_descriptions.json', 'w') as f:
    json.dump(jli, f, indent=4)
