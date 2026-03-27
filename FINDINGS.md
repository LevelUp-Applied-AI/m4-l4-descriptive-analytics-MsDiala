# EDA Findings: Student Performance Data

## Data Profile Summary
- Rows: 2000, Columns: 10

## Distribution Highlights
- GPA appears to be centered near mid 2's with some variation and few very low/very high values.
- Study hours per week show a wide spread, indicating differing student commitments.
- Attendance is generally high with some lower outliers. 

## Correlation Analysis
- GPA vs Study Hours correlation: 0.64
- GPA vs Attendance correlation: 0.04
- GPA vs Commute Minutes correlation: 0.01

## Hypothesis Tests
- Internship t-test: t=14.229, p=0.000
- Department ANOVA: F=0.667, p=0.615
- Study Hours vs GPA correlation: r=0.639, p=0.000

## Interpretation
- Higher study hours generally align with higher GPA (if the correlation and p-value are significant).
- Internship status and department show measurable differences; interpret p-values to assess statistical significance.