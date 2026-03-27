"""Lab 4 — Descriptive Analytics: Student Performance EDA

Conduct exploratory data analysis on the student performance dataset.
Produce distribution plots, correlation analysis, hypothesis tests,
and a written findings report.

Usage:
    python eda_analysis.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_and_profile(filepath):
    """Load the dataset and generate a data profile report.

    Args:
        filepath: path to the CSV file (e.g., 'data/student_performance.csv')

    Returns:
        DataFrame: the loaded dataset

    Side effects:
        Saves a text profile to output/data_profile.txt containing:
        - Shape (rows, columns)
        - Data types for each column
        - Missing value counts per column
        - Descriptive statistics for numeric columns
    """
    df = pd.read_csv(filepath)

    report_lines = [
        "Data Profile Report",
        "===================",
        f"Dataset: {filepath}",
        f"Shape: {df.shape[0]} rows, {df.shape[1]} columns",
        "",
        "Data types:",
    ]
    report_lines += [f"{col}: {dtype}" for col, dtype in df.dtypes.items()]

    report_lines += ["", "Missing values per column:"]
    report_lines += [f"{col}: {cnt}" for col, cnt in df.isna().sum().items()]

    report_lines += ["", "Descriptive statistics (numeric columns):", df.describe().to_string()]

    os.makedirs("output", exist_ok=True)
    with open(os.path.join("output", "data_profile.txt"), "w") as fh:
        fh.write("\n".join(report_lines))

    return df


def plot_distributions(df):
    """Create distribution plots for key numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least 3 distribution plots (histograms with KDE or box plots)
        as PNG files in the output/ directory. Each plot should have a
        descriptive title that states what the distribution reveals.
    """
    numeric_cols = ["gpa", "study_hours_weekly", "attendance_pct", "commute_minutes"]
    for col in numeric_cols:
        if col not in df.columns:
            continue

        values = df[col].dropna()
        if values.empty:
            continue

        hist_path = os.path.join("output", f"{col}_distribution.png")
        plt.figure(figsize=(8, 5))
        sns.histplot(values, kde=True, stat="density", bins=15, color="steelblue")
        plt.title(f"Distribution of {col.replace('_', ' ').title()}")
        plt.xlabel(col.replace('_', ' ').title())
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()

        box_path = os.path.join("output", f"{col}_boxplot.png")
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=values, color="lightgreen")
        plt.title(f"Boxplot of {col.replace('_', ' ').title()}")
        plt.xlabel(col.replace('_', ' ').title())
        plt.tight_layout()
        plt.savefig(box_path)
        plt.close()


def plot_correlations(df):
    """Analyze and visualize relationships between numeric variables.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        None

    Side effects:
        Saves at least one correlation visualization to the output/ directory
        (e.g., a heatmap, scatter plot, or pair plot).
    """
    numeric_cols = ["gpa", "study_hours_weekly", "attendance_pct", "commute_minutes"]
    subset = df[numeric_cols].dropna()
    if subset.empty:
        print("No numeric data available for correlations.")
        return

    corr = subset.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap for Key Numeric Variables")
    plt.tight_layout()
    plt.savefig(os.path.join("output", "correlation_heatmap.png"))
    plt.close()

    # Pairplot optional but useful
    pair_path = os.path.join("output", "pairplot.png")
    sns.pairplot(subset)
    plt.savefig(pair_path)
    plt.close()


def run_hypothesis_tests(df):
    """Run statistical tests to validate observed patterns.

    Args:
        df: pandas DataFrame with the student performance data

    Returns:
        dict: test results with keys like 'internship_ttest', 'dept_anova',
              each containing the test statistic and p-value

    Side effects:
        Prints test results to stdout with interpretation.

    Tests to consider:
        - t-test: Does GPA differ between students with and without internships?
        - ANOVA: Does GPA differ across departments?
        - Correlation test: Is the correlation between study hours and GPA significant?
    """
    results = {}

    if "has_internship" in df.columns and "gpa" in df.columns:
        gpa_yes = df.loc[df["has_internship"].str.lower() == "yes", "gpa"].dropna()
        gpa_no = df.loc[df["has_internship"].str.lower() == "no", "gpa"].dropna()
        if len(gpa_yes) >= 2 and len(gpa_no) >= 2:
            tstat, pval = stats.ttest_ind(gpa_yes, gpa_no, equal_var=False, nan_policy="omit")
            results["internship_ttest"] = {"statistic": float(tstat), "pvalue": float(pval)}
            print(f"Internship vs non-internship GPA t-test: t={tstat:.4f}, p={pval:.4f}")

    if "department" in df.columns and "gpa" in df.columns:
        groups = [group.dropna() for _, group in df.groupby("department")["gpa"] if len(group.dropna()) >= 2]
        if len(groups) >= 2:
            fstat, pval = stats.f_oneway(*groups)
            results["dept_anova"] = {"statistic": float(fstat), "pvalue": float(pval)}
            print(f"Department GPA ANOVA: F={fstat:.4f}, p={pval:.4f}")

    if "study_hours_weekly" in df.columns and "gpa" in df.columns:
        pair = df[["study_hours_weekly", "gpa"]].dropna()
        if len(pair) >= 2:
            corr_val, pval = stats.pearsonr(pair["study_hours_weekly"], pair["gpa"])
            results["study_gpa_corr"] = {"correlation": float(corr_val), "pvalue": float(pval)}
            print(f"Study hours vs GPA Pearson correlation: r={corr_val:.4f}, p={pval:.4f}")

    if not results:
        print("No valid hypothesis tests could be run due to missing or insufficient data.")

    return results


def main():
    """Orchestrate the full EDA pipeline."""
    os.makedirs("output", exist_ok=True)

    data_path = os.path.join("data", "student_performance.csv")
    df = load_and_profile(data_path)

    plot_distributions(df)
    plot_correlations(df)
    test_results = run_hypothesis_tests(df)

    findings_text = []
    findings_text.append("# EDA Findings: Student Performance Data")
    findings_text.append("")
    findings_text.append("## Data Profile Summary")
    findings_text.append(f"- Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    findings_text.append("")
    findings_text.append("## Distribution Highlights")
    findings_text.append("- GPA appears to be centered near mid 2's with some variation and few very low/very high values.")
    findings_text.append("- Study hours per week show a wide spread, indicating differing student commitments.")
    findings_text.append("- Attendance is generally high with some lower outliers. ")
    findings_text.append("")

    corr = df[["gpa", "study_hours_weekly", "attendance_pct", "commute_minutes"]].corr()
    findings_text.append("## Correlation Analysis")
    findings_text.append(f"- GPA vs Study Hours correlation: {corr.loc['gpa','study_hours_weekly']:.2f}")
    findings_text.append(f"- GPA vs Attendance correlation: {corr.loc['gpa','attendance_pct']:.2f}")
    findings_text.append(f"- GPA vs Commute Minutes correlation: {corr.loc['gpa','commute_minutes']:.2f}")
    findings_text.append("")

    findings_text.append("## Hypothesis Tests")
    if "internship_ttest" in test_results:
        findings_text.append(f"- Internship t-test: t={test_results['internship_ttest']['statistic']:.3f}, p={test_results['internship_ttest']['pvalue']:.3f}")
    if "dept_anova" in test_results:
        findings_text.append(f"- Department ANOVA: F={test_results['dept_anova']['statistic']:.3f}, p={test_results['dept_anova']['pvalue']:.3f}")
    if "study_gpa_corr" in test_results:
        findings_text.append(f"- Study Hours vs GPA correlation: r={test_results['study_gpa_corr']['correlation']:.3f}, p={test_results['study_gpa_corr']['pvalue']:.3f}")

    findings_text.append("")
    findings_text.append("## Interpretation")
    findings_text.append("- Higher study hours generally align with higher GPA (if the correlation and p-value are significant).")
    findings_text.append("- Internship status and department show measurable differences; interpret p-values to assess statistical significance.")

    with open("FINDINGS.md", "w") as fh:
        fh.write("\n".join(findings_text))


if __name__ == "__main__":
    main()
