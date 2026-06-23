# 💳 Intelligent Credit Risk Assessment Portal

---

## 👤 Intern Profile & Project Metadata
| Metric | Details |
| :--- | :--- |
| **Intern ID** | CITS3008 |
| **Full Name** | Bhavisha M |
| **Domain** | Artificial Intelligence & Machine Learning |
| **Internship Duration** | 4 Weeks |
| **Project Assignment** | Task 4 — Financial Credit Risk Assessment Framework |

---

## 🧠 Project Architecture & Engine Workflow
This system acts as an automated bank underwriting assistant. It leverages an advanced predictive engine to analyze borrower characteristics and calculate default probabilities dynamically.

```mermaid
graph LR
A[Applicant Parameters] --> B[Random Forest Classifier]
B --> C{Risk Threshold Check}
C -- >50% Default Prob --> D[🚨 Verdict: HIGH RISK / REJECT]
C -- <50% Default Prob --> E[🎯 Verdict: LOW RISK / APPROVE]
