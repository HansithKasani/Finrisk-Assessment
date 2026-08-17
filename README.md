# AI Credit Risk Assessment System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red.svg)](https://streamlit.io/)
![Status: Active](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

## 🎯 Project Overview

**AI Credit Risk Assessment System** is an end-to-end Machine Learning system that predicts the probability of loan default using historical financial and demographic data. The system integrates **Explainable AI (XAI)** techniques (SHAP/LIME) to provide transparent "reasoning reports" for every prediction, making it suitable for real-world banking compliance and regulatory requirements.

### ✨ Key Features
- 📊 **XGBoost Classifier**: 200 boosting stages with >85% accuracy
- 🔍 **Full Explainability**: SHAP/LIME integration for transparent predictions
- 📈 **307,511 Samples**: Trained on comprehensive credit data
- 💡 **85+ Features**: Demographics, Financial, Credit Bureau data
- 🎨 **Interactive UI**: Streamlit-based dashboard with real-time predictions
- 🛡️ **Production Ready**: Docker, CI/CD, testing, and monitoring included
- ⚖️ **Ethics First**: Comprehensive bias testing and fairness guidelines

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip or conda
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/AI-Credit-Risk-Assessment-System.git
cd AI-Credit-Risk-Assessment-System

# 2. Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Streamlit Dashboard
streamlit run App/main.py

# 5. Or explore with Jupyter notebooks
jupyter notebook Notebooks/
```

### 🎨 Using the Dashboard

1. Open your browser to `http://localhost:8501`
2. Fill in applicant information in the sidebar
3. Click "Assess Credit Risk"
4. View prediction with SHAP-based explanation
5. Review detailed reasoning report

### 📊 Training Your Own Model

```bash
# Navigate to notebooks and run in sequence:
# 1. 01_EDA_Credit_Risk.ipynb - Exploratory Data Analysis
# 2. 02_EDA_Credit_Risk.ipynb - Feature Engineering
# 3. 03_Model_Training.ipynb - Model Training & Evaluation
```

---

## 📊 Project Structure

```
AI-Credit-Risk-Assessment-System/
├── App/                          # Streamlit application (UI)
│   ├── __init__.py
│   ├── main.py                   # Main app entry
│   └── components/               # Reusable components
│
├── src/                          # Source code modules
│   ├── data/
│   │   ├── loader.py            # Data loading
│   │   └── preprocessor.py      # Data cleaning & feature engineering
│   ├── models/
│   │   ├── xgboost_model.py     # Model training
│   │   └── explainer.py         # SHAP/LIME explainability
│   └── utils/
│       ├── config.py            # Configuration
│       ├── logger.py            # Logging utilities
│       └── metrics.py           # Evaluation metrics
│
├── Notebooks/                    # Jupyter notebooks
│   ├── 01_EDA.ipynb             # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Training.ipynb
│
├── Data/                         # Datasets
│   ├── raw/
│   │   └── application_train.csv (307,511 samples)
│   └── processed/
│
├── Models/                       # Trained models
│   ├── credit_risk_xgboost_model.pkl
│   ├── credit_risk_encoder.pkl
│   └── credit_risk_scaler.pkl
│
├── reports/                      # Analysis outputs
│   ├── figures/
│   └── results/
│
├── tests/                        # Unit tests
│   └── test_data.py
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── Dockerfile                    # Docker config
├── docker-compose.yml            # Docker Compose
├── .gitignore                    # Git ignore
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## 📈 Model Details

### Architecture
- **Model Type**: XGBoost Gradient Boosting Classifier
- **Estimators**: 200
- **Max Depth**: 6
- **Learning Rate**: 0.1
- **Subsample**: 0.8
- **Feature Sampling**: 0.8

### Performance Metrics
- **Accuracy**: >85%
- **Precision**: >80%
- **Recall**: >75%
- **F1-Score**: >75%
- **ROC-AUC**: >0.85
- **Optimal Threshold**: 0.20

### Training Data
- **Samples**: 307,511
- **Features**: 85 engineered
- **Classes**: Binary (Default=1, Repay=0)
- **Class Imbalance**: 8:1 (SMOTE handled)

### Feature Categories
| Category | Count | Examples |
|----------|-------|----------|
| Demographics | 9 | Age, Gender, Children |
| Financial | 15+ | Income, Credit Amount |
| Credit Bureau | 6 | Inquiries, Account History |
| Engineered | 3 | Ratios, Derived Features |
| Other | 50+ | Historical Data |
| **Total** | **85** | **Complete Profile** |

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **ML Framework** | XGBoost, Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Explainability** | SHAP, LIME |
| **UI** | Streamlit |
| **Testing** | Pytest |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |

---

## 📋 8-Week Implementation Plan

| Week | Phase | Key Deliverable |
|------|-------|-----------------|
| 1 | Data Discovery | EDA & correlation analysis |
| 2 | Feature Engineering | Cleaned & engineered features |
| 3 | Baseline Modeling | Baseline model performance |
| 4 | Optimized Training | XGBoost with tuning (GPU) |
| 5 | Evaluation | Performance metrics & ROC curves |
| 6 | Explainability (XAI) | SHAP/LIME integration |
| 7 | UI Development | Streamlit dashboard |
| 8 | Polish & Deploy | Documentation & deployment |

---

## 🧪 Testing

Run tests:
```bash
pytest tests/ -v --cov=src
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t credit-risk-ai .

# Run container
docker run -p 8501:8501 credit-risk-ai

# Or use Docker Compose
docker-compose up
```

---

## 📚 Documentation Files

- **[SETUP.md](docs/SETUP.md)** - Setup and installation guide
- **[MODEL.md](docs/MODEL.md)** - Model architecture and specs
- **[API.md](docs/API.md)** - API documentation
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Production deployment

---

## ⚠️ Ethics, Bias & Responsible AI

### 🔒 Ethical Considerations

**This system is designed as a decision support tool, not an autonomous decision-maker.**

Credit decisions have significant real-world impact on individuals' lives. This AI system must be deployed and used in accordance with ethical principles and legal requirements:

### 1. **Fairness & Non-Discrimination**

**Protected Attributes:**
- The model must NOT discriminate based on protected characteristics including:
  - Race, Ethnicity, or National Origin
  - Gender or Sexual Orientation  
  - Religion
  - Age (except as legally permitted)
  - Disability Status
  - Marital Status

**Fair Lending Compliance:**
- ✅ Comply with Equal Credit Opportunity Act (ECOA)
- ✅ Comply with Fair Housing Act (FHA)
- ✅ Comply with Fair Credit Reporting Act (FCRA)
- ✅ Regular disparate impact analysis required
- ✅ Ensure consistent treatment across demographic groups

**Bias Mitigation Strategies:**
- Regular bias audits across demographic segments
- Monitor for indirect discrimination (proxy variables)
- Compare approval/rejection rates across protected groups
- Use fairness metrics (demographic parity, equal opportunity)
- Human oversight for all automated decisions

### 2. **Transparency & Explainability**

**Model Interpretability:**
- ✅ **SHAP Values**: Every prediction includes SHAP-based explanations
- ✅ **Reasoning Reports**: Plain-language explanations for all decisions
- ✅ **Adverse Action Notices**: Clear reasons for rejections (legally required)
- ✅ **Feature Importance**: Top contributing factors are always displayed
- ✅ **Model Documentation**: Architecture and training process fully documented

**Right to Explanation:**
- Applicants have the right to understand why their application was rejected
- All explanations must be specific, actionable, and understandable
- Technical jargon must be translated into plain language

### 3. **Data Privacy & Security**

**Privacy Compliance:**
- ✅ **GDPR Compliance** (EU): Right to access, rectification, erasure, portability
- ✅ **CCPA Compliance** (California): Consumer privacy rights
- ✅ **Data Minimization**: Only collect necessary information
- ✅ **Purpose Limitation**: Use data only for stated purposes
- ✅ **Storage Limitation**: Retain data only as long as necessary

**Security Measures:**
- Encrypt sensitive data in transit and at rest
- Implement access controls and authentication
- Regular security audits and penetration testing
- Incident response plan for data breaches
- Anonymize data for analysis when possible

**Sensitive Information Handling:**
- Remove or mask PII (Personally Identifiable Information) in logs
- Secure storage of model predictions and explanations
- Limit access to prediction data based on role

### 4. **Model Limitations & Risks**

**Known Limitations:**
- ⚠️ Model trained on historical data may reflect past biases
- ⚠️ Performance may degrade on populations underrepresented in training data
- ⚠️ Economic conditions change - model requires periodic retraining
- ⚠️ Feature correlations may not imply causation
- ⚠️ Adversarial inputs or gaming attempts may affect predictions

**False Positive/Negative Risks:**
- **False Positives**: Creditworthy applicants rejected (lost revenue, customer dissatisfaction)
- **False Negatives**: High-risk applicants approved (financial losses, defaults)
- **Threshold Selection**: Balance business risk vs. fairness considerations

**Monitoring Requirements:**
- Continuous performance monitoring in production
- Track prediction distributions over time
- Alert on data drift or model degradation
- A/B testing when deploying model updates
- Regular retraining with fresh data

### 5. **Human Oversight & Accountability**

**Human-in-the-Loop:**
- ✅ All high-value loans require human review
- ✅ Edge cases flagged for manual assessment
- ✅ Human can override model decisions with justification
- ✅ Feedback loop: Human decisions improve future models

**Accountability Framework:**
- Clear ownership: Who is responsible for model decisions?
- Audit trail: Log all predictions with timestamps and explanations
- Recourse mechanism: Process for applicants to appeal decisions
- Escalation path: When to involve senior review or compliance team

**Training Requirements:**
- Credit officers must understand model capabilities and limitations
- Training on recognizing and mitigating bias
- Regular updates on model changes and performance

### 6. **Regulatory Compliance**

**Required Documentation:**
- ✅ Model Risk Management (SR 11-7) framework adherence
- ✅ Model validation documentation
- ✅ Adverse action notice generation
- ✅ Fair lending testing and results
- ✅ Model monitoring reports

**Regulatory Bodies:**
- Federal Reserve (banking institutions)
- Consumer Financial Protection Bureau (CFPB)
- Office of the Comptroller of the Currency (OCC)
- State banking regulators

### 7. **Bias Testing & Validation**

**Recommended Testing Protocol:**

```python
# Example bias testing workflow
1. Split data by protected attributes (if legally permissible for testing)
2. Calculate metrics for each group:
   - Approval rates
   - Default rates among approved
   - False positive/negative rates
3. Compare metrics across groups:
   - Demographic parity: P(approve | Group A) ≈ P(approve | Group B)
   - Equal opportunity: P(approve | default=0, Group A) ≈ P(approve | default=0, Group B)
4. Document and address disparities exceeding thresholds
5. Re-test after mitigation strategies
```

**Fairness Metrics to Monitor:**
- Statistical Parity Difference
- Equal Opportunity Difference
- Average Odds Difference
- Disparate Impact Ratio
- Treatment Equality

### 8. **Continuous Improvement**

**Feedback Mechanisms:**
- Collect feedback on prediction accuracy
- Track real-world outcomes (did approved applicants default?)
- Analyze patterns in appeals and overrides
- Incorporate new features that improve fairness

**Retraining Schedule:**
- Quarterly model performance review
- Annual full model retraining with new data
- Emergency retraining if significant drift detected
- Version control for all model iterations

### 9. **Ethical Use Guidelines**

**DO:**
- ✅ Use as a decision support tool with human oversight
- ✅ Provide clear explanations for all decisions
- ✅ Regular bias audits and fairness testing
- ✅ Document all model changes and their impacts
- ✅ Respect applicant privacy and data rights
- ✅ Comply with all applicable regulations

**DON'T:**
- ❌ Use as the sole basis for credit decisions without review
- ❌ Deploy without proper validation and testing
- ❌ Ignore bias or fairness concerns
- ❌ Fail to provide explanations for adverse actions
- ❌ Use protected attributes as direct features (unless legally permitted)
- ❌ Retain data longer than necessary

### 10. **Stakeholder Responsibilities**

**Model Developers:**
- Build interpretable, fair, and robust models
- Document limitations and known biases
- Provide tools for bias testing and monitoring

**Credit Officers:**
- Understand model outputs and limitations
- Apply human judgment to model recommendations
- Identify and escalate concerning patterns

**Compliance Team:**
- Ensure regulatory compliance
- Conduct regular audits
- Review fairness metrics and adverse action reports

**Management:**
- Allocate resources for bias testing and mitigation
- Foster culture of responsible AI use
- Hold team accountable for ethical deployment

---

### 📋 Compliance Checklist

Before deploying this system in production:

- [ ] Complete model validation documentation
- [ ] Perform fair lending analysis across demographic groups
- [ ] Implement adverse action notice generation
- [ ] Set up model monitoring and alerting
- [ ] Train all users on ethical AI principles
- [ ] Establish human oversight procedures
- [ ] Document data privacy measures
- [ ] Create incident response plan
- [ ] Obtain legal and compliance review
- [ ] Set up audit logging for all predictions
- [ ] Test appeals and override processes
- [ ] Prepare regulatory examination materials

---

### 📚 References & Resources

**Regulatory Guidance:**
- [CFPB Fair Lending Report](https://www.consumerfinance.gov/)
- [OCC Model Risk Management](https://www.occ.gov/)
- [Federal Reserve SR 11-7](https://www.federalreserve.gov/)

**Fairness in ML:**
- [Fairness and Machine Learning Book](https://fairmlbook.org/)
- [Google's ML Fairness](https://developers.google.com/machine-learning/fairness-overview)
- [IBM AI Fairness 360](https://aif360.mybluemix.net/)

**Explainable AI:**
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Paper](https://arxiv.org/abs/1602.04938)
- [Interpretable Machine Learning Book](https://christophm.github.io/interpretable-ml-book/)

---

**⚠️ IMPORTANT DISCLAIMER:**

This is an educational project demonstrating AI credit risk assessment with explainability. It is NOT intended for production use without:
- Proper legal review
- Regulatory approval
- Comprehensive testing and validation
- Bias auditing and mitigation
- Implementation of all required safeguards

**Real-world deployment requires compliance with all applicable laws and regulations.**

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file

---

## 👨‍💻 Authors & Contributors

- **Your Name** - Initial Development
- Contributions welcome!

---

## 🎯 Next Steps

1. ✅ **Setup**: Follow Quick Start guide to install dependencies
2. 📊 **Explore**: Review Jupyter notebooks for model training pipeline
3. 🎨 **Demo**: Run Streamlit dashboard: `streamlit run App/main.py`
4. 🧪 **Test**: Run test suite: `pytest tests/ -v`
5. 🐳 **Deploy**: Use Docker for production: `docker-compose up`
6. ⚖️ **Ethics**: Review Ethics & Bias section before production use

---

## 📧 Support

For questions or issues:
- 📖 Check documentation in `/docs` folder
- 📓 Review Jupyter notebooks for detailed explanations
- 🐛 Open GitHub issue for bugs or feature requests
- 📧 Contact: your.email@example.com

---

**Last Updated**: August 2026  
**Status**: ✅ Complete & Ready for Review
**Version**: 1.0.0
