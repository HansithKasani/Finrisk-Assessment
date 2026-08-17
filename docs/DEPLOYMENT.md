# Deployment Guide

## Local Development

### Streamlit App

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run App/main.py

# App opens at http://localhost:8501
```

### Jupyter Notebooks

```bash
jupyter notebook
```

Navigate to `Notebooks/` folder to run analysis and training.

## Docker Deployment

### Build Docker Image

```bash
docker build -t credit-risk-ai:latest .
```

### Run Single Container

```bash
docker run -p 8501:8501 \
  -v $(pwd)/Data:/app/Data \
  -v $(pwd)/Models:/app/Models \
  -v $(pwd)/reports:/app/reports \
  credit-risk-ai:latest
```

### Docker Compose

```bash
docker-compose up -d
```

Container runs at `http://localhost:8501`

## Cloud Deployment

### AWS Deployment

#### EC2 Instance

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3.9 python3-pip

# Clone and setup
git clone https://github.com/yourname/AI-Credit-Risk-Assessment.git
cd AI-Credit-Risk-Assessment-System
pip install -r requirements.txt

# Run with PM2 (process manager)
npm install -g pm2
pm2 start "streamlit run App/main.py" --name credit-risk
pm2 startup
pm2 save
```

#### SageMaker

```bash
# Create SageMaker notebook instance
# Upload requirements.txt and train.py

# In notebook:
import sagemaker
from sagemaker.xgboost import XGBoost

estimator = XGBoost(
    entry_point='train.py',
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    framework_version='1.0'
)
estimator.fit(data)
```

#### Elastic Beanstalk

```bash
# Initialize Elastic Beanstalk app
eb init -p python-3.9 credit-risk-app

# Create environment
eb create credit-risk-env

# Deploy
eb deploy

# Monitor
eb status
eb logs
```

### Google Cloud Platform

#### App Engine

```bash
# app.yaml
runtime: python39
env: standard

handlers:
- url: /.*
  script: auto

runtime_config:
  python_version: 3.9
```

```bash
# Deploy
gcloud app deploy

# View logs
gcloud app logs read
```

#### Cloud Run

```bash
# Build and push Docker image
docker tag credit-risk-ai gcr.io/PROJECT_ID/credit-risk:latest
docker push gcr.io/PROJECT_ID/credit-risk:latest

# Deploy
gcloud run deploy credit-risk \
  --image gcr.io/PROJECT_ID/credit-risk:latest \
  --platform managed \
  --region us-central1 \
  --port 8501
```

### Azure Deployment

#### App Service

```bash
# Create resource group
az group create -n credit-risk-rg -l eastus

# Create App Service Plan
az appservice plan create \
  -n credit-risk-plan \
  -g credit-risk-rg \
  --sku B2 --is-linux

# Create Web App
az webapp create \
  -n credit-risk-app \
  -g credit-risk-rg \
  -p credit-risk-plan \
  --runtime "PYTHON|3.9"

# Deploy
az webapp up -n credit-risk-app -g credit-risk-rg
```

## Production Best Practices

### Environment Variables

Create `.env` file:
```
FLASK_ENV=production
MODEL_PATH=/app/Models/credit_risk_xgboost_model.pkl
DEBUG=False
LOG_LEVEL=INFO
```

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### Monitoring

#### Health Checks
```python
@app.route('/health')
def health():
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now()
    }
```

#### Metrics
- Request rate
- Response time
- Error rate
- Model accuracy drift

### SSL/TLS

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem -keyout key.pem -days 365

# Run with SSL
python app.py --ssl_keyfile=key.pem --ssl_certfile=cert.pem
```

### Backup Strategy

```bash
# Backup models
tar -czf Models_backup_$(date +%Y%m%d).tar.gz Models/

# Backup database
pg_dump production_db > backup_$(date +%Y%m%d).sql

# Upload to S3
aws s3 cp Models_backup_*.tar.gz s3://bucket/backups/
```

## CI/CD Pipeline

### GitHub Actions

See `.github/workflows/ci.yml` for automated:
- Testing on push
- Linting checks
- Security scanning
- Docker build
- Auto-deployment on main branch

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    stages {
        stage('Build') {
            steps {
                sh 'docker build -t credit-risk:$BUILD_NUMBER .'
            }
        }
        stage('Test') {
            steps {
                sh 'pytest tests/'
            }
        }
        stage('Deploy') {
            steps {
                sh 'docker push registry.example.com/credit-risk:$BUILD_NUMBER'
            }
        }
    }
}
```

## Performance Optimization

### Model Optimization
- Use ONNX for faster inference
- Quantization for smaller models
- Batch predictions for throughput

### Caching
- Redis for model caching
- CDN for static assets
- Database query caching

### Scaling
- Horizontal scaling with load balancer
- Database replication for reads
- Message queues for async processing

## Security Checklist

- [ ] SSL/TLS enabled
- [ ] Authentication implemented
- [ ] Input validation on all endpoints
- [ ] Rate limiting configured
- [ ] Secrets management (AWS Secrets Manager, Azure KeyVault)
- [ ] Regular security audits
- [ ] Data encryption at rest
- [ ] Audit logging enabled
- [ ] Regular backups scheduled
- [ ] Incident response plan documented

## Troubleshooting

### Model Load Errors
```python
import joblib
model = joblib.load("Models/credit_risk_xgboost_model.pkl")
```

### Out of Memory
- Use batch processing
- Reduce model size (pruning)
- Implement caching

### High Latency
- Profile code with cProfile
- Use async processing
- Implement result caching

