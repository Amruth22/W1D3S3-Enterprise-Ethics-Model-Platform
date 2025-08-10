# Enterprise Ethics Platform

Enterprise-level extensions built on top of the Gemini Ethics Testing Framework.

## 🏢 Components

### 1. Production Bias Monitoring
**File**: `scheduled_monitor.py`
- Runs ethics testing every hour automatically
- Logs monitoring results and errors
- Background process for continuous monitoring

**Usage**:
```bash
python enterprise_platform/scheduled_monitor.py
```

### 2. Automated Compliance
**File**: `compliance_checker.py` 
- Pass/fail compliance validation against ethics reports
- Configurable thresholds for bias/fairness/ethics ratings
- Generates compliance reports with violations

**Usage**:
```bash
python enterprise_platform/compliance_checker.py
```

### 3. A/B Testing for Ethics
**File**: `ab_testing.py`
- Determines winning model based on ethics performance
- Ranks all models by ethics ratings
- Provides deployment recommendations

**Usage**:
```bash
python enterprise_platform/ab_testing.py
```

### 4. Governance Dashboard  
**File**: `governance_dashboard.py`
- Executive-level performance summary
- Risk assessment across all models
- Governance recommendations for leadership

**Usage**:
```bash
python enterprise_platform/governance_dashboard.py
```

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r enterprise_platform/requirements.txt
   ```

2. **Run base comparison first**:
   ```bash
   python multi_model_comparison.py
   ```

3. **Run enterprise components**:
   ```bash
   python enterprise_platform/compliance_checker.py
   python enterprise_platform/ab_testing.py  
   python enterprise_platform/governance_dashboard.py
   ```

## 📊 Output Files

All components generate timestamped output files in the `enterprise_platform/` directory:
- `compliance_report_YYYYMMDD_HHMMSS.json`
- `ab_test_results_YYYYMMDD_HHMMSS.json`
- `governance_dashboard_YYYYMMDD_HHMMSS.json`
- `monitoring_results.log`

## 🎯 Enterprise Features

- **Automated Monitoring**: Hourly ethics testing
- **Risk Assessment**: High/Medium/Low risk classification
- **Compliance Gates**: Pass/fail validation with thresholds
- **Executive Reporting**: Performance summaries for leadership
- **Production Ready**: Built on existing tested framework