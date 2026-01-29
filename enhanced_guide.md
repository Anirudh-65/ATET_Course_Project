# ENHANCED PROJECT: DATASET SOURCES & CLASSIFICATION STRATEGY
## Condition-Specific ML Model Selection for Battery Temperature Prediction

---

## PROJECT ENHANCEMENT OVERVIEW

### Enhanced Project Concept

**Original Approach:**
- Train 6 ML models
- Compare performance
- Select best overall model

**ENHANCED Approach:**
- Train 6 ML models + Second-order RC electrical model
- Compare performance across ALL models
- **NEW**: Analyze which ML model performs best under SPECIFIC operating conditions
- **NEW**: Deploy condition-specific ML models on MCU for optimal real-time performance
- **NEW**: Create a classification/selection system that chooses the right ML model based on:
  - Battery chemistry (NMC, LFP, LCO, NCA)
  - Operating conditions (temperature, terrain, drive cycle)
  - Usage patterns (urban, highway, mountainous)

### Why This Enhancement Matters

**Real-World Relevance:**
- Different EVs operate under vastly different conditions
- A single model may not be optimal for all scenarios
- Edge computing requires model efficiency trade-offs
- MCU resources are limited - deploying multiple lightweight models based on context is practical

**Novel Contribution:**
- First study to create a **condition-aware ML model selection framework** for battery temperature prediction
- Practical deployment strategy for embedded BMS systems
- Demonstrates how to optimize accuracy vs computational cost based on operating context

---

## CLASSIFICATION CATEGORIES

Based on available datasets, we can classify operating conditions into:

### OPTION 1: Battery Chemistry-Based Classification
**Categories:**
1. **NMC (Nickel Manganese Cobalt)** - Most common in modern EVs
2. **LFP (Lithium Iron Phosphate)** - Growing in popularity, safer
3. **NCA (Nickel Cobalt Aluminum)** - Tesla, high energy density
4. **LCO (Lithium Cobalt Oxide)** - Consumer electronics

**Advantage:** Large variation in thermal behavior between chemistries

### OPTION 2: Temperature/Climate-Based Classification
**Categories:**
1. **Cold Climate** (<10°C) - Northern regions, winter
2. **Moderate Climate** (10-25°C) - Optimal operating range
3. **Hot Climate** (>35°C) - Desert regions, summer

**Advantage:** Temperature is dominant factor in battery thermal behavior

### OPTION 3: Drive Cycle/Terrain-Based Classification
**Categories:**
1. **Urban/City** - Frequent start-stop, low speed
2. **Highway** - Constant high speed, steady power
3. **Mountainous/Hilly** - High elevation changes, variable load

**Advantage:** Directly relates to power demand patterns

### OPTION 4: Hybrid Classification (RECOMMENDED)
**Primary Category:** Battery Chemistry (NMC vs LFP - most data available)
**Secondary Category:** Operating Temperature (Cold, Moderate, Hot)

**Example Classification:**
- NMC + Cold (NMC battery in <10°C)
- NMC + Moderate (NMC battery in 10-25°C)
- NMC + Hot (NMC battery in >35°C)
- LFP + Cold
- LFP + Moderate
- LFP + Hot

**Why This Works:**
- These 6 categories align well with your 6 ML models
- Sufficient data available for all combinations
- Practical relevance (chemistry + climate are key factors)
- Achievable within 12-week timeline

---

## COMPREHENSIVE DATASET SOURCES

### 🔴 PRIMARY DATASETS (Highly Recommended - Rich Temperature & Chemistry Data)

#### 1. NASA PCoE Battery Dataset ⭐⭐⭐⭐⭐
**Source:** https://data.nasa.gov/dataset/li-ion-battery-aging-datasets
**Description:** Industry-standard dataset, most cited in research

**Details:**
- **Batteries:** 34 × 18650 cells (2Ah capacity)
- **Chemistry:** LiCoO2/graphite (LCO)
- **Temperature Conditions:** 5°C, 15°C, 25°C, 35°C, 45°C ✓
- **Data Available:** Voltage, Current, Temperature, Time, Capacity, EIS
- **Cycles:** ~168 cycles per battery
- **Quality:** Excellent, well-documented
- **Format:** .mat (MATLAB) files

**Use Case:** 
- Temperature-based classification
- Cold (5°C), Moderate (15°C, 25°C), Hot (35°C, 45°C)

**Citation:**
Saha, B., & Goebel, K. (2007). Battery Data Set. NASA Prognostics Data Repository, NASA Ames Research Center.

---

#### 2. CALCE Battery Dataset ⭐⭐⭐⭐⭐
**Source:** https://calce.umd.edu/battery-data
**Description:** Center for Advanced Life Cycle Engineering - Comprehensive multi-chemistry data

**Details:**
- **Batteries:** Multiple form factors (cylindrical, pouch, prismatic)
- **Chemistry:** LCO, LFP, NMC ✓✓✓
- **Temperature Conditions:** Various (0°C to 45°C) ✓
- **Data Available:** Full/partial cycling, storage, dynamic driving profiles, OCV, EIS
- **Quality:** Excellent, research-grade
- **Format:** Multiple formats

**Use Case:**
- Chemistry-based classification
- NMC vs LFP comparison
- Temperature effects across chemistries

**Citation:**
CALCE Battery Research Group, University of Maryland. "Battery Data." Available: https://calce.umd.edu/battery-data

---

#### 3. TU Berlin - Lithium-Ion Battery Drive Cycle Dataset ⭐⭐⭐⭐⭐
**Source:** https://depositonce.tu-berlin.de/items/7f68932b-4d43-4f49-a5d8-914b00039f87
**DOI:** 10.14279/depositonce-21133

**Details:**
- **Drive Cycles:** 12 distinct patterns (urban, highway, mixed) ✓✓✓
- **Ambient Temperatures:** 5 different levels (-10°C to 40°C) ✓✓✓
- **Data Available:** Voltage, Current, Temperature, SOC
- **Purpose:** SOC estimation under realistic conditions
- **Quality:** Excellent, recently published (2024)
- **Format:** Well-structured, public

**Use Case:**
- Drive cycle classification (Urban, Highway, Mixed)
- Temperature classification
- Most comprehensive for your project

**Citation:**
Wang, Y., et al. (2024). "A multi-scale data-driven framework for online state of charge estimation." Energy Storage Materials.

---

#### 4. Experimental Data: NCA, NMC, LFP at Multiple Temperatures ⭐⭐⭐⭐⭐
**Source:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7941039/
**Description:** Galvanostatic discharge tests at different rates and temperatures

**Details:**
- **Batteries:** 
  - Panasonic NCR-18650B (NCA, 3350mAh)
  - LG Chem INR21700-M50 (NMC, 4850mAh)
  - A123 Systems ANR26650m1-B (LFP, 2500mAh)
- **Chemistry:** NCA, NMC, LFP ✓✓✓
- **Temperatures:** 5°C, 25°C, 35°C ✓✓✓
- **Discharge Rates:** C/20 to 5C (NCA, NMC), C/20 to 20C (LFP)
- **Data Available:** Voltage, Current, Temperature vs Time
- **Quality:** Excellent, published research
- **Format:** Structured files

**Use Case:**
- Perfect for Chemistry + Temperature classification
- Direct comparison of NCA/NMC/LFP thermal behavior

**Citation:**
Carnovale, A., et al. (2021). "Experimental data of lithium-ion batteries under galvanostatic discharge tests at different rates and temperatures of operation." Data in Brief, 35, 106894.

---

#### 5. Stanford Energy Control Lab - EV Drive Cycle Aging Dataset ⭐⭐⭐⭐
**Source:** https://www.sciencedirect.com/science/article/pii/S2352340922002062
**Description:** Real EV driving profile with temperature control

**Details:**
- **Battery:** INR21700-M50T (NMC, Samsung)
- **Chemistry:** NMC/Silicon-Graphite
- **Drive Cycle:** UDDS (Urban Dynamometer Driving Schedule) ✓
- **Temperature:** 23°C (controlled)
- **Data:** Voltage, Current, Temperature, Capacity, HPPC, EIS
- **Duration:** 23 months of aging
- **Quality:** Excellent, Stanford research
- **Format:** Structured research data

**Use Case:**
- Urban drive cycle data
- Realistic EV usage patterns
- Long-term aging effects

**Citation:**
Gasper, P., et al. (2022). "Lithium-ion battery aging dataset based on electric vehicle real-driving profiles." Data in Brief, 41, 107995.

---

### 🟡 SECONDARY DATASETS (Useful for Extended Analysis)

#### 6. Battery Archive (Sandia National Labs) ⭐⭐⭐⭐
**Source:** https://www.batteryarchive.org/
**Description:** Comprehensive repository of battery test data

**Details:**
- **Chemistry Coverage:** NMC, LFP, LCO, NCA, LMO ✓✓✓
- **Temperature Range:** Various studies, -20°C to 60°C
- **Multiple Studies:** Thermal runaway, degradation, abuse testing
- **Quality:** High, DOE-sponsored
- **Access:** Public, well-documented

**Use Case:**
- Validation across multiple chemistries
- Extreme temperature data
- Safety-related thermal behavior

---

#### 7. University of Oxford - Battery Degradation Datasets ⭐⭐⭐⭐
**Source:** Referenced in Battery Archive and GitHub repositories
**Description:** Multiple degradation studies

**Details:**
- **Battery Type:** LCO pouch cells
- **Temperature:** 40°C testing
- **Drive Cycle:** Urban Artemis profile
- **Data:** Characterization every 100 cycles
- **Quality:** Research-grade

**Use Case:**
- LCO chemistry data
- Elevated temperature performance

---

#### 8. IEEE DataPort - Battery and Heating Data in Real Driving Cycles ⭐⭐⭐
**Source:** https://ieee-dataport.org/open-access/battery-and-heating-data-real-driving-cycles
**DOI:** 10.21227/6jr9-5235

**Details:**
- **Drive Cycles:** Real-world driving patterns ✓
- **Data:** Battery thermal behavior
- **Access:** IEEE membership (free for students)
- **Quality:** Good

**Use Case:**
- Real-world validation
- Heating system effects on temperature

---

#### 9. Kaggle - Battery and Heating Data in Real Driving Cycles ⭐⭐⭐
**Source:** https://www.kaggle.com/datasets/atechnohazard/battery-and-heating-data-in-real-driving-cycles

**Details:**
- **Platform:** Kaggle (easy access)
- **Drive Cycles:** Various patterns
- **Quality:** Community-verified
- **Format:** CSV, easy to use

**Use Case:**
- Quick prototyping
- Drive cycle classification

---

#### 10. Nature - WLTP-Based Battery Pack Dataset ⭐⭐⭐
**Source:** https://www.nature.com/articles/s41597-025-06229-5
**Description:** Recent (Dec 2024) comprehensive dataset

**Details:**
- **Battery Pack:** 36 Li-ion cells in parallel-series
- **Drive Cycle:** WLTP (Worldwide Harmonized Light Vehicles Test Procedure) ✓
- **Temperature:** Controlled thermal chamber
- **Data:** Voltage, Current, Temperature (multiple sensors per cell), Resistance
- **Quality:** Excellent, peer-reviewed
- **Format:** PARQUET (efficient)

**Use Case:**
- Pack-level thermal behavior
- Standardized drive cycle
- Multi-point temperature sensing

---

#### 11. EVBattery - Large-Scale Real EV Dataset ⭐⭐⭐
**Source:** https://arxiv.org/pdf/2201.12358
**Description:** Hundreds of real EVs, multiple manufacturers

**Details:**
- **Scale:** 515 vehicles, 18.2M entries
- **Manufacturers:** 3 different (anonymized)
- **Data:** Charging records, temperature, voltage, current, SOC
- **Conditions:** Real-world varied conditions ✓
- **Quality:** Large-scale, diverse

**Use Case:**
- Real-world condition diversity
- Statistical validation across many vehicles

---

### 🟢 SPECIALIZED DATASETS (For Specific Conditions)

#### 12. Mountainous Terrain - Medellín Dataset ⭐⭐⭐
**Source:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11902439/
**Description:** Electric bicycle battery in mountainous terrain

**Details:**
- **Terrain:** High altitude (>1450m), steep gradients ✓✓
- **Location:** Medellín, Colombia (Andean region)
- **Data:** Speed, altitude, temperature, distance
- **Observations:** 71,839 across 19 variables
- **Quality:** Real-world extreme conditions

**Use Case:**
- Mountainous/hilly terrain classification
- High-stress thermal behavior

---

#### 13. Chongqing Mountain Environment Dataset ⭐⭐⭐
**Source:** https://www.sciopen.com/article/10.26599/HTRD.2024.9480004
**Description:** BEV driving in mountain city

**Details:**
- **Location:** Chongqing, China (mountainous city) ✓
- **Drive Cycle:** Custom developed for mountain terrain
- **Data:** Battery data, driving patterns
- **Quality:** Recent (2024), specialized

**Use Case:**
- Mountain terrain classification
- Chinese EV data

---

#### 14. Geotab - 22,700 Real EV Climate Analysis ⭐⭐⭐⭐
**Source:** https://www.geotab.com/blog/ev-battery-health/
**Description:** Large-scale climate impact study

**Details:**
- **Vehicles:** 22,700 EVs, 21 models
- **Climate Categories:** 
  - Mild (<35% days above 25°C)
  - Hot (>35% days above 25°C) ✓✓
- **Degradation Data:** Annual rates by climate
- **Quality:** Industry-scale, practical insights

**Use Case:**
- Climate classification validation
- Real-world degradation patterns
- Hot vs mild climate comparison

---

#### 15. DOE - EV Watts Public Database ⭐⭐⭐⭐
**Source:** https://www.osti.gov/dataexplorer/biblio/dataset/1970735
**Description:** U.S. Department of Energy EV charging data

**Details:**
- **Vehicles:** Light, medium, heavy-duty EVs
- **Data:** High-frequency (10Hz) charging data
- **Conditions:** Various (SOC, temperature, EVSE) ✓
- **Access:** Public (some attributes require NDA)
- **Quality:** Government research-grade

**Use Case:**
- Diverse vehicle types
- Charging temperature effects

---

## RECOMMENDED DATASET STRATEGY FOR YOUR PROJECT

### Phase 1: Primary Datasets (Weeks 1-4)
Focus on these 3 core datasets that give you everything needed:

**1. CALCE Multi-Chemistry Dataset**
- Provides: NMC, LFP, LCO data
- Use for: Chemistry-based classification
- Download: All chemistry types

**2. TU Berlin Drive Cycle Dataset**  
- Provides: 12 drive cycles, 5 temperatures
- Use for: Temperature + drive cycle classification
- Download: Full dataset

**3. NCA/NMC/LFP Galvanostatic Dataset**
- Provides: Direct 3-chemistry comparison at 5°C, 25°C, 35°C
- Use for: Chemistry-temperature interaction analysis
- Download: All batteries, all temperatures

### Phase 2: Validation Datasets (Weeks 5-7)
Use these to validate your condition-specific models:

**4. Stanford UDDS Dataset**
- Urban drive cycle validation
- Real EV profile

**5. NASA PCoE Dataset**
- Temperature validation across 5°C to 45°C
- Industry benchmark

### Phase 3: Extension (If Time Permits, Weeks 8-9)
**6. Battery Archive** - Additional chemistries
**7. Geotab Data** - Real-world climate validation

---

## IMPLEMENTATION STRATEGY

### Step 1: Data Acquisition & Preparation (Week 1-2)

**Action Items:**

1. **Download Primary Datasets:**
```python
# Recommended download structure
data/
├── CALCE/
│   ├── NMC/
│   ├── LFP/
│   └── LCO/
├── TU_Berlin/
│   ├── drive_cycles/
│   └── temperatures/
├── Galvanostatic/
│   ├── NCA_5C/
│   ├── NCA_25C/
│   ├── NCA_35C/
│   ├── NMC_5C/
│   ├── NMC_25C/
│   ├── NMC_35C/
│   ├── LFP_5C/
│   ├── LFP_25C/
│   └── LFP_35C/
└── Stanford_UDDS/
```

2. **Standardize Data Format:**
```python
# Target columns for all datasets
standard_columns = [
    'time',           # seconds
    'voltage',        # V
    'current',        # A
    'temperature',    # °C (target variable)
    'ambient_temp',   # °C
    'SOC',           # 0-1
    'chemistry',     # categorical: NMC, LFP, LCO, NCA
    'condition'      # categorical: cold, moderate, hot
]
```

3. **Create Metadata File:**
```python
metadata = {
    'NMC_cold': {
        'chemistry': 'NMC',
        'temp_range': '5-10°C',
        'source': ['CALCE', 'Galvanostatic'],
        'samples': 15000
    },
    'NMC_moderate': {
        'chemistry': 'NMC',
        'temp_range': '15-25°C',
        'source': ['CALCE', 'TU_Berlin', 'Stanford'],
        'samples': 45000
    },
    # ... etc for all 6 categories
}
```

### Step 2: Condition Classification System (Week 3-4)

**Classification Logic:**

```python
def classify_condition(chemistry, ambient_temp):
    """
    Classify operating condition based on chemistry and temperature
    
    Returns: condition_category (str)
    """
    # Temperature thresholds
    if ambient_temp < 10:
        temp_cat = 'cold'
    elif ambient_temp < 28:
        temp_cat = 'moderate'
    else:
        temp_cat = 'hot'
    
    # Chemistry mapping
    chem_map = {
        'NMC': 'NMC',
        'NCA': 'NMC',  # Group with NMC (similar thermal behavior)
        'LFP': 'LFP',
        'LCO': 'LCO'
    }
    
    chem_cat = chem_map.get(chemistry, 'unknown')
    
    return f"{chem_cat}_{temp_cat}"

# Example usage
condition = classify_condition('NMC', 6)  # Returns: 'NMC_cold'
condition = classify_condition('LFP', 35) # Returns: 'LFP_hot'
```

### Step 3: Condition-Specific Model Training (Week 5-7)

**Training Strategy:**

```python
# For each condition category
conditions = [
    'NMC_cold', 'NMC_moderate', 'NMC_hot',
    'LFP_cold', 'LFP_moderate', 'LFP_hot'
]

# For each ML model
ml_models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(),
    'RandomForest': RandomForestRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor()
}

# Training matrix: 6 conditions × 6 models = 36 trained models
results = {}

for condition in conditions:
    # Get condition-specific data
    X_train, y_train = get_condition_data(condition)
    
    for model_name, model in ml_models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        score = evaluate(model, X_test, y_test)
        
        # Store results
        results[f"{condition}_{model_name}"] = {
            'model': model,
            'rmse': score['rmse'],
            'mae': score['mae'],
            'r2': score['r2'],
            'training_time': score['train_time'],
            'prediction_time': score['pred_time']
        }
```

### Step 4: Model Selection Analysis (Week 8-9)

**Analysis Questions to Answer:**

1. **Which model is best for each condition?**
```python
best_models = {}
for condition in conditions:
    condition_models = {k: v for k, v in results.items() if k.startswith(condition)}
    best_model = min(condition_models.items(), key=lambda x: x[1]['rmse'])
    best_models[condition] = best_model
```

2. **Is there a universal best model, or do different conditions need different models?**
```python
# Statistical significance testing
from scipy.stats import friedmanchisquare, wilcoxon

# Compare XGBoost vs Random Forest across all conditions
xgb_scores = [results[f"{c}_XGBoost"]['rmse'] for c in conditions]
rf_scores = [results[f"{c}_RandomForest"]['rmse'] for c in conditions]

stat, p_value = wilcoxon(xgb_scores, rf_scores)
print(f"Significance: p={p_value}")
```

3. **What are the accuracy vs efficiency trade-offs?**
```python
# Create Pareto front analysis
import matplotlib.pyplot as plt

for condition in conditions:
    accuracies = [results[f"{condition}_{m}"]['rmse'] for m in ml_models]
    speeds = [results[f"{condition}_{m}"]['prediction_time'] for m in ml_models]
    
    plt.scatter(speeds, accuracies, label=condition)
    
plt.xlabel('Prediction Time (ms)')
plt.ylabel('RMSE (°C)')
plt.title('Accuracy vs Speed Trade-off')
plt.legend()
```

### Step 5: Deployment Strategy (Week 10-11)

**MCU Deployment Logic:**

```python
class ConditionAwarePredictor:
    """
    Deploys appropriate ML model based on operating conditions
    """
    def __init__(self, model_library):
        """
        model_library: dict of {condition: best_model}
        """
        self.models = model_library
        
    def predict_temperature(self, voltage, current, SOC, 
                          chemistry, ambient_temp):
        """
        Select model and predict temperature
        """
        # Classify current condition
        condition = classify_condition(chemistry, ambient_temp)
        
        # Get appropriate model
        model = self.models.get(condition)
        
        if model is None:
            # Fallback to general model
            model = self.models['general']
        
        # Prepare features
        features = self.engineer_features(voltage, current, SOC)
        
        # Predict
        temp_pred = model.predict(features)
        
        return temp_pred, condition

# Example deployment
predictor = ConditionAwarePredictor({
    'NMC_cold': best_models['NMC_cold'],
    'NMC_moderate': best_models['NMC_moderate'],
    'NMC_hot': best_models['NMC_hot'],
    'LFP_cold': best_models['LFP_cold'],
    'LFP_moderate': best_models['LFP_moderate'],
    'LFP_hot': best_models['LFP_hot']
})

# Real-time prediction
temp, condition = predictor.predict_temperature(
    voltage=3.7, 
    current=2.0, 
    SOC=0.65,
    chemistry='NMC',
    ambient_temp=5
)
print(f"Predicted: {temp}°C under {condition} conditions")
```

---

## EXPECTED RESULTS & INSIGHTS

### Hypothesis 1: Chemistry Matters
**Expected Finding:**
- LFP will show more stable thermal behavior (lower temperature rise)
- NMC/NCA will be more temperature-sensitive
- Models for LFP may be simpler (Decision Tree sufficient)
- Models for NMC may need more complexity (XGBoost better)

### Hypothesis 2: Temperature Effects
**Expected Finding:**
- Cold conditions: Higher internal resistance → more heat generation
- Hot conditions: Lower heat dissipation → temperature accumulation
- Different models may excel in different temperature ranges

### Hypothesis 3: Model Selection Patterns
**Possible Outcomes:**

**Scenario A: Universal Winner**
- One model (e.g., XGBoost) is best for all conditions
- Conclusion: Deploy single model for all cases

**Scenario B: Condition-Specific Winners**
- Cold: Random Forest best (handles high R0 variability)
- Moderate: XGBoost best (balanced complexity)
- Hot: Decision Tree sufficient (more linear behavior at high temp)
- Conclusion: Deploy condition-specific models

**Scenario C: Accuracy-Speed Trade-off**
- XGBoost most accurate but slowest
- Random Forest: 95% of XGBoost accuracy, 3x faster
- Conclusion: Deploy RF for real-time, XGBoost for offline analysis

---

## DELIVERABLES FOR ENHANCED PROJECT

### 1. Comprehensive Dataset Analysis Report
- Summary of all datasets used
- Data preprocessing pipeline
- Condition classification methodology

### 2. Condition-Specific Model Performance Matrix
```
               | NMC_cold | NMC_mod | NMC_hot | LFP_cold | LFP_mod | LFP_hot |
---------------|----------|---------|---------|----------|---------|---------|
Linear Reg     |   1.8    |   1.5   |   2.1   |   1.2    |   1.0   |   1.4   |
Decision Tree  |   1.1    |   0.9   |   1.3   |   0.7    |   0.6   |   0.9   |
Random Forest  |   0.7    |   0.6   |   0.8   |   0.5    |   0.4   |   0.6   |
XGBoost        |   0.6    |   0.5   |   0.7   |   0.4    |   0.3   |   0.5   |
SVR            |   0.9    |   0.7   |   1.0   |   0.6    |   0.5   |   0.7   |
KNN            |   1.0    |   0.8   |   1.2   |   0.7    |   0.6   |   0.8   |
---------------|----------|---------|---------|----------|---------|---------|
BEST MODEL     |   XGB    |   XGB   |   XGB   |   XGB    |   XGB   |   RF    |
```
(Values are example RMSE in °C)

### 3. Model Selection Framework
- Classification algorithm
- Deployment decision tree
- Performance vs computational cost analysis

### 4. Visualization Suite
- Heatmaps showing best model per condition
- Performance comparison across conditions
- Accuracy vs speed Pareto fronts

### 5. Code Implementation
- Condition classifier module
- Model selector class
- Real-time prediction framework

---

## UPDATED PROJECT TIMELINE

### Week 1-2: Data Acquisition & Condition Definition
- Download 3 primary datasets
- Define condition categories
- Implement classification logic

### Week 3-4: Electrical Model + Feature Engineering
- Second-order RC model
- Condition-specific feature engineering
- Data preprocessing pipeline

### Week 5-7: ML Training (36 models total)
- Train all 6 models for each of 6 conditions
- Hyperparameter tuning per condition
- Cross-validation within conditions

### Week 8-9: Analysis & Model Selection
- Performance comparison across conditions
- Statistical significance testing
- Develop deployment strategy

### Week 10-11: Documentation & Validation
- Write technical report
- Create visualizations
- Validate deployment framework

### Week 12: Presentation & Final Touches
- Prepare presentation
- Final testing
- Submission

---

## CITATIONS FOR DATASETS

**Reference Format (IEEE Style):**

[1] B. Saha and K. Goebel, "Battery data set," NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA, 2007.

[2] CALCE Battery Research Group, "Battery data," University of Maryland Center for Advanced Life Cycle Engineering. [Online]. Available: https://calce.umd.edu/battery-data

[3] Y. Wang et al., "A multi-scale data-driven framework for online state of charge estimation of lithium-ion batteries with a novel public drive cycle dataset," Energy Storage Materials, 2024.

[4] A. Carnovale, X. Li, B. Du, and S. Xie, "Experimental data of lithium-ion batteries under galvanostatic discharge tests at different rates and temperatures of operation," Data in Brief, vol. 35, p. 106894, 2021.

[5] P. Gasper et al., "Lithium-ion battery aging dataset based on electric vehicle real-driving profiles," Data in Brief, vol. 41, p. 107995, 2022.

[6] "Battery archive," Sandia National Laboratories. [Online]. Available: https://www.batteryarchive.org/

[7] M. Steinstraeter, J. Buberger, and D. Trifonov, "Battery and heating data in real driving cycles," IEEE Dataport, Oct. 2020, doi: 10.21227/6jr9-5235.

[8] "EV battery health: Key findings from 22,700 vehicle data analysis," Geotab, Jan. 2025. [Online]. Available: https://www.geotab.com/blog/ev-battery-health/

---

## FINAL RECOMMENDATIONS

### For Maximum Impact in 12 Weeks:

1. **Use Hybrid Classification:** Chemistry (NMC, LFP) × Temperature (Cold, Moderate, Hot) = 6 categories

2. **Focus on 3 Primary Datasets:**
   - CALCE (multi-chemistry)
   - TU Berlin (drive cycles + temperatures)
   - Galvanostatic NCA/NMC/LFP (direct comparison)

3. **Train 36 Models:** 6 conditions × 6 ML models

4. **Key Analysis:** Show that different conditions require different optimal models

5. **Deliverable:** Condition-aware model selector for BMS deployment

### Success Criteria:
- Demonstrate at least 20% performance improvement when using condition-specific models vs universal model
- Achieve <1°C RMSE for at least 4 out of 6 conditions
- Create working deployment framework

---

**This enhanced project demonstrates:**
✓ Real-world applicability
✓ Novel contribution to field
✓ Practical BMS implementation
✓ Systematic scientific approach
✓ Comprehensive dataset utilization

Good luck with your enhanced project! 🚀
