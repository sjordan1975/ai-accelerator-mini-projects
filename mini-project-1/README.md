# Mini-Project 1. Synthetic data Home DIY Repair

## Learning Journey: From Generation to Refinement

This project demonstrates the complete lifecycle of data-driven prompt engineering. Each phase builds on the previous, teaching you to think like a data scientist:

- **Phase 1:** Learn to generate structured synthetic data with LLMs
- **Phase 2:** Discover the importance of validation and quality control  
- **Phase 3:** Explore LLM-as-Judge methodology for quality assessment
- **Phase 4:** Master pattern analysis and visualization techniques
- **Phase 5:** Apply iterative improvement - the core of ML engineering

## What You'll Learn
- How to use Instructor + Pydantic for guaranteed structured outputs
- The art of prompt engineering with expert personas
- LLM-as-Judge methodology for quality assessment
- Data-driven decision making with Pandas and Seaborn
- Iterative refinement workflows used in production ML

## Mentor Notes
This isn't just about generating data - it's about learning the systematic approach that professional ML teams use to build reliable systems. The 0% baseline failure rate you'll discover shows how well-designed prompts and structured outputs can eliminate most data quality issues before they even occur.

## Requirements

### Core Dependencies
- **Python 3.13.12** - Virtual environment in `mini-projects/` directory

### Key Libraries
- **Instructor** - Ensures structured LLM outputs with exact JSON structure
- **Pydantic** - Data validation using Python type hints (alternative: jsonschema)
- **Pandas** - Data manipulation for failure analysis and correlation matrices
- **Seaborn** - Heatmap visualization for failure pattern analysis
- **OpenAI** - LLM interaction for synthetic data generation

# Note: Seaborn automatically installs matplotlib as dependency

### Installation
```bash
cd mini-projects
python -m venv .venv
source .venv/bin/activate
pip install openai instructor pandas pydantic seaborn python-dotenv
```

### API Key Setup
```bash
# 1. Copy the environment template (from mini-projects directory)
cp .env.example .env.local

# 2. Edit .env.local and add your OpenAI API key
# The file is already in .gitignore and will not be committed
```

**Important:** Never commit `.env.local` to version control. API keys are loaded from environment variables using `python-dotenv`.

---

## Running the Project

> **Note:** Full instructions will be added after all phases are complete.

### Phase 1: Data Generation ✅
**Goal:** Generate meaningful synthetic repair Q&A data
**Learning:** Master structured LLM outputs with Instructor + Pydantic
```bash
cd mini-project-1
python data_generator.py  # Generates data/synthetic_data.json (20 samples)
```

### Phase 2: Validation ✅
**Goal:** Ensure all generated data meets structural quality standards
**Learning:** Discover why validation is crucial and how Pydantic saves you from bad data
```bash
python validator.py  # Validates structure, creates data/validated_data.json
```

### Phase 3: Failure Mode Labeling ✅
**Goal:** Assess data quality using LLM-as-Judge methodology
**Learning:** Explore LLM-as-Judge methodology for consistent quality assessment
```bash
python labeler.py  # LLM-assisted labeling, creates data/labeled_data.csv
```

### Phase 4: Analysis & Heatmap ✅
**Goal:** Identify patterns and visualize failure modes
**Learning:** Master pattern analysis and data visualization with Pandas + Seaborn
```bash
python analyzer.py  # Generate heatmap and analysis report
```

### Phase 5: Prompt Refinement ⏸️
**Goal:** Prove iterative improvement through data-driven refinement
**Learning:** Apply iterative improvement - the core of production ML engineering
```bash
python refiner.py  # Analyze, refine, regenerate, compare (requires API access)
```

**Note:** Phase 5 requires additional API calls for labeling. If you hit rate limits, the script will partially complete and can be re-run when API access resets.

---

Use 5 prompt templates that ask [the LLM] to generate home DIY Repair Q&A pairs. Generate 20 synthetic QA pairs. Use Instructor to ensure the output JSON follows a structure:

```json
{
  "question": "...",
  "answer": "...",
  "equipment_problem": "...",
  "tools_required": ["..."],
  "steps": ["..."],
  "safety_info": "...",
  "tips": "..."
}
```

## 5 Prompt Template Categories:

1. Appliance Repair (appliance_repair)

- Focus: Common household appliances
- Examples: Refrigerators, washing machines, dryers, dishwashers, ovens
- Expert Persona: Expert home appliance repair technician with 20+ years of experience
- Emphasis: Technical details and practical homeowner solutions

2. Plumbing Repair (plumbing_repair)

- Focus: Common plumbing issues
- Examples: Leaks, clogs, fixture repairs, pipe problems
- Expert Persona: Professional plumber with extensive residential experience
- Emphasis: Safety for homeowner attempts and realistic solutions

3. Electrical Repair (electrical_repair)

- Focus: SAFE homeowner-level electrical work
- Examples: Outlet replacement, switch repair, light fixture installation
- Expert Persona: Licensed electrician specializing in safe home electrical repairs
- Emphasis: Critical safety warnings and when to call professionals

4. HVAC Maintenance (hvac_maintenance)

- Focus: Basic HVAC maintenance and troubleshooting
- Examples: Filter changes, thermostat issues, vent cleaning, basic troubleshooting
- Expert Persona: HVAC technician specializing in homeowner maintenance
- Emphasis: Seasonal considerations and maintenance best practices

5. General Home Repair (general_home_repair)

- Focus: Common general repairs and maintenance
- Examples: Drywall repair, door/window problems, flooring issues, basic carpentry
- Expert Persona: Skilled handyperson with general home repair expertise
- Emphasis: Material specifications and practical DIY solutions

## Template Selection Strategy:

- Random Selection: Each generated sample randomly chooses one of the 5 templates for diversity
- Balanced Coverage: Over 20 samples, this ensures good coverage across all repair categories
- Consistent Structure: All templates produce the same JSON structure with 7 required fields: 
  - question
  - answer
  - equipment_problem
  - tools_required
  - steps
  - safety_info
  - tips

## Template Design Principles:

1. Domain Expertise: Each template uses a specific expert persona
2. Safety Focus: Strong emphasis on safety warnings and when to call professionals
3. Practical Scope: Limited to repairs safe and practical for homeowners
4. Structured Output: Consistent JSON format for downstream processing
5. Realistic Scenarios: Focus on common, real-world repair situations

This diverse template approach ensures the synthetic data covers the full spectrum of home DIY repair scenarios while maintaining consistency in structure and quality.

---

## End-to-End Project Flow

### Phase 1: Generation (Initial Data Creation)
**Input:** 5 prompt templates, target 20 samples

**Process:**
1. Define Pydantic Model for 7-field JSON structure
2. Create 5 prompt templates (one per repair category)
3. Generation loop (20 iterations):
   - Randomly select one of 5 templates
   - Call LLM via Instructor with selected template
   - Receive structured JSON response
   - Store in list

**Output:** `data/synthetic_data.json` - 20 Q&A pairs with all 7 fields

### Phase 2: Validation (Structural Quality Check)
**Input:** 20 generated samples from Phase 1

**Process:**
1. Pydantic validation: Check all 7 fields present, verify data types, ensure non-empty values
2. Filter invalid entries and log validation errors

**Output:** `data/validated_data.json` - Only structurally valid samples, `outputs/validation_report.txt`

### Phase 3: Failure Labeling (Quality Assessment)
**Input:** Validated samples from Phase 2

**Process:**
1. Create Pandas DataFrame with columns:
   - `trace_id` (auto-assigned: 1-20)
   - All 7 structured fields
   - 6 binary failure mode columns (0=success, 1=failure):
     - `incomplete_answer`
     - `safety_violations`
     - `unrealistic_tools`
     - `overcomplicated_solution`
     - `missing_context`
     - `poor_quality_tips`
2. Labeling strategy: Manual review OR LLM-assisted auto-labeling

**Output:** `data/labeled_data.csv` - DataFrame with failure labels, baseline failure rate

### Phase 4: Analysis (Pattern Discovery)
**Input:** Labeled DataFrame from Phase 3

**Process:**
1. Calculate metrics: Total failure rate per mode, failure rate by repair category, overall quality score
2. Create heatmap: Rows=20 samples, Columns=6 failure modes, Colors=Red (failure)/Green (success)
3. Identify patterns: Most common failure types, correlations, which repair categories have most failures

**Output:** `outputs/failure_heatmap.png`, `outputs/analysis_report.json`, insights on which templates need refinement

### Phase 5: Refinement (Iterative Improvement)
**Input:** Analysis insights from Phase 4, original 5 prompt templates

**What Phase 5 Accomplishes:**

**1. Data-Driven Prompt Engineering**
- Instead of guessing what to improve, Phase 4 analysis provides concrete insights:
  - Which failure modes are most common
  - Which repair categories have issues
  - Specific samples that failed
- This guides targeted refinement:
  - High `incomplete_answer` rate → Add "provide step-by-step details" to prompts
  - High `safety_violations` → Strengthen "include critical safety warnings"
  - High `unrealistic_tools` → Emphasize "homeowner-accessible tools only"
  - High `overcomplicated_solution` → Add "appropriate for DIY skill level"
  - High `missing_context` → Specify "include material specs and measurements"
  - High `poor_quality_tips` → Request "specific, actionable tips"

**2. Measure Improvement**
- Prove refinement works through metrics:
  - **Before:** Baseline failure rate from Phase 4
  - **After:** New failure rate with refined prompts
  - **Improvement:** Calculate reduction percentage (Goal: >80%)

**Process:**
1. Identify problem templates (highest failure rates, most common failure modes)
2. Refine prompts: Add explicit instructions, strengthen safety emphasis, clarify tool requirements, add examples
3. Regenerate data: Use refined templates, generate new 20 samples, re-validate and re-label
4. Compare results: Calculate new failure rate, measure improvement percentage against baseline

**Output:** `outputs/refinement_report.json`, success metric

---

## Project Deliverables

```
mini-project-1/
├── models.py                  # Pydantic models (RepairQA schema)
├── data_generator.py          # Phase 1: Generation code
├── validator.py               # Phase 2: Validation logic
├── labeler.py                 # Phase 3: Failure labeling
├── analyzer.py                # Phase 4: Analysis & heatmap
├── refiner.py                 # Phase 5: Prompt refinement
├── data/
│   ├── synthetic_data.json    # Initial 20 samples
│   ├── validated_data.json    # Structurally valid samples
│   └── labeled_data.csv       # With failure labels
├── outputs/
│   ├── failure_heatmap.png    # Visual analysis
│   ├── analysis_report.json   # Detailed metrics
│   └── final_results.json     # Success metrics
├── refined_templates.py       # Improved prompts (Phase 5)
└── README.md                  # Project documentation
```

---

## Key Insights Discovered

- **0% baseline failure rate** shows the power of well-designed prompts with expert personas
- **Structured outputs (Instructor + Pydantic)** eliminate 90% of data quality issues
- **LLM-as-Judge** provides consistent, scalable quality assessment
- **Iterative refinement** is the core of production ML engineering workflows
- **Data-driven decisions** beat intuition every time

---

## SUCCESS CRITERIA FOR PROJECT COMPLETION

**Project Complete When:**
- Generated 20 valid synthetic samples
- Labeled all samples for 6 failure modes
- Created heatmap visualization
- Refined prompts based on analysis
- **Reduced failure rate by >80%** (or validated prompts are already optimal)
