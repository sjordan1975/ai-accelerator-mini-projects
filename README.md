# Mini-Projects - AI Accelerator Course

This directory contains 8 mini-projects required for course completion certification.

---

## 🎯 Mentor-Mentee Learning Approach

This workspace follows a **systematic, step-by-step learning methodology** where:
- **You (mentee)** are building practical AI/ML skills through hands-on projects
- **AI assistant (mentor)** guides you through best practices and industry standards
- **Learning is prioritized** over speed - understanding trumps completion

---

## 📋 Coding Standards & Guidelines

### Python Version
- **Python 3.13.12** (or latest stable 3.x)
- Managed via `pyenv` for version consistency
- Single shared virtual environment in `mini-projects/.venv/`

### Code Documentation Philosophy
**Slightly pedantic explanations in comments** for learning purposes:
- Explain *what* libraries do (e.g., "Pydantic is a data validation library...")
- Explain *why* we use specific patterns
- Include context that experienced developers might skip
- **Goal:** Future you should understand the code without external resources

**Example:**
```python
"""
About Pydantic:
Pydantic is a Python library for data validation using type hints. It ensures
data matches a defined structure and automatically validates types, converts
values when possible, and raises clear errors for invalid data.
"""
```

### Best Practices
1. **Type Hints:** Use Python type hints for all function signatures
2. **Pydantic Models:** Separate data models in `models.py`
3. **Docstrings:** Module, class, and function docstrings following Google style
4. **Error Handling:** Explicit error handling with descriptive messages
5. **Testing:** Write tests before major implementations when possible
6. **Validation:** Validate data at boundaries (API inputs, file reads, LLM outputs)
7. **Environment Variables:** Store sensitive data (API keys) in `.env.local`, never hardcode in source

### AI/ML Specific Standards
- **Structured LLM Outputs:** Use Instructor + Pydantic for guaranteed JSON structure
- **Prompt Engineering:** Document prompt templates with clear intent
- **Evaluation Metrics:** Always establish baseline before optimization
- **Iterative Refinement:** Measure → Analyze → Refine → Validate
- **Reproducibility:** Set random seeds, log parameters, version datasets

### Code Organization
```
mini-project-X/
├── models.py              # Pydantic schemas (data structure definitions)
├── main_script.py         # Primary implementation
├── utils.py               # Helper functions (if needed)
├── requirements.txt       # Project-specific dependencies (if any)
├── data/                  # Generated or input data
├── outputs/               # Results, visualizations, reports
└── README.md              # Project-specific documentation
```

---

## 🔧 Environment Setup

### Initial Setup (One-time)
```bash
cd ./mini-projects
python -m venv .venv
source .venv/bin/activate
```

### API Key Configuration (IMPORTANT)
```bash
# 1. Copy the environment template
cp .env.example .env.local

# 2. Edit .env.local and add your actual API key
# NEVER commit .env.local to version control!
```

**Security Best Practice:**
- API keys and secrets go in `.env.local` (gitignored)
- Use `python-dotenv` to load environment variables
- Never hardcode credentials in source code
- Share `.env.example` as a template (without actual keys)

### Activate Environment (Each session)
```bash
source ./mini-projects/.venv/bin/activate
```

### IDE Python Interpreter Path
```
./mini-projects/.venv/bin/python
```

### Common Dependencies
```bash
# Install as needed across projects
pip install openai instructor pandas pydantic seaborn matplotlib numpy scikit-learn python-dotenv
```

---

## 📚 Learning Principles

### 1. **Understand Before Implementing**
- Read documentation and examples first
- Ask "why" before "how"
- Clarify requirements before coding

### 2. **Systematic Debugging**
- Analyze error patterns systematically
- Make targeted, evidence-based changes
- Validate improvements with metrics

### 3. **Industry Best Practices**
- Follow Python PEP 8 style guide
- Use modern Python features (3.10+)
- Consult latest library documentation
- Verify approaches against current industry standards

### 4. **Iterative Development**
- Start with baseline/minimal viable implementation
- Measure performance
- Identify specific issues
- Refine incrementally
- Document learnings

### 5. **Documentation as Learning**
- Write README files that explain the "why"
- Comment non-obvious decisions
- Create solution summaries after completion
- Build a knowledge base for future reference

---

## 🎓 Project Completion Criteria

Each mini-project is complete when:
- ✅ **Functional:** Meets all specified requirements
- ✅ **Documented:** Clear README with approach explanation
- ✅ **Validated:** Results verified against success criteria
- ✅ **Understood:** You can explain design decisions and trade-offs
- ✅ **Reproducible:** Can be run by others following documentation

---

## 📊 Progress Tracking

**Completion Requirement:** 8 out of 8 mini-projects

| # | Project Name | Status | Completion Date |
|---|--------------|--------|-----------------|
| 1 | Synthetic Data Home DIY Repair | 🚧 In Progress | - |
| 2 | TBD | ⏳ Pending | - |
| 3 | TBD | ⏳ Pending | - |
| 4 | TBD | ⏳ Pending | - |
| 5 | TBD | ⏳ Pending | - |
| 6 | TBD | ⏳ Pending | - |
| 7 | TBD | ⏳ Pending | - |
| 8 | TBD | ⏳ Pending | - |

---

## 🤝 Mentor-Mentee Workflow

### When Starting a New Project:
1. **Understand requirements** - Read project description thoroughly
2. **Clarify objectives** - Discuss success criteria and approach
3. **Plan phases** - Break down into manageable steps
4. **Set up structure** - Create files and documentation skeleton

### During Implementation:
1. **Step-by-step progress** - One phase at a time
2. **Validate frequently** - Test after each major component
3. **Ask questions** - Clarify when uncertain
4. **Document learnings** - Capture insights as you go

### After Completion:
1. **Solution summary** - Document approach and results
2. **Reflect on learnings** - What worked? What didn't?
3. **Update progress** - Mark project complete
4. **Prepare for next** - Apply learnings to next project

---

## 🔍 Quality Standards

### Code Quality
- **Readable:** Clear variable names, logical structure
- **Maintainable:** Modular functions, separated concerns
- **Tested:** Validated against expected behavior
- **Documented:** Comments explain non-obvious logic

### AI/ML Quality
- **Reproducible:** Consistent results with same inputs
- **Validated:** Metrics confirm expected performance
- **Explainable:** Can describe how and why it works
- **Practical:** Solves real problems effectively

---

## 📖 Resources

### Python & ML Libraries
- [Python 3.13 Documentation](https://docs.python.org/3.13/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Instructor Library](https://python.useinstructor.com/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

### Best Practices
- [PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Real Python Tutorials](https://realpython.com/)

---

**Remember:** This is a learning journey. Take your time, ask questions, and build deep understanding. Quality over speed. 🚀
