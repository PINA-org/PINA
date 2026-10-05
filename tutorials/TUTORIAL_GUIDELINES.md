# PINA Tutorial Guidelines

Welcome to the **PINA Tutorial Guidelines** — a guiding document that defines the structure, style, and pedagogical philosophy for all tutorials in the **PINA** package. The goal of this guideline is to ensure that all learning materials are **clear, consistent, pedagogically sound, and beginner-friendly**, while remaining powerful enough to support advanced use cases.


## Purpose

The purpose of the PINA tutorials is to help users:

- Gaining a solid understanding of the PINA library and its core functionalities.
- Learning how to work with the PINA modules.
- Explore practical and advanced applications using consistent, hands-on code examples.


## Guiding Principles

1. **Clarity Over Cleverness**  
   Tutorials should aim to teach, not impress. Prioritize readable and understandable code and explanations.

2. **Progressive Disclosure of Complexity**  
   Start simple and gradually introduce complexity. Avoid overwhelming users early on.

3. **Consistency is Key**  
   All tutorials should follow a common structure (see below), use the same markdown and code formatting, and have a predictable flow.

4. **Real Applications, Real Problems**  
   Ground tutorials in real Scientific Applications or datasets, wherever possible. Bridge theory and implementation.


## Tutorial Structure

To ensure clarity, consistency, and accessibility, all PINA tutorials should follow the same standardized format.

### 1. Title

Each tutorial must begin with a clear and descriptive title in the following format: **Tutorial: TUTORIAL_TITLE**. The title should succinctly communicate the focus and objective of the tutorial.

### 2. Introducing the Topic

Immediately after the title, include a short introduction that outlines the tutorial's purpose and scope.

- Briefly explain what the tutorial covers and why it’s useful.
- Link to relevant research papers, publications, or external resources if applicable.
- List the core PINA components or modules that will be utilized.

### 3. Imports and Setup

Include a Python code cell with the necessary setup. This ensures that the tutorial runs both locally and on platforms like Google Colab.

```python
## Routine needed to run the notebook on Google Colab
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    !pip install "pina-mathlab[tutorial]"

import torch                        # if used
import matplotlib.pyplot as plt     # if used
import warnings                     # if needed

warnings.filterwarnings("ignore")

# Additional PINA and problem-specific imports
...
```

### 3. Data Generation or Loading
* Describe how the data is generated or loaded.
* Include commentary on data structure, format, and content.
* If applicable, visualize key features of the dataset or simulation domain.

### 4. Main Body
The core section of the tutorial should present the problem-solving process in a clear, structured, and pedagogical way. This is where the tutorial delivers the key learning objectives.

- Guide the user step-by-step through the PINA workflow.
- Introduce relevant PINA components as they are used.
- Provide context and explain the rationale behind modeling decisions.
- Break down complex sections with inline comments and markdown explanations.
- Emphasize the relevance of each step to the broader goal of the tutorial.

### 5. Results, Visualization and Error Analysis
- Show relevant plots of results (e.g., predicted vs. ground truth).
- Quantify performance using metrics like loss or relative error.
- Discuss the outcomes: strengths, limitations, and any unexpected behavior

### 6. What's Next?
All the tutorials are concluded with the **What's Next?** section,giving suggestions for further exploration. For this use the following format:
```markdown
## What's Next?

Congratulations on completing the ..., here are a few directions you can explore:

1. **Direction 1** — Suggestion ....

2. **Direction 2** — Suggestion ....

3. **...and many more!** — Other suggestions ....

For more resources and tutorials, check out the [PINA Documentation](https://PINA-org.github.io/PINA/).
```

## Writing Style

- Use **clear markdown headers** to segment sections.
- Include **inline math** with `$...$` and display math with `$$...$$`.
- Keep paragraphs short and focused.
- Use **bold** and *italic* for emphasis and structure.
- Include comments in code for clarity.


## Testing Tutorials

Every tutorial should:
- Be executable from top to bottom.
- Use the `tutorial` requirements in the [`pyproject.toml`](https://github.com/PINA-org/PINA/blob/6ed3ca04fee3ae3673d53ea384437ce270f008da/pyproject.toml#L40) file.


## Contributing Checklist

We welcome contributions! If you’re writing a tutorial:
1. The tutorial follows this guidelines for structure and tone.
2. The tutorial is simple and modular — one tutorial per concept.
3. The tutorial PRs contains only the `.ipynb` file, and the updated `README.md` file.

