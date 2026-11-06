---
name: write-docstring
description: >-
  Write, update, or review docstrings in PINA Python files. Use when the user
  asks to document modules, classes, constructors, methods, properties, or
  helper functions, or to check docstring formatting and accuracy. Covers
  PINA's Sphinx/reStructuredText conventions, parameters, exceptions, returns,
  examples, mathematics, and documentation-only changes.
license: MIT
compatibility: opencode, codex, claude
metadata:
  audience: developers
  workflow: documentation
---

# Write Docstrings in PINA

> [!IMPORTANT]
> Read [RULES.md](../RULES.md) before using this skill — it applies to all skills.

Use this skill to document the current behavior of Python code in PINA.
Only modify docstrings in Python files inside the repository's `pina/`
directory. Do not modify files in `tests/`, `docs/`, examples, or other
directories when applying this skill. Read related files elsewhere if needed
to verify behavior. This scope restriction does not prevent separately
requested maintenance of the skill itself or its cross-tool setup.
For creating or editing skills themselves, use [create-skill](../create-skill/SKILL.md).

## Step 1 — Inspect the implementation and establish scope

1. Identify the requested files and objects. If the scope is unclear, ask the
   user before making broad documentation changes.
2. Read each implementation, its signature, and relevant helpers or base
   classes. Verify defaults, accepted types, shapes, return values, side
   effects, keyword options, and validation errors rather than guessing.
3. Before editing Python docstrings under `pina/`, announce the specific files
   and wait for explicit user approval, as required by `RULES.md`.
4. Keep documentation-only work limited to docstrings. Do not change
   signatures, defaults, control flow, validation, imports, or other
   implementation behavior. Report discovered bugs separately.

Existing docstrings can be outdated; the implementation is the source of truth.

### Minimal-change rule

Review each scoped docstring against the implementation and this skill's
conventions before deciding whether to edit it. Modify it only to correct a
factual error, supply required missing information, or fix a documented
convention violation. Add a missing docstring when required.

Leave already accurate, complete, and compliant docstrings unchanged. Do not
rewrite correct text merely for stylistic preference, replace an entire
docstring when a targeted correction suffices, or touch unrelated docstrings.
If no changes are needed, report that result without editing files.

Harmonize wording across interfaces, base classes, and concrete classes only
to resolve meaningful inconsistencies in shared contracts. Different but
equally clear and accurate wording alone does not justify an edit.

## Step 2 — Cover every relevant object

Within the requested scope:

- Begin each module with a short docstring describing its contents.
- Give every class a docstring describing its purpose and behavior.
- Give every method a docstring describing what it does, its parameters,
  applicable exceptions, and its return value when not exclusively `None`.
- Apply the same conventions to constructors, properties, and helper
  functions, including private helpers.
- Document constructor arguments in the constructor docstring; use the class
  docstring to explain the class's role and behavior.
- Describe property values as returns in getter docstrings. Document setter
  arguments and validation where applicable.
- Never document `self` or `cls` as parameters.
- Compare corresponding methods in interfaces, base classes, and concrete
  classes. Reuse consistent or identical descriptions, parameter wording,
  and return documentation wherever their contracts and behavior agree.
  Preserve specific logical differences, including accepted types, defaults,
  validation, side effects, and return values. Do not copy implementation
  details into an interface contract unless they are required by it, or edit
  related classes outside the approved scope just to harmonize wording.

## Step 3 — Use the required structure and syntax

Use Sphinx/reStructuredText inside docstrings, not Google/NumPy-style sections
or Markdown formatting. Markdown in this skill is only for presenting the
instructions and templates.

Start with a short summary. Follow it with a blank line before any detailed
explanation or fields. Use this order, omitting sections that do not apply:

1. Summary and detailed explanation.
2. Parameters, with any associated type fields immediately after the parameter.
3. Exceptions (`:raises ExceptionType:`).
4. Return description and type (`:return:`, then `:rtype:`).
5. Helpful notes, warnings, and examples.

Indent multiline field descriptions beneath their field. Do not continue a
description as an unindented paragraph.

### Line length

Every physical source line belonging to a docstring must be at most 80
characters long, including indentation and opening or closing quotes. There
are no exceptions for examples, mathematics, URLs, cross-references, or type
fields. Wrap prose and field continuations, split examples using valid
Python continuation syntax, and restructure long mathematics or references
without breaking Sphinx syntax. Use indented continuation lines for long
type fields, breaking at valid type-expression boundaries.

Check the raw source lines, not dedented docstring text. Do not rely on
`python -m black -l 80 <dir>` to wrap docstring contents. Do not reformat
implementation code as part of documentation-only work.

### Constructors

Every `__init__` docstring must begin with this exact sentence, replacing
`ClassName` with the name of the class that defines the constructor:

```rst
Initialization of the :class:`ClassName` class.

:param int count: The number of samples.
```

Follow the sentence with a blank line, then parameter descriptions when
parameters exist. Do not insert additional prose between the opening
sentence and parameter fields. If needed to satisfy the 80-column limit,
wrap the sentence across source lines without changing its rendered text.
For constructors without parameters, omit parameter fields; document any
applicable exceptions after the summary and a blank line.

### Parameters

Document each parameter's meaning and accepted type. Explain defaults,
expected shapes, and accepted keyword options wherever relevant. Document
supported `*args` and `**kwargs` contents rather than leaving their meaning
unspecified.

Start every parameter description with a capital letter, including keyword
option descriptions within `**kwargs` documentation.

For a single accepted type, put the type inline:

```rst
:param int count: The number of samples.
```

For multiple accepted types, use a separate `:type:` field and separate types
with `|`:

```rst
:param value: The input data.
:type value: torch.Tensor | LabelTensor
```

Include `None` when accepted. State default values explicitly in descriptions,
using “Default is ...” and double backticks around literal values:

```rst
:param batch_size: The number of samples per batch. If ``None``, all samples
    form a single batch. Default is ``None``.
:type batch_size: int | None
```

Explain what `None` means when it selects special behavior. Do not infer that
a parameter accepts `None` merely because it is optional to pass.
Inspect the implementation and delegated helpers to explain the actual
fallback, inference, automatic selection, or disabled behavior triggered by
`None`. Merely stating that the default is `None` is insufficient when it
selects logic. The example above applies only to implementations that batch
all samples together for `None`.

### PINA types and framework hooks

Trace PINA call sites, validation, and inheritance before choosing a documented
type. Use the most specific PINA abstraction that describes the supported
contract, not merely a broader third-party superclass from an inherited
signature. Apply this to parameters and returns, including unused hook
arguments; names such as `pl_module` or `__` do not determine their types.

For callbacks invoked through `pina.Trainer`, document the solver argument as
`pina.solver.BaseSolver`, including when it is named `pl_module` or `__`.
PINA's trainer validates a `BaseSolver` and passes it to Lightning's loops;
its inheritance from `lightning.pytorch.LightningModule` does not make the
broader framework type the appropriate PINA-facing description. Document the
trainer as `pina.Trainer` when PINA-specific trainer behavior is required.
Preserve narrower solver requirements where verified, such as
`pina.solver.PhysicsInformedSingleModelSolver` for refinement callbacks.

Do not narrow a genuinely framework-generic API solely because it lives in
PINA, and do not name a concrete solver when a PINA base class or interface
captures the supported contract. Verify the actual requirements instead.

### Container types

Specify element types for lists and tuples, and key and value types for
dictionaries, whenever they can be verified. Apply this to parameter types,
return types, nested containers, and keyword options. Use Python generic
syntax, for example `list[int]` and `dict[str, LabelTensor]`.

Distinguish fixed-length from variable-length tuples:

- `tuple[float]`: Exactly one float.
- `tuple[float, float]`: Exactly two floats.
- `tuple[float, ...]`: A variable-length tuple of floats.
- `tuple[str, LabelTensor]`: A string followed by a LabelTensor.

Use unions for genuinely mixed element types and `Any` only when elements
are unconstrained or their types cannot be established. Never invent
element types. A container with one accepted type remains inline, such as
`:param list[int] indices: The selected indices.` A union of container
types uses a separate `:type:` field as usual.

### Exceptions

Document explicit errors and delegated validation errors relevant to callers.
Inspect called validators to verify the exception class and triggering
condition. Do not list every possible runtime error, such as incidental
out-of-memory failures, or invent validation that does not exist.

Use one field per relevant exception condition, after parameters and before
returns:

```rst
:raises ValueError: If the input and target sample counts differ.
```

### Returns

Use separate description and type fields:

```rst
:return: The computed residual.
:rtype: torch.Tensor | LabelTensor
```

Describe relevant shapes, structure, and meaning. If the result can be `None`
as well as another type, include `None` in `:rtype:` and explain when it is
returned. If the return type is only `None`, omit both `:return:` and
`:rtype:`. This includes constructors and in-place methods returning nothing.

### Cross-references and inline literals

- Use Sphinx roles for documented Python objects: ``:class:`LabelTensor` ``,
  ``:meth:`CartesianDomain.sample` ``, and ``:func:`package.function` ``.
  Use actual, resolvable object names in the docstrings you write; qualify
  names when needed to avoid ambiguity.
- Use double backticks for parameter names, values, dictionary keys, shapes,
  and code expressions, such as ``batch_size``, ``None``, ``"x"``,
  ``(n_samples, n_variables)``, and ``x.shape``.

### Mathematics

Use `:math:` for inline mathematics and `.. math::` for displayed equations.
Use a raw docstring (`r"""..."""`) when LaTeX contains backslashes, so Python
does not interpret them as escape sequences.

```python
r"""
Describe the residual for :math:`u` on the domain.

.. math::

    \mathcal{R}(u) = \frac{\partial u}{\partial t} - \Delta u
"""
```

Only include equations that accurately describe the implementation. When
documenting equation code, consult [define-equations](../define-equations/SKILL.md)
for relevant PINA conventions without changing the implementation.

### Notes and warnings

Explain relevant side effects in the description. Use `.. note::` for important
qualifications and `.. warning::` for potentially surprising behavior. Leave a
blank line after the directive and indent its contents:

```rst
.. note::

    The total number of points depends on the sampling mode.

.. warning::

    This operation modifies the input tensor in place.
```

These are formatting examples; include such statements only when true for
the documented object.

### Examples

Require examples only where they help clarify usage, non-obvious behavior,
shapes, or interactions. Do not add examples mechanically to every public
class or trivial method.

Use the existing `:Example:` section with indented `>>>` examples. Include
necessary imports or setup, and avoid fabricated outputs, especially for
random operations. Check examples against the current API.

```rst
:Example:

    >>> from pina.domain import CartesianDomain
    >>> domain = CartesianDomain({"x": [0.0, 1.0], "y": 1.0})
    >>> points = domain.sample(n=3, mode="grid")
```

## Step 4 — Adapt a complete template

The following self-contained example illustrates module, class, constructor,
property, helper, and method docstrings. It is a template, not a proposed PINA
API. Adapt descriptions and fields to the actual implementation.

```python
"""Utilities for scaling sample counts."""


class SampleCounter:
    """
    Store a sample count and compute scaled counts without modifying it.

    :Example:

        >>> counter = SampleCounter(3)
        >>> counter.scaled(2)
        6
    """

    def __init__(self, count):
        """
        Initialization of the :class:`SampleCounter` class.

        :param int count: The non-negative number of samples.
        :raises TypeError: If ``count`` is not an integer.
        :raises ValueError: If ``count`` is negative.
        """
        self._validate_count(count)
        self._count = count

    @staticmethod
    def _validate_count(count):
        """
        Validate a sample count.

        :param int count: The number of samples to validate.
        :raises TypeError: If ``count`` is not an integer.
        :raises ValueError: If ``count`` is negative.
        """
        if not isinstance(count, int):
            raise TypeError("count must be an integer")
        if count < 0:
            raise ValueError("count must be non-negative")

    @property
    def count(self):
        """
        Return the stored sample count.

        :return: The non-negative number of samples.
        :rtype: int
        """
        return self._count

    def scaled(self, factor=1):
        """
        Compute a scaled count without changing the stored count.

        :param int factor: The non-negative multiplier. Default is ``1``.
        :raises TypeError: If ``factor`` is not an integer.
        :raises ValueError: If ``factor`` is negative.
        :return: The stored count multiplied by ``factor``.
        :rtype: int
        """
        self._validate_count(factor)
        return self._count * factor
```

Both `__init__` and `scaled` document validation errors delegated to
`_validate_count`. The constructor and validator omit return fields because
they return only `None`.

## Step 5 — Review and verify

- Compare every documented type, default, shape, side effect, keyword option,
  return value, and exception condition against the implementation.
- Check that every docstring edit addresses an identified factual error,
  required omission, or convention violation. Keep corrections targeted and
  confirm already compliant and unrelated docstrings remain unchanged.
- Verify every changed docstring's physical source lines are at most 80
  characters, including indentation, examples, and delimiters. Use a
  source-level check over AST docstring line ranges; dedented text can hide
  overlong lines. Report the check result and fix violations before delivery.
- Compare corresponding interface, base-class, and concrete-class contracts
  for consistent wording without concealing logical differences.
- Verify hook argument and return types against PINA call sites and validation,
  rather than copying third-party superclass types. Confirm callback solver
  arguments use `pina.solver.BaseSolver` or a verified narrower PINA contract
  where appropriate, even when the argument is unused.
- Check that references resolve and examples use existing APIs. Run focused
  examples or documentation checks when available and report what was checked.
- Require edited docstrings to support PINA's documentation build without
  Sphinx warnings or errors. From the repository root, run the CI-equivalent
  command:

  ```bash
  make -C docs html SPHINXOPTS+='-W'
  ```

  If documentation dependencies are missing, explicitly ask the user permission
  to install them with `python3 -m pip install '.[doc]'`.
  Fix build warnings and errors caused by the edited docstrings without
  suppressing diagnostics or changing Sphinx configuration. Report unrelated
  failures or environment blockers separately; do not expand the approved
  scope to fix them. State whether the build passed, failed, or could not run,
  and do not claim successful verification unless it passed.
- Inspect the diff to ensure that documentation-only changes have not altered
  implementation code. Do not fix unrelated behavior while documenting it.
- Confirm all modified Python files are inside `pina/`, and that only actual
  module, class, and function docstrings changed. Where possible, compare
  parsed ASTs before and after with docstrings and source positions removed.
- Ask the user about unresolved intent instead of documenting assumed behavior.

## Checklist

- [ ] The documentation build passed with
      `make -C docs html SPHINXOPTS+='-W'`, or failures/blockers were explicitly
      reported without claiming successful verification.
- [ ] Only docstrings of Python files inside `pina/` were modified.
- [ ] Every changed docstring addresses a specific, identified issue;
      already compliant and unrelated docstrings remain unchanged.
- [ ] Every scoped module, class, constructor, method, property, and helper has
      the appropriate docstring.
- [ ] A short summary precedes details and fields, separated by a blank line.
- [ ] Constructors start with ``Initialization of the :class:`ClassName` class.``
      and a blank line before parameter descriptions.
- [ ] Every physical docstring source line is at most 80 characters,
      including indentation, examples, mathematics, types, and delimiters.
- [ ] Sphinx/reStructuredText is used instead of Google/NumPy sections or Markdown.
- [ ] Parameters have descriptions and types; `self` and `cls` are omitted.
- [ ] Every parameter description starts with a capital letter.
- [ ] Single types are inline; unions use a separate `:type:` field with `|`.
- [ ] Container element, key, and value types are specified where verifiable;
      fixed-length and variable-length tuples use accurate generic syntax.
- [ ] Parameter and return types reflect the verified PINA contract, including
      unused callback arguments, rather than unnecessarily broad framework types.
- [ ] Accepted `None` values, defaults, shapes, keyword options, and side effects
      are explained where relevant.
- [ ] Defaults of `None` explain the specific logic they trigger, if any.
- [ ] Interface, base-class, and concrete-class docstrings are consistent
      wherever behavior agrees, with actual logical differences preserved.
- [ ] Fields are ordered parameters → raises → returns, with indented continuations.
- [ ] Explicit and delegated validation errors state their actual conditions;
      incidental runtime errors are not exhaustively listed.
- [ ] Returns have separate `:return:` and `:rtype:` fields, unless only `None`
      is returned, in which case both are omitted.
- [ ] Cross-references, inline literals, mathematics, notes, and warnings use
      the required Sphinx syntax where applicable.
- [ ] Helpful examples use `:Example:` and indented `>>>` lines and match the API.
- [ ] Documentation describes current behavior, and implementation is unchanged.
