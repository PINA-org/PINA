# Global Rules for All Skills

These rules apply to **every skill** in this repository.

1. **No source code modification** — Never modify, create, or delete any source
   code files (`.py`, `.cpp`, `.js`, `.ts`, `.rs`, etc.) or any files under
   `pina/`, `tests/`, `docs/`, or similar source directories.

2. **Read-only guidance** — Skills exist solely to provide guidance, templates,
   and interactive Q&A. Output explanations, code snippets for the user to
   copy, and checklists — never write files to the user's project.

3. **Scope** — If the user asks to edit or write source files, politely decline
   and explain that this skill only provides guidance.

4. **Avoid redundant operator calls in equations** — Differential operators
   (`grad`, `div`, `laplacian`, etc.) are **not cached**: every call triggers a
   new backward pass through the autograd graph.  Always compute an operator
   once for all relevant components and extract what you need, rather than
   calling it repeatedly for individual components.

   **Bad** — 6 separate backward passes:
   ```python
   Ex_t = grad(output_, input_, components=["Ex"], d=["t"])
   Ey_t = grad(output_, input_, components=["Ey"], d=["t"])
   Ez_t = grad(output_, input_, components=["Ez"], d=["t"])
   Hx_t = grad(output_, input_, components=["Hx"], d=["t"])
   Hy_t = grad(output_, input_, components=["Hy"], d=["t"])
   Hz_t = grad(output_, input_, components=["Hz"], d=["t"])
   ```

   **Good** — 1 backward pass, 6 extractions:
   ```python
   out_t = grad(output_, input_, d=["t"])
   Ex_t = out_t.extract("dExdt")
   Ey_t = out_t.extract("dEydt")
   Ez_t = out_t.extract("dEzdt")
   Hx_t = out_t.extract("dHxdt")
   Hy_t = out_t.extract("dHydt")
   Hz_t = out_t.extract("dHzdt")
   ```

   This also applies to spatial gradients, divergences, and any other
   operator that can be computed for all components at once.

5. **Inference goes through the solver, not the raw model** — Always call
   `solver(input_tensor)` for evaluation at inference time,
   not `model(input_tensor)` or `solver.model(input_tensor)`.
   The solver handles label propagation and any post-processing internally.
   Only bypass the solver and call `model(input)` directly when there is a
   specific reason (e.g., multiple models with different forward passes at
   training vs. inference).

6. **Prefer solver defaults, acknowledge in chat** — Use the default values
   for solver parameters (optimizer, scheduler, loss, weighting, etc.) unless
   the user explicitly provides different values. Let the user know in chat
   which defaults are active so they're aware of what's running under the hood.

7. **Use float values in domain fixed-point variables** —
   `CartesianDomain({"t": 0})` or `EllipsoidDomain({"t": 0})` stores the value
   `0` as a Python `int`, causing `torch.full` to produce an int64 tensor. This
   later fails when `requires_grad_()` is called (int tensors don't support
   gradients). Always pass `float` values for any fixed-point variable.

   **Bad** — Produces int64 domain:
   ```python
   CartesianDomain({"t": 0})
   EllipsoidDomain({"t": 0})
   ```

   **Good** — Produces float32 domain:
   ```python
   CartesianDomain({"t": 0.0})
   EllipsoidDomain({"t": 0.0})
   ```

8. **Detach LabelTensors before converting to NumPy** —
   A `LabelTensor` with `requires_grad=True` cannot be passed to
   `matplotlib` or `numpy` directly — PyTorch raises
   `"Can't call numpy() on Tensor that requires grad"`. Always call
   `.detach().numpy()` on tensors that flow through a gradient computation
   before plotting or serialising.

   **Bad**:
   ```python
   plt.plot(theta, dtheta_dt)  # raises if requires_grad
   ```

   **Good**:
    ```python
    plt.plot(theta.detach().numpy(), dtheta_dt.detach().numpy())
    ```

9. **Don't call `problem.move_discretisation_into_conditions()` in user scripts** —
   The Trainer calls this internally before training — the user doesn't need
   to invoke it manually. Keep user-facing scripts simple:
   `problem.discretise_domain(...)` per domain and nothing more.

10. **Smoke-Test Training Runs** - Before delivering training code, run a short 
   smoke test using only a few epochs to confirm that the pipeline executes 
   successfully and does not fail. After the test passes, restore the final 
   configuration to the intended number of training epochs.
