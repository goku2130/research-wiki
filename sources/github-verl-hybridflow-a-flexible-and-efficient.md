---
id: github:verl-hybridflow-a-flexible-and-efficient
type: web
title: 'verl: HybridFlow - A Flexible and Efficient RL Post-Training Framework'
url: https://github.com/verl-project/verl
retrieved: '2026-07-12'
maturity: comprehensive
topic: distributed-rl-training
---

The provided source is a GitHub repository landing page for **verl**, which implements **HybridFlow**, a framework described as a "Flexible and Efficient RL Post-Training Framework." Because the source text consists of repository metadata, file listings, and commit histories rather than a technical paper or documentation, the available technical details are limited to project infrastructure and repository maintenance.

### Core Problem
The primary objective of the project is to provide a flexible and efficient framework for the post-training phase of Reinforcement Learning (RL). On an operational level, the source identifies a specific problem regarding the management of "shared agent skills" across multiple frameworks (specifically Claude and Codex). The previous use of file-level symlinks was insufficient because it prevented supporting files—such as references and helper scripts—from flowing to every framework without requiring individual "per-file plumbing."

### Method: Agent Skill Infrastructure Restructuring
The source details a technical recipe for transitioning the agent skill discovery system from a file-symlink layout to a directory-symlink layout. This ensures that any supporting files added alongside a skill are automatically available across all integrated frameworks.

**Step-by-Step Implementation:**
1. **Canonical Content Migration**: Move the primary skill content from a flat file structure (`.agent/skills/<name>.md`) to a package-based layout (`.agent/skills/<name>/SKILL.md`).
2. **Symlink Conversion**: Replace existing file-level symlinks in the `.claude/skills/` and `.codex/skills/` directories with directory-level symlinks.
3. **Path Mapping**: Configure these new directory symlinks (e.g., `.claude/skills/issue`) to point to the canonical `.agent` skill directory (e.g., `../../.agent/skills/issue`).
4. **Documentation Update**: Update the `docs/contributing/editing-agent-instructions.md` file to document the new canonical skill directory and variant directory-symlink convention for future contributions.

**Verification Process:**
* **Hook Validation**: Execution of `pre-commit run --all-files --show-diff-on-failure --color=always` to ensure all pre-commit hooks pass.
* **Topology Verification**: Manual verification of the symlink paths to ensure they resolve correctly to directories containing `SKILL.md`.
* **Integrity Check**: Spot-checking the contents of `SKILL.md` to confirm that moves were "100% similarity renames" and no content was altered.

### Technical Integrations and Fixes
* **Code Assistance**: The framework integrates a Gemini code assistant configuration. The configuration includes a "high" threshold setting intended to reduce the volume of review comments.
* **Model Support**: Recent updates include technical fixes for `vocab_weights` device and shard handling specifically for the `qwen3_vl` model.

### Key Formulas, Quantitative Results, and Limitations
The provided source text does not contain any LaTeX formulas, quantitative performance metrics, numerical results, or stated limitations regarding the RL performance or efficiency of the HybridFlow framework.
