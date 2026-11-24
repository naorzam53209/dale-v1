"""Reflection module for verbal reinforcement learning.

Based on Reflexion: Language Agents with Verbal Reinforcement Learning
https://arxiv.org/abs/2303.11366
"""

import json
import os
from enum import Enum
from typing import List, Optional
from pathlib import Path


# browser_click starts its result with this after the user approved a final submit (book, pay...)
SUBMITTED_MARKER = "SUBMITTED:"
# answers the harness writes itself when an attempt produced nothing usable
HARNESS_FAILURE_MARKERS = ("(the model gave an empty answer", "(stopped after")


class ReflexionStrategy(Enum):
    """Reflection strategies for verbal reinforcement learning."""
    NONE = 'none'
    LAST_ATTEMPT = 'last_attempt'
    REFLEXION = 'reflexion'
    LAST_ATTEMPT_AND_REFLEXION = 'last_attempt_and_reflexion'


class ReflectionStorage:
    """Persistent storage for reflections organized by task type."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.reflections_dir = self.data_dir / "reflections"
        self.reflections_dir.mkdir(parents=True, exist_ok=True)

    def _get_task_file(self, task_type: str) -> Path:
        """Get the reflection file for a specific task type."""
        return self.reflections_dir / f"{task_type}.jsonl"

    def save_reflection(self, task_type: str, reflection: str, context: dict) -> None:
        """Save a reflection with its context."""
        file_path = self._get_task_file(task_type)
        entry = {
            "reflection": reflection,
            "context": context,
            "timestamp": self._get_timestamp()
        }
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def load_reflections(self, task_type: str, limit: int = 5) -> List[str]:
        """Load the most recent reflections for a task type."""
        file_path = self._get_task_file(task_type)
        if not file_path.exists():
            return []

        reflections = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    reflections.append(entry["reflection"])

        return reflections[-limit:] if reflections else []

    def load_last_attempt(self, task_type: str) -> Optional[str]:
        """Load the most recent full attempt (scratchpad) for a task type."""
        # This would need to be stored separately if we want to persist full attempts
        # For now, return None as it's managed in-memory
        return None

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


class ReflectionEngine:
    """Engine for generating and managing reflections."""

    # Prompt templates for reflection
    REFLECTION_HEADER = (
        "You have attempted to answer the following request before. "
        "The following reflection(s) give a plan to avoid failing in the same way. "
        "Use them to improve your strategy.\n"
    )

    LAST_ATTEMPT_HEADER = (
        "You have attempted to handle the following request before. "
        "Below is the last attempt you made.\n"
    )

    # Generic reflection instruction (fallback) - optimized for 4B model
    REFLECT_INSTRUCTION = """You are an agent that improves through self-reflection.
You tried to complete a task but failed. Analyze what went wrong and how to fix it.

Previous attempt:
Request: {request}
Actions: {actions}
Result: {result}

In 1-2 sentences, identify the problem and suggest a fix:

Reflection:"""

    # Booking-specific reflection instruction - optimized for 4B model
    BOOKING_REFLECT_INSTRUCTION = """You are a booking assistant. You failed to complete a reservation.
Analyze the failure and suggest how to succeed next time.

Previous attempt:
Request: {request}
Actions: {actions}
Result: {result}

What went wrong? (site blocked, missing info, wrong time, captcha?)
How to fix it? (try different site, ask for details, different approach?)

In 1-2 sentences:

Reflection:"""

    # Research-specific reflection instruction - optimized for 4B model
    RESEARCH_REFLECT_INSTRUCTION = """You are a research analyst. You provided an answer but it could be better.
Analyze your research approach and suggest improvements.

Previous attempt:
Request: {request}
Actions: {actions}
Result: {result}

What was weak? (unreliable sources, missing verification, incomplete answer?)
How to improve? (better sources, cross-check, more details?)

In 1-2 sentences:

Reflection:"""

    def __init__(
        self,
        strategy: ReflexionStrategy = ReflexionStrategy.REFLEXION,
        storage: Optional[ReflectionStorage] = None,
        task_type: str = "default",
        agent_type: str = "default"
    ):
        self.strategy = strategy
        self.storage = storage or ReflectionStorage()
        self.task_type = task_type
        self.agent_type = agent_type  # "booking", "research", or "default"
        self.reflections: List[str] = []
        self.last_attempt: Optional[str] = None
        self.reflections_str = ""

    def set_strategy(self, strategy: ReflexionStrategy) -> None:
        """Update the reflection strategy."""
        self.strategy = strategy

    def set_task_type(self, task_type: str) -> None:
        """Update the task type for reflection storage."""
        self.task_type = task_type

    def set_agent_type(self, agent_type: str) -> None:
        """Update the agent type for reflection prompts."""
        self.agent_type = agent_type

    def load_stored_reflections(self, limit: int = 5) -> None:
        """Load previously stored reflections for the current task type."""
        if self.strategy in (ReflexionStrategy.REFLEXION, ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION):
            self.reflections = self.storage.load_reflections(self.task_type, limit)
            self.reflections_str = self._format_reflections()

    def detect_success(self, result: str) -> bool:
        """Detect if the task succeeded based on the result.

        For booking: success if confirmation patterns are found (e.g., "Booked:", "Confirmation #")
        For research: success if answer has sources and is substantive
        For default: always treat as success
        """
        if self.agent_type == "booking":
            # Booking success patterns
            success_patterns = [
                "booked:",
                "confirmation #",
                "reservation confirmed",
                "your table is reserved",
                "booking completed"
            ]
            result_lower = result.lower()
            return any(pattern.lower() in result_lower for pattern in success_patterns)
        elif self.agent_type == "research":
            # Research success: has sources OR is substantive (not just "I don't know")
            result_lower = result.lower()
            has_sources = "http" in result_lower or "source" in result_lower or "www." in result_lower
            is_substantive = len(result) > 50 and "don't know" not in result_lower and "cannot" not in result_lower
            # Success if it has sources OR is substantive (either is acceptable for research)
            return has_sources or is_substantive
        # Default always treat as success
        return True

    def _get_reflection_prompt(self) -> str:
        """Get the appropriate reflection prompt based on agent type."""
        if self.agent_type == "booking":
            return self.BOOKING_REFLECT_INSTRUCTION
        elif self.agent_type == "research":
            return self.RESEARCH_REFLECT_INSTRUCTION
        else:
            return self.REFLECT_INSTRUCTION

    def generate_reflection(
        self,
        request: str,
        actions: str,
        result: str,
        client,
        model: dict
    ) -> str:
        """Generate a reflection using the LLM."""
        from openai import OpenAI

        prompt_template = self._get_reflection_prompt()
        prompt = prompt_template.format(
            request=request,
            actions=actions,
            result=result
        )

        response = client.chat.completions.create(
            model=model["name"],
            temperature=model.get("temperature", 0.7),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200  # Reduced for 4B model optimization
        )

        reflection = (response.choices[0].message.content or "").strip()
        return reflection

    def update_reflections(
        self,
        request: str,
        actions: str,
        result: str,
        client,
        model: dict,
        reflect_after: str = "every_turn"
    ) -> None:
        """Update reflections based on the latest attempt.

        Args:
            request: The user's original request
            actions: Summary of actions taken
            result: The final result/answer
            client: OpenAI client for generating reflections
            model: Model configuration
            reflect_after: When to reflect ("every_turn" or "failure_only")
        """
        if self.strategy == ReflexionStrategy.NONE:
            return

        # Check if we should reflect based on success/failure
        if reflect_after == "failure_only":
            success = self.detect_success(result)
            if success:
                # Don't reflect on successful attempts when failure_only is set
                return

        if self.strategy == ReflexionStrategy.LAST_ATTEMPT:
            self.last_attempt = f"Request: {request}\nActions: {actions}\nResult: {result}"
            self.reflections_str = self._format_last_attempt()
        elif self.strategy == ReflexionStrategy.REFLEXION:
            reflection = self.generate_reflection(request, actions, result, client, model)
            self.reflections.append(reflection)
            self.storage.save_reflection(self.task_type, reflection, {"request": request})
            self.reflections_str = self._format_reflections()
        elif self.strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
            self.last_attempt = f"Request: {request}\nActions: {actions}\nResult: {result}"
            self.reflections_str = self._format_last_attempt()
            reflection = self.generate_reflection(request, actions, result, client, model)
            self.reflections.append(reflection)
            self.storage.save_reflection(self.task_type, reflection, {"request": request})
            self.reflections_str += "\n" + self._format_reflections(header="")

    def attempt_failed(self, result: str, outcome: Optional[dict] = None) -> bool:
        """Did this attempt fail badly enough that running it again could help?

        outcome comes from the harness: submitted (a final submit the user approved),
        out_of_steps, empty. For booking, an answer without a confirmation phrase is
        not a failure: availability checks, "book it here" links and questions back to
        the user are all correct answers. Only clear failures count there."""
        outcome = outcome or {}
        if outcome.get("submitted"):
            return False  # never redo a submitted booking: it could book twice
        if outcome.get("out_of_steps") or outcome.get("empty"):
            return True
        if result.strip().startswith(HARNESS_FAILURE_MARKERS):
            return True
        if self.agent_type == "booking":
            return False
        return not self.detect_success(result)

    def should_retry(self, result: str, max_retries: int = 3, current_attempt: int = 0,
                     outcome: Optional[dict] = None) -> bool:
        """Retry while attempts are left and this attempt failed (see attempt_failed)."""
        if current_attempt >= max_retries:
            return False
        return self.attempt_failed(result, outcome)

    def get_reflection_for_retry(self, request: str, actions: str, result: str, client, model: dict) -> str:
        """Generate a reflection specifically for retry context.

        This is a simplified version that returns just the reflection text
        without updating the stored reflections.
        """
        return self.generate_reflection(request, actions, result, client, model)

    def get_reflections_context(self) -> str:
        """Get the formatted reflections to inject into the agent's context."""
        return self.reflections_str

    def _format_reflections(self, header: Optional[str] = None) -> str:
        """Format reflections as a string."""
        if not self.reflections:
            return ""

        header = header or self.REFLECTION_HEADER
        return header + "Reflections:\n- " + "\n- ".join([r.strip() for r in self.reflections])

    def _format_last_attempt(self) -> str:
        """Format the last attempt as a string."""
        if not self.last_attempt:
            return ""

        return self.LAST_ATTEMPT_HEADER + self.last_attempt + "\n(END PREVIOUS ATTEMPT)\n"

    def reset(self) -> None:
        """Reset reflections for a new conversation."""
        self.reflections = []
        self.last_attempt = None
        self.reflections_str = ""
