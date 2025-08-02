#!/usr/bin/env python3
"""
Advanced Socratic Agentic RAG System
Combines version 5.2 functionality with full multi-agent implementation
Compatible with free Claude version
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProjectContext:
    """Stores comprehensive project context information"""
    goals: List[str]
    requirements: List[str]
    tech_stack: List[str]
    constraints: List[str]
    team_structure: str
    language_preferences: List[str]
    deployment_target: str
    code_style: str
    phase: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectContext':
        return cls(**data)


@dataclass
class KnowledgeEntry:
    """Individual knowledge base entry with embedding"""
    content: str
    category: str
    phase: str
    keywords: List[str]
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntry':
        if 'embedding' in data and data['embedding'] is not None:
            data['embedding'] = np.array(data['embedding'])
        return cls(**data)


class SimpleEmbedding:
    """Simple embedding system for free Claude version compatibility"""

    def __init__(self):
        self.vocab = {}
        self.vocab_size = 1000  # Limited vocabulary for simplicity

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate simple bag-of-words embedding"""
        words = text.lower().split()
        embedding = np.zeros(self.vocab_size)

        for word in words:
            if word not in self.vocab:
                if len(self.vocab) < self.vocab_size:
                    self.vocab[word] = len(self.vocab)
                else:
                    continue

            if word in self.vocab:
                embedding[self.vocab[word]] = 1.0

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


class BaseAgent:
    """Base class for all agents"""

    def __init__(self, name: str, role: str, personality: str):
        self.name = name
        self.role = role
        self.personality = personality
        self.conversation_history = []

    def add_to_history(self, message: str, sender: str = "user"):
        """Add message to conversation history"""
        self.conversation_history.append({
            "sender": sender,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })

    def get_context_summary(self) -> str:
        """Get summary of recent conversation"""
        if not self.conversation_history:
            return "No previous conversation."

        recent = self.conversation_history[-3:]
        summary = "Recent conversation:\n"
        for entry in recent:
            summary += f"{entry['sender']}: {entry['message']}\n"
        return summary


class SocraticAgent(BaseAgent):
    """Main Socratic questioning agent - like Socrates"""

    def __init__(self):
        super().__init__(
            "Socrates",
            "Primary Questioner",
            "Asks probing questions to help users discover solutions themselves"
        )
        self.question_types = [
            "clarification", "assumption", "evidence", "perspective",
            "implication", "meta"
        ]

    def generate_question(self, context: ProjectContext, user_input: str, knowledge_base: List[KnowledgeEntry]) -> str:
        """Generate Socratic question based on context and input"""
        phase = context.phase.lower()

        if phase == "discovery":
            return self._discovery_questions(user_input, context)
        elif phase == "analysis":
            return self._analysis_questions(user_input, context)
        elif phase == "design":
            return self._design_questions(user_input, context)
        elif phase == "implementation":
            return self._implementation_questions(user_input, context)
        else:
            return self._general_questions(user_input, context)

    def _discovery_questions(self, user_input: str, context: ProjectContext) -> str:
        """Questions for discovery phase"""
        questions = [
            "What specific problem are you trying to solve for your users?",
            "Who is your target audience, and what are their current pain points?",
            "What would success look like for this project?",
            "What assumptions are you making about your users' needs?",
            "How will you measure whether your solution actually works?",
            "What similar solutions already exist, and why aren't they sufficient?",
            "What constraints or limitations should we consider from the start?"
        ]

        # Simple selection based on context completeness
        if not context.goals:
            return questions[0]
        elif not context.requirements:
            return questions[1]
        elif len(context.goals) < 2:
            return questions[2]
        else:
            return questions[3]

    def _analysis_questions(self, user_input: str, context: ProjectContext) -> str:
        """Questions for analysis phase"""
        questions = [
            "What are the core challenges in implementing this solution?",
            "Which parts of this project carry the highest risk?",
            "What dependencies might cause problems down the road?",
            "How will different components interact with each other?",
            "What performance requirements should we consider?",
            "What security considerations are important for your use case?",
            "How might your requirements change as the project grows?"
        ]

        if not context.tech_stack:
            return "What technologies or tools are you considering, and why?"
        elif not context.constraints:
            return questions[0]
        else:
            return questions[1]

    def _design_questions(self, user_input: str, context: ProjectContext) -> str:
        """Questions for design phase"""
        questions = [
            "How will users interact with your system?",
            "What's the simplest version that would still provide value?",
            "How will you handle errors and edge cases?",
            "What data will flow through your system, and how?",
            "How will you structure your code for maintainability?",
            "What testing strategy will ensure your solution works reliably?",
            "How will you deploy and monitor your application?"
        ]

        return questions[min(len(context.requirements), len(questions) - 1)]

    def _implementation_questions(self, user_input: str, context: ProjectContext) -> str:
        """Questions for implementation phase"""
        questions = [
            "What's the most critical piece to implement first?",
            "How will you validate each component as you build it?",
            "What development workflow will help you stay organized?",
            "How will you handle configuration and environment setup?",
            "What documentation will help others understand your code?",
            "How will you manage version control and releases?",
            "What's your plan for gathering user feedback?"
        ]

        return questions[0] if len(context.goals) < 3 else questions[1]

    def _general_questions(self, user_input: str, context: ProjectContext) -> str:
        """General probing questions"""
        return "What specific aspect would you like to explore further?"


class AnalystAgent(BaseAgent):
    """Analysis and synthesis agent - like Theaetetus"""

    def __init__(self):
        super().__init__(
            "Theaetetus",
            "Analyst",
            "Analyzes responses and synthesizes insights"
        )

    def analyze_response(self, user_response: str, context: ProjectContext) -> Dict[str, Any]:
        """Analyze user response and extract insights"""
        analysis = {
            "key_points": self._extract_key_points(user_response),
            "implications": self._identify_implications(user_response, context),
            "missing_info": self._identify_gaps(user_response, context),
            "next_focus": self._suggest_next_focus(user_response, context)
        }
        return analysis

    def _extract_key_points(self, response: str) -> List[str]:
        """Extract key points from user response"""
        # Simple keyword-based extraction
        key_indicators = [
            "want to", "need to", "must", "should", "will",
            "problem", "solution", "challenge", "goal",
            "user", "customer", "client", "team"
        ]

        sentences = response.split('.')
        key_points = []

        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in key_indicators):
                key_points.append(sentence)

        return key_points[:3]  # Return top 3

    def _identify_implications(self, response: str, context: ProjectContext) -> List[str]:
        """Identify implications of the user's response"""
        implications = []

        # Technical implications
        if any(tech in response.lower() for tech in ["web", "api", "database", "mobile"]):
            implications.append("This suggests a technical architecture with multiple components")

        # Scale implications
        if any(word in response.lower() for word in ["many", "scale", "users", "traffic"]):
            implications.append("Scalability and performance will be important considerations")

        # Team implications
        if any(word in response.lower() for word in ["team", "collaborate", "multiple"]):
            implications.append("Team coordination and communication will be critical")

        return implications

    def _identify_gaps(self, response: str, context: ProjectContext) -> List[str]:
        """Identify information gaps"""
        gaps = []

        if not context.tech_stack and "technology" not in response.lower():
            gaps.append("Technical stack not defined")

        if not context.constraints and "budget" not in response.lower() and "time" not in response.lower():
            gaps.append("Project constraints unclear")

        if not context.deployment_target and "deploy" not in response.lower():
            gaps.append("Deployment strategy not specified")

        return gaps

    def _suggest_next_focus(self, response: str, context: ProjectContext) -> str:
        """Suggest what to focus on next"""
        if context.phase == "discovery" and len(context.goals) < 2:
            return "Clarify project goals and user needs"
        elif context.phase == "discovery" and not context.requirements:
            return "Define specific requirements"
        elif context.phase == "analysis" and not context.tech_stack:
            return "Explore technology options"
        elif context.phase == "design" and not context.constraints:
            return "Identify project constraints"
        else:
            return "Dive deeper into implementation details"


class ContextAgent(BaseAgent):
    """Context management and memory agent - like Plato"""

    def __init__(self):
        super().__init__(
            "Plato",
            "Context Keeper",
            "Maintains project context and provides relevant information"
        )
        self.embedding_system = SimpleEmbedding()

    def update_context(self, context: ProjectContext, user_response: str, analysis: Dict[str, Any]) -> ProjectContext:
        """Update project context based on user response and analysis"""
        # Extract and add goals
        if "goal" in user_response.lower() or "want" in user_response.lower():
            new_goals = self._extract_goals(user_response)
            context.goals.extend(new_goals)
            context.goals = list(set(context.goals))  # Remove duplicates

        # Extract and add requirements
        if "need" in user_response.lower() or "require" in user_response.lower():
            new_requirements = self._extract_requirements(user_response)
            context.requirements.extend(new_requirements)
            context.requirements = list(set(context.requirements))

        # Extract tech stack
        tech_keywords = ["python", "javascript", "react", "django", "flask", "node", "sql", "nosql"]
        for keyword in tech_keywords:
            if keyword in user_response.lower() and keyword not in context.tech_stack:
                context.tech_stack.append(keyword)

        # Extract constraints
        constraint_keywords = ["budget", "time", "deadline", "resource", "limitation"]
        if any(keyword in user_response.lower() for keyword in constraint_keywords):
            new_constraints = self._extract_constraints(user_response)
            context.constraints.extend(new_constraints)
            context.constraints = list(set(context.constraints))

        # Update timestamp
        context.timestamp = datetime.now().isoformat()

        return context

    def _extract_goals(self, text: str) -> List[str]:
        """Extract goals from text"""
        goals = []
        sentences = text.split('.')
        goal_indicators = ["want to", "goal is", "aim to", "objective", "purpose"]

        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in goal_indicators):
                goals.append(sentence.strip())

        return goals

    def _extract_requirements(self, text: str) -> List[str]:
        """Extract requirements from text"""
        requirements = []
        sentences = text.split('.')
        req_indicators = ["need", "require", "must have", "should have", "necessary"]

        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in req_indicators):
                requirements.append(sentence.strip())

        return requirements

    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraints from text"""
        constraints = []
        sentences = text.split('.')
        constraint_indicators = ["budget", "deadline", "limited", "constraint", "restriction"]

        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in constraint_indicators):
                constraints.append(sentence.strip())

        return constraints

    def get_relevant_context(self, query: str, context: ProjectContext) -> str:
        """Get relevant context information for a query"""
        relevant_info = []

        query_lower = query.lower()

        # Add relevant goals
        if "goal" in query_lower or "objective" in query_lower:
            if context.goals:
                relevant_info.append(f"Goals: {', '.join(context.goals)}")

        # Add relevant requirements
        if "requirement" in query_lower or "need" in query_lower:
            if context.requirements:
                relevant_info.append(f"Requirements: {', '.join(context.requirements)}")

        # Add relevant tech stack
        if "tech" in query_lower or "technology" in query_lower:
            if context.tech_stack:
                relevant_info.append(f"Technologies: {', '.join(context.tech_stack)}")

        # Add relevant constraints
        if "constraint" in query_lower or "limitation" in query_lower:
            if context.constraints:
                relevant_info.append(f"Constraints: {', '.join(context.constraints)}")

        return "\n".join(relevant_info) if relevant_info else "No specific context available."


class KnowledgeAgent(BaseAgent):
    """Knowledge base management agent"""

    def __init__(self):
        super().__init__(
            "Aristotle",
            "Knowledge Keeper",
            "Manages and retrieves relevant knowledge"
        )
        self.embedding_system = SimpleEmbedding()
        self.knowledge_base = self._initialize_knowledge_base()

    def _initialize_knowledge_base(self) -> List[KnowledgeEntry]:
        """Initialize knowledge base with software development best practices"""
        knowledge_entries = [
            # Discovery phase
            KnowledgeEntry(
                "Start with user research to understand real problems before building solutions",
                "user_research", "discovery", ["user", "research", "problem", "solution"]
            ),
            KnowledgeEntry(
                "Define clear success metrics and acceptance criteria early in the project",
                "planning", "discovery", ["metrics", "success", "criteria", "planning"]
            ),
            KnowledgeEntry(
                "Validate assumptions through prototypes and user feedback before full development",
                "validation", "discovery", ["prototype", "validation", "feedback", "assumptions"]
            ),

            # Analysis phase
            KnowledgeEntry(
                "Break complex problems into smaller, manageable components",
                "decomposition", "analysis", ["complexity", "decomposition", "components", "manageable"]
            ),
            KnowledgeEntry(
                "Consider non-functional requirements like performance, security, and scalability",
                "requirements", "analysis", ["performance", "security", "scalability", "non-functional"]
            ),
            KnowledgeEntry(
                "Identify and mitigate risks early in the development process",
                "risk_management", "analysis", ["risk", "mitigation", "early", "development"]
            ),

            # Design phase
            KnowledgeEntry(
                "Design for maintainability with clear separation of concerns",
                "architecture", "design", ["maintainability", "separation", "concerns", "clean"]
            ),
            KnowledgeEntry(
                "Choose technologies based on project requirements, not personal preference",
                "technology_selection", "design", ["technology", "requirements", "selection", "appropriate"]
            ),
            KnowledgeEntry(
                "Plan for testing at all levels: unit, integration, and system tests",
                "testing", "design", ["testing", "unit", "integration", "system", "quality"]
            ),

            # Implementation phase
            KnowledgeEntry(
                "Implement incrementally with frequent integration and testing",
                "development", "implementation", ["incremental", "integration", "testing", "iterative"]
            ),
            KnowledgeEntry(
                "Use version control effectively with meaningful commit messages",
                "version_control", "implementation", ["version", "control", "git", "commits", "collaboration"]
            ),
            KnowledgeEntry(
                "Document code and decisions for future maintainers",
                "documentation", "implementation", ["documentation", "code", "decisions", "maintainers"]
            )
        ]

        # Generate embeddings for all entries
        for entry in knowledge_entries:
            entry.embedding = self.embedding_system.get_embedding(entry.content + " " + " ".join(entry.keywords))

        return knowledge_entries

    def search_knowledge(self, query: str, context: ProjectContext, top_k: int = 3) -> List[KnowledgeEntry]:
        """Search knowledge base for relevant entries"""
        query_embedding = self.embedding_system.get_embedding(query)

        # Calculate similarities
        similarities = []
        for entry in self.knowledge_base:
            if entry.embedding is not None:
                similarity = cosine_similarity([query_embedding], [entry.embedding])[0][0]
                similarities.append((similarity, entry))

        # Sort by similarity and filter by phase if relevant
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Prefer entries from current phase
        phase_bonus = 0.1
        for i, (sim, entry) in enumerate(similarities):
            if entry.phase == context.phase.lower():
                similarities[i] = (sim + phase_bonus, entry)

        # Re-sort and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in similarities[:top_k]]


class SocraticRAGSystem:
    """Main orchestrator for the multi-agent Socratic RAG system"""

    def __init__(self):
        self.socratic_agent = SocraticAgent()
        self.analyst_agent = AnalystAgent()
        self.context_agent = ContextAgent()
        self.knowledge_agent = KnowledgeAgent()

        self.context = ProjectContext(
            goals=[], requirements=[], tech_stack=[], constraints=[],
            team_structure="", language_preferences=[], deployment_target="",
            code_style="", phase="discovery", timestamp=datetime.now().isoformat()
        )

        self.conversation_history = []
        self.max_history = 50

    def process_user_input(self, user_input: str) -> str:
        """Process user input through the multi-agent system"""
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })

        # Trim history if too long
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        # Agent collaboration process
        try:
            # 1. Analyst analyzes the user input
            analysis = self.analyst_agent.analyze_response(user_input, self.context)

            # 2. Context agent updates project context
            self.context = self.context_agent.update_context(self.context, user_input, analysis)

            # 3. Knowledge agent searches for relevant information
            relevant_knowledge = self.knowledge_agent.search_knowledge(user_input, self.context)

            # 4. Context agent provides relevant context
            contextual_info = self.context_agent.get_relevant_context(user_input, self.context)

            # 5. Socratic agent generates the next question
            next_question = self.socratic_agent.generate_question(self.context, user_input, relevant_knowledge)

            # 6. Prepare comprehensive response
            response = self._prepare_response(next_question, analysis, relevant_knowledge, contextual_info)

            # Add response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })

            # Check if we should advance to next phase
            self._check_phase_advancement()

            return response

        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            return "I apologize, but I encountered an error processing your input. Could you please rephrase your response?"

    def _prepare_response(self, question: str, analysis: Dict[str, Any],
                          knowledge: List[KnowledgeEntry], context_info: str) -> str:
        """Prepare comprehensive response from all agents"""
        response_parts = []

        # Add analysis insights if significant
        if analysis["key_points"]:
            response_parts.append("I see that you've mentioned: " + ", ".join(analysis["key_points"][:2]))

        # Add relevant knowledge if applicable
        if knowledge and len(knowledge) > 0:
            best_knowledge = knowledge[0]
            if any(keyword in question.lower() for keyword in best_knowledge.keywords):
                response_parts.append(f"💡 Consider this: {best_knowledge.content}")

        # Add the main Socratic question
        response_parts.append(question)

        # Add implications or next steps if relevant
        if analysis["implications"]:
            response_parts.append(f"This might mean: {analysis['implications'][0]}")

        return "\n\n".join(response_parts)

    def _check_phase_advancement(self):
        """Check if we should advance to the next development phase"""
        current_phase = self.context.phase.lower()

        if current_phase == "discovery":
            if (len(self.context.goals) >= 2 and len(self.context.requirements) >= 2):
                self.context.phase = "analysis"
                logger.info("Advanced to Analysis phase")

        elif current_phase == "analysis":
            if (len(self.context.tech_stack) >= 1 and len(self.context.constraints) >= 1):
                self.context.phase = "design"
                logger.info("Advanced to Design phase")

        elif current_phase == "design":
            if (len(self.context.goals) >= 3 and len(self.context.requirements) >= 3):
                self.context.phase = "implementation"
                logger.info("Advanced to Implementation phase")

    def get_project_summary(self) -> str:
        """Generate a summary of the current project context"""
        summary_parts = [
            f"🎯 **Project Phase**: {self.context.phase.title()}",
            "",
            f"**Goals** ({len(self.context.goals)}):"
        ]

        for i, goal in enumerate(self.context.goals, 1):
            summary_parts.append(f"  {i}. {goal}")

        if not self.context.goals:
            summary_parts.append("  (No goals defined yet)")

        summary_parts.extend([
            "",
            f"**Requirements** ({len(self.context.requirements)}):"
        ])

        for i, req in enumerate(self.context.requirements, 1):
            summary_parts.append(f"  {i}. {req}")

        if not self.context.requirements:
            summary_parts.append("  (No requirements defined yet)")

        summary_parts.extend([
            "",
            f"**Technology Stack**: {', '.join(self.context.tech_stack) if self.context.tech_stack else 'Not specified'}",
            f"**Constraints**: {', '.join(self.context.constraints) if self.context.constraints else 'None identified'}",
            f"**Team Structure**: {self.context.team_structure if self.context.team_structure else 'Not specified'}",
            "",
            f"**Next Steps**: Continue exploring {self.context.phase.lower()} phase considerations"
        ])

        return "\n".join(summary_parts)

    def save_session(self, filename: str = "socratic_session.json"):
        """Save current session to file"""
        session_data = {
            "context": self.context.to_dict(),
            "conversation_history": self.conversation_history,
            "timestamp": datetime.now().isoformat()
        }

        try:
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2)
            logger.info(f"Session saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving session: {e}")

    def load_session(self, filename: str = "socratic_session.json"):
        """Load session from file"""
        try:
            with open(filename, 'r') as f:
                session_data = json.load(f)

            self.context = ProjectContext.from_dict(session_data["context"])
            self.conversation_history = session_data["conversation_history"]
            logger.info(f"Session loaded from {filename}")

        except FileNotFoundError:
            logger.info("No previous session found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading session: {e}")


def main():
    """Main function to run the Socratic RAG system"""
    print("🤖 Advanced Socratic Counselor for Project Development")
    print("=" * 60)
    print("This system uses multiple AI agents to guide you through project development:")
    print("• Socrates: Asks probing questions")
    print("• Theaetetus: Analyzes your responses")
    print("• Plato: Maintains project context")
    print("• Aristotle: Provides relevant knowledge")
    print()
    print("Commands: 'summary' (project overview), 'save' (save session), 'quit' (exit)")
    print("=" * 60)

    # Initialize the system
    rag_system = SocraticRAGSystem()

    # Try to load previous session
    rag_system.load_session()

    # Start with initial question
    if not rag_system.conversation_history:
        initial_question = "What exactly do you want to achieve with this project? Think about the core problem you're trying to solve."
        print(f"\nAssistant: {initial_question}")
    else:
        print("\nPrevious session loaded. Continuing conversation...")
        print(f"\nCurrent phase: {rag_system.context.phase.title()}")

    # Main conversation loop
    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'end', 'bye']:
                rag_system.save_session()
                print("\nSession saved. Goodbye! 👋")
                break

            elif user_input.lower() == 'summary':
                print(f"\n{rag_system.get_project_summary()}")
                continue

            elif user_input.lower() == 'save':
                rag_system.save_session()
                print("Session saved successfully! ✅")
                continue

            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("• summary - Show project overview")
                print("• save - Save current session")
                print("• quit/exit/end - Exit the program")
                continue

            # Process user input through the agent system
            response = rag_system.process_user_input(user_input)
            print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Saving...")
            rag_system.save_session()
            print("Session saved. Goodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"\nI encountered an error: {e}")
            print("Let's continue - please try rephrasing your input.")


if __name__ == "__main__":
    main()
