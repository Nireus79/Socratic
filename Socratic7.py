import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from anthropic import Anthropic
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid
import shutil


class ProjectContext:
    """Manages project context and state"""

    def __init__(self, project_id: str = None):
        self.project_id = project_id or str(uuid.uuid4())
        self.name = "Unnamed Project"
        self.goals = []
        self.requirements = []
        self.tech_stack = []
        self.constraints = []
        self.team_structure = ""
        self.language_preference = ""
        self.deployment_target = ""
        self.code_style = ""
        self.phase = "discovery"  # discovery, analysis, design, implementation
        self.conversation_history = []
        self.created_at = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()

    def update_phase(self, new_phase: str):
        """Update project phase"""
        self.phase = new_phase
        self.last_updated = datetime.now().isoformat()

    def add_context_item(self, category: str, item: str):
        """Add item to specific context category"""
        if category == "goals":
            self.goals.append(item)
        elif category == "requirements":
            self.requirements.append(item)
        elif category == "tech_stack":
            self.tech_stack.append(item)
        elif category == "constraints":
            self.constraints.append(item)
        self.last_updated = datetime.now().isoformat()

    def to_dict(self):
        """Convert to dictionary for storage"""
        return {
            "project_id": self.project_id,
            "name": self.name,
            "goals": self.goals,
            "requirements": self.requirements,
            "tech_stack": self.tech_stack,
            "constraints": self.constraints,
            "team_structure": self.team_structure,
            "language_preference": self.language_preference,
            "deployment_target": self.deployment_target,
            "code_style": self.code_style,
            "phase": self.phase,
            "conversation_history": self.conversation_history,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary"""
        project = cls(data["project_id"])
        for key, value in data.items():
            setattr(project, key, value)
        return project


class VectorStore:
    """Vector database manager using ChromaDB"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Initialize collections
        self.knowledge_collection = self._get_or_create_collection("knowledge_base")
        self.conversation_collection = self._get_or_create_collection("conversations")
        self.project_collection = self._get_or_create_collection("projects")

        # Initialize with base knowledge if empty
        if self.knowledge_collection.count() == 0:
            self._initialize_knowledge_base()

    def _get_or_create_collection(self, name: str):
        """Get or create a collection"""
        try:
            return self.client.get_collection(name)
        except ValueError:
            return self.client.create_collection(name)

    def _initialize_knowledge_base(self):
        """Initialize knowledge base with software development best practices"""
        knowledge_entries = [
            ("Start with user needs and work backwards to technical solutions", "methodology",
             "Always prioritize understanding the actual problem before jumping to solutions"),
            ("Break large problems into smaller, manageable pieces", "methodology",
             "Complex projects become manageable when decomposed into smaller tasks"),
            ("Build the minimum viable version first, then iterate", "development",
             "Start simple and add complexity based on real feedback and needs"),
            ("Test early and often to catch problems quickly", "development",
             "Early testing prevents expensive fixes later in development"),
            ("Document decisions and assumptions for future reference", "documentation",
             "Clear documentation helps team members understand context and reasoning"),
            ("Consider scalability, security, and maintainability from the start", "architecture",
             "These concerns are harder to add later than to build in from the beginning"),
            ("Choose technologies your team knows well unless there's a compelling reason to change", "technology",
             "Team expertise is often more valuable than using the latest technology"),
            ("Plan for deployment and monitoring from day one", "operations",
             "Production considerations should influence development decisions early"),
            ("User feedback is more valuable than internal assumptions", "methodology",
             "Real user input trumps theoretical requirements"),
            ("Simple solutions are often better than complex ones", "architecture",
             "Complexity should be justified by clear benefits"),
            ("Automate repetitive tasks to reduce human error", "development",
             "Automation improves consistency and frees up time for creative work"),
            ("Version control everything, including documentation and configuration", "development",
             "Track all changes to understand evolution and enable rollbacks"),
        ]

        documents = []
        metadatas = []
        ids = []

        for i, (content, category, context) in enumerate(knowledge_entries):
            documents.append(f"{content}. {context}")
            metadatas.append({
                "category": category,
                "type": "knowledge",
                "content": content,
                "context": context
            })
            ids.append(f"knowledge_{i}")

        self.knowledge_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def add_knowledge_entry(self, content: str, category: str, context: str = ""):
        """Add new knowledge entry"""
        entry_id = f"knowledge_{uuid.uuid4()}"
        document = f"{content}. {context}" if context else content

        self.knowledge_collection.add(
            documents=[document],
            metadatas=[{
                "category": category,
                "type": "knowledge",
                "content": content,
                "context": context,
                "added_at": datetime.now().isoformat()
            }],
            ids=[entry_id]
        )

    def search_knowledge(self, query: str, category: str = None, n_results: int = 3) -> List[Dict]:
        """Search knowledge base"""
        where_clause = {"type": "knowledge"}
        if category:
            where_clause["category"] = category

        results = self.knowledge_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )

        return [{
            "content": metadata["content"],
            "category": metadata["category"],
            "context": metadata.get("context", ""),
            "distance": distance
        } for metadata, distance in zip(results["metadatas"][0], results["distances"][0])]

    def add_conversation(self, project_id: str, user_input: str, assistant_response: str, phase: str):
        """Add conversation exchange to vector store"""
        conversation_id = f"conv_{project_id}_{uuid.uuid4()}"

        # Combine user input and response for better semantic search
        document = f"User: {user_input} Assistant: {assistant_response}"

        self.conversation_collection.add(
            documents=[document],
            metadatas=[{
                "project_id": project_id,
                "user_input": user_input,
                "assistant_response": assistant_response,
                "phase": phase,
                "timestamp": datetime.now().isoformat(),
                "type": "conversation"
            }],
            ids=[conversation_id]
        )

    def search_conversations(self, query: str, project_id: str = None, phase: str = None, n_results: int = 5) -> List[
        Dict]:
        """Search conversation history"""
        where_clause = {"type": "conversation"}
        if project_id:
            where_clause["project_id"] = project_id
        if phase:
            where_clause["phase"] = phase

        try:
            results = self.conversation_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause
            )

            return [{
                "user_input": metadata["user_input"],
                "assistant_response": metadata["assistant_response"],
                "phase": metadata["phase"],
                "timestamp": metadata["timestamp"],
                "distance": distance
            } for metadata, distance in zip(results["metadatas"][0], results["distances"][0])]
        except Exception:
            return []

    def store_project(self, project: ProjectContext):
        """Store project data"""
        project_data = project.to_dict()
        document = (f"Project: {project.name} "
                    f"Goals: {', '.join(project.goals)} "
                    f"Requirements: {', '.join(project.requirements)} "
                    f"Tech: {', '.join(project.tech_stack)}")

        # Try to update existing, otherwise add new
        try:
            self.project_collection.upsert(
                documents=[document],
                metadatas=[{
                    "project_id": project.project_id,
                    "name": project.name,
                    "phase": project.phase,
                    "type": "project",
                    "last_updated": project.last_updated,
                    "data": json.dumps(project_data)
                }],
                ids=[f"project_{project.project_id}"]
            )
        except Exception:
            self.project_collection.add(
                documents=[document],
                metadatas=[{
                    "project_id": project.project_id,
                    "name": project.name,
                    "phase": project.phase,
                    "type": "project",
                    "last_updated": project.last_updated,
                    "data": json.dumps(project_data)
                }],
                ids=[f"project_{project.project_id}"]
            )

    def get_project(self, project_id: str) -> Optional[ProjectContext]:
        """Retrieve project by ID"""
        try:
            results = self.project_collection.get(
                ids=[f"project_{project_id}"],
                include=["metadatas"]
            )

            if results["metadatas"]:
                project_data = json.loads(results["metadatas"][0]["data"])
                return ProjectContext.from_dict(project_data)
        except Exception:
            pass
        return None

    def list_projects(self) -> List[Dict]:
        """List all projects"""
        try:
            results = self.project_collection.get(
                where={"type": "project"},
                include=["metadatas"]
            )

            return [{
                "project_id": metadata["project_id"],
                "name": metadata["name"],
                "phase": metadata["phase"],
                "last_updated": metadata["last_updated"]
            } for metadata in results["metadatas"]]
        except Exception:
            return []

    def delete_project(self, project_id: str) -> bool:
        """Delete project and related conversations"""
        try:
            # Delete project
            self.project_collection.delete(ids=[f"project_{project_id}"])

            # Delete related conversations
            conv_results = self.conversation_collection.get(
                where={"project_id": project_id},
                include=["ids"]
            )
            if conv_results["ids"]:
                self.conversation_collection.delete(ids=conv_results["ids"])

            return True
        except Exception:
            return False

    def search_similar_projects(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search for similar projects"""
        try:
            results = self.project_collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"type": "project"}
            )

            return [{
                "project_id": metadata["project_id"],
                "name": metadata["name"],
                "phase": metadata["phase"],
                "distance": distance
            } for metadata, distance in zip(results["metadatas"][0], results["distances"][0])]
        except Exception:
            return []


class SocraticRAG:
    """Enhanced Socratic RAG system with vector database"""

    def __init__(self, api_key: str, persist_directory: str = "./chroma_db"):
        self.client = Anthropic(api_key=api_key)
        self.vector_store = VectorStore(persist_directory)
        self.current_project = None
        self.users = {}
        self.current_user = None

        # Core Socratic questioning templates (from branch 5 logic)
        self.socratic_templates = {
            "discovery": [
                "What exactly do you want to achieve with this project?",
                "Who will be using this system and how?",
                "What problems are you trying to solve?",
                "What would success look like for this project?",
                "What constraints or limitations do you need to work within?"
            ],
            "analysis": [
                "What challenges do you anticipate with this approach?",
                "How does this compare to existing solutions?",
                "What are the most critical aspects to get right?",
                "What could go wrong and how would you handle it?",
                "What assumptions are you making?"
            ],
            "design": [
                "How would you break this down into smaller components?",
                "What would the user experience flow look like?",
                "How will different parts of your system communicate?",
                "What data will you need to store and how?",
                "How will you ensure your system can scale?"
            ],
            "implementation": [
                "What would you build first and why?",
                "How will you test that each part works correctly?",
                "What tools and technologies will you use?",
                "How will you deploy and maintain this system?",
                "What documentation will your team need?"
            ]
        }

        # Enhanced suggestions with vector search context
        self.suggestion_templates = {
            "discovery": [
                "Consider starting with: What specific pain point does this solve?",
                "Think about: Who would benefit most from this solution?",
                "Ask yourself: What would happen if this problem isn't solved?",
                "Reflect on: What similar solutions exist and what's missing?"
            ],
            "analysis": [
                "Consider: What are the technical risks and how severe are they?",
                "Think about: What resources (time, people, money) do you have?",
                "Ask yourself: What's the minimum viable version of this?",
                "Reflect on: What expertise do you need that you don't have?"
            ],
            "design": [
                "Consider: What are the core functions this system must perform?",
                "Think about: How will users interact with your system?",
                "Ask yourself: What's the simplest architecture that could work?",
                "Reflect on: What parts can you reuse or buy instead of building?"
            ],
            "implementation": [
                "Consider: What can you prototype quickly to test your assumptions?",
                "Think about: What's the riskiest part to build first?",
                "Ask yourself: How will you know if it's working correctly?",
                "Reflect on: What could you automate to save time later?"
            ]
        }

        self._load_users()

    def _load_users(self):
        """Load users from file if exists"""
        users_file = os.path.join(self.vector_store.persist_directory, "users.json")
        if os.path.exists(users_file):
            try:
                with open(users_file, 'r') as f:
                    self.users = json.load(f)
            except Exception:
                pass

    def _save_users(self):
        """Save users to file"""
        users_file = os.path.join(self.vector_store.persist_directory, "users.json")
        os.makedirs(self.vector_store.persist_directory, exist_ok=True)
        try:
            with open(users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception:
            pass

    def create_user(self, username: str) -> str:
        """Create new user"""
        user_id = str(uuid.uuid4())
        self.users[user_id] = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "projects": [],
            "preferences": {}
        }
        self._save_users()
        return user_id

    def delete_user(self, user_id: str) -> bool:
        """Delete user and all their projects"""
        if user_id not in self.users:
            return False

        # Delete all user's projects
        user_projects = self.users[user_id]["projects"].copy()
        for project_id in user_projects:
            self.vector_store.delete_project(project_id)

        # Delete user
        del self.users[user_id]
        if self.current_user == user_id:
            self.current_user = None

        self._save_users()
        return True

    def create_project(self, project_name: str) -> str:
        """Create new project"""
        project = ProjectContext()
        project.name = project_name

        # Store in vector database
        self.vector_store.store_project(project)

        if self.current_user:
            self.users[self.current_user]["projects"].append(project.project_id)
            self._save_users()

        return project.project_id

    def delete_project(self, project_id: str) -> bool:
        """Delete project"""
        # Remove from user's project list
        for user_id, user_data in self.users.items():
            if project_id in user_data["projects"]:
                user_data["projects"].remove(project_id)

        # Delete from vector store
        success = self.vector_store.delete_project(project_id)

        if self.current_project and self.current_project.project_id == project_id:
            self.current_project = None

        if success:
            self._save_users()

        return success

    def set_current_project(self, project_id: str):
        """Set current active project"""
        project = self.vector_store.get_project(project_id)
        if project:
            self.current_project = project

    def list_projects(self) -> List[Dict]:
        """List all projects"""
        return self.vector_store.list_projects()

    def generate_socratic_question(self, user_input: str) -> str:
        """Generate Socratic question using enhanced vector search"""
        if not self.current_project:
            return "What exactly do you want to achieve with this project?"

        # Handle "I don't know" responses with enhanced suggestions
        if self._is_uncertain_response(user_input):
            return self._generate_enhanced_suggestion(user_input)

        # Handle requests for suggestions
        if self._is_request_for_suggestions(user_input):
            return self._generate_enhanced_suggestion(user_input)

        # Search for relevant knowledge using vector similarity
        relevant_knowledge = self.vector_store.search_knowledge(
            user_input,
            category=None,  # Search all categories
            n_results=3
        )

        # Search for similar past conversations
        similar_conversations = self.vector_store.search_conversations(
            user_input,
            project_id=self.current_project.project_id,
            n_results=2
        )

        # Search for similar projects for additional context
        similar_projects = self.vector_store.search_similar_projects(user_input, n_results=2)

        # Build enhanced context
        context = self._build_project_context()
        knowledge_context = "\n".join([f"- {entry['content']}" for entry in relevant_knowledge[:2]])

        # Enhanced Socratic questioning prompt with vector context
        prompt = f"""You are a Socratic counselor helping a software developer think through their project. 
Use the Socratic method: ask ONE thoughtful question that helps them discover the next step.

Current project phase: {self.current_project.phase}
Project context: {context}
User's latest input: {user_input}

Relevant knowledge from similar situations:
{knowledge_context}

Guidelines:
- Ask exactly ONE question
- Be specific and actionable  
- Help them think deeper about their approach
- Focus on discovery through questioning, not giving answers
- Build on the relevant knowledge to ask more targeted questions
- If they seem stuck, gently guide them to consider a new angle
- Use insights from similar situations to ask better questions

Your question:"""

        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=150,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            question = response.content[0].text.strip()

            # Store conversation in vector database
            self.vector_store.add_conversation(
                self.current_project.project_id,
                user_input,
                question,
                self.current_project.phase
            )

            # Update conversation history
            self.current_project.conversation_history.append({
                "user": user_input,
                "assistant": question,
                "timestamp": datetime.now().isoformat(),
                "phase": self.current_project.phase
            })

            # Save updated project
            self.vector_store.store_project(self.current_project)

            return question

        except Exception as e:
            return f"I'd like to understand better - could you elaborate on that? (Error: {str(e)})"

    def _generate_enhanced_suggestion(self, user_input: str) -> str:
        """Generate enhanced suggestion using vector search"""
        if not self.current_project:
            return "What exactly do you want to achieve with this project?"

        # Search for relevant knowledge and past conversations
        relevant_knowledge = self.vector_store.search_knowledge(
            f"{user_input} {self.current_project.phase} suggestions help",
            n_results=2
        )

        similar_situations = self.vector_store.search_conversations(
            f"I don't know suggestions help {self.current_project.phase}",
            n_results=2
        )

        # Build context-aware suggestion
        phase = self.current_project.phase
        base_suggestions = self.suggestion_templates.get(phase, [])

        knowledge_insights = ""
        if relevant_knowledge:
            knowledge_insights = f"\nBased on similar situations: {relevant_knowledge[0]['content']}"

        if base_suggestions:
            import random
            base_suggestion = random.choice(base_suggestions)
            return (f"Here's a way to think about it: {base_suggestion}{knowledge_insights}"
                    f"\n\nWhich of these approaches feels most relevant to your situation?")

        return f"What aspect of this would you like to explore first?{knowledge_insights}"

    def _is_uncertain_response(self, user_input: str) -> bool:
        """Check if user response indicates uncertainty"""
        uncertain_phrases = [
            "i don't know", "not sure", "i'm not sure", "no idea",
            "don't know", "not certain", "i'm uncertain", "unsure", "confused"
        ]
        return any(phrase in user_input.lower() for phrase in uncertain_phrases)

    def _is_request_for_suggestions(self, user_input: str) -> bool:
        """Check if user is asking for suggestions"""
        suggestion_phrases = [
            "suggest", "suggestion", "what should i", "what would you",
            "any ideas", "help me think", "what do you think", "advice", "recommend"
        ]
        return any(phrase in user_input.lower() for phrase in suggestion_phrases)

    def _build_project_context(self) -> str:
        """Build project context string"""
        if not self.current_project:
            return ""

        context_parts = []
        if self.current_project.goals:
            context_parts.append(f"Goals: {', '.join(self.current_project.goals)}")
        if self.current_project.requirements:
            context_parts.append(f"Requirements: {', '.join(self.current_project.requirements)}")
        if self.current_project.tech_stack:
            context_parts.append(f"Tech Stack: {', '.join(self.current_project.tech_stack)}")
        if self.current_project.constraints:
            context_parts.append(f"Constraints: {', '.join(self.current_project.constraints)}")

        return " | ".join(context_parts) if context_parts else "New project - no context yet"

    def generate_code(self, requirements: str) -> str:
        """Code generation agent with vector-enhanced context"""
        if not self.current_project:
            return "Please create or select a project first."

        # Search for relevant code patterns and knowledge
        relevant_knowledge = self.vector_store.search_knowledge(
            f"{requirements} code implementation {' '.join(self.current_project.tech_stack)}",
            n_results=3
        )

        context = self._build_project_context()
        knowledge_context = "\n".join([f"- {entry['content']}: {entry['context']}" for entry in relevant_knowledge])

        prompt = f"""Generate code based on these requirements:
Requirements: {requirements}
Project Context: {context}
Tech Stack: {', '.join(self.current_project.tech_stack) if self.current_project.tech_stack else 'Not specified'}

Relevant best practices:
{knowledge_context}

Provide clean, well-commented code with explanations. Follow the best practices mentioned above."""

        try:
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text

        except Exception as e:
            return f"Code generation failed: {str(e)}"

    def add_knowledge(self, content: str, category: str, context: str = ""):
        """Add new knowledge entry to vector store"""
        self.vector_store.add_knowledge_entry(content, category, context)
        return "Knowledge entry added successfully."

    def search_project_insights(self, query: str) -> str:
        """Search for insights across projects and conversations"""
        # Search knowledge base
        knowledge_results = self.vector_store.search_knowledge(query, n_results=3)

        # Search conversations
        conversation_results = self.vector_store.search_conversations(query, n_results=3)

        # Search similar projects
        project_results = self.vector_store.search_similar_projects(query, n_results=2)

        insights = []

        if knowledge_results:
            insights.append("📚 Relevant Knowledge:")
            for result in knowledge_results:
                insights.append(f"  • {result['content']}")

        if conversation_results:
            insights.append("\n💬 Similar Past Discussions:")
            for result in conversation_results[:2]:
                insights.append(f"  • In {result['phase']} phase: {result['assistant_response'][:100]}...")

        if project_results:
            insights.append("\n🔍 Similar Projects:")
            for result in project_results:
                insights.append(f"  • {result['name']} (Phase: {result['phase']})")

        return "\n".join(insights) if insights else "No relevant insights found for this query."

    def show_menu(self) -> str:
        """Enhanced menu system with vector capabilities"""
        menu = """
╔═══════════════════════════════════════════╗
║        ENHANCED SOCRATIC RAG SYSTEM       ║
║                Version 7.1                ║
║       Ουδέν οίδα, ούτε διδάσκω τι,        ║
║            αλλά διαπορώ μόνον.            ║
╠═══════════════════════════════════════════╣
║ PROJECT MANAGEMENT:                       ║
║   1. Create new project                   ║
║   2. List projects                        ║
║   3. Select project                       ║
║   4. Delete project                       ║
║   5. Project summary                      ║
║   6. Search project insights              ║
║                                           ║
║ USER MANAGEMENT:                          ║
║   7. Create user                          ║
║   8. Delete user                          ║
║   9. Switch user                          ║
║                                           ║
║ CONVERSATION:                             ║
║   10. Start/Continue Socratic dialogue    ║
║   11. Ask for suggestions                 ║
║   12. Change project phase                ║
║   13. Search conversation history         ║
║                                           ║
║ KNOWLEDGE & CODE:                         ║
║   14. Generate code                       ║
║   15. Add knowledge entry                 ║
║   16. Search knowledge base               ║
║                                           ║
║ DATA MANAGEMENT:                          ║
║   17. Export project data                 ║
║   18. Database statistics                 ║
║   19. Help                                ║
║   0. Exit                                 ║
╚═══════════════════════════════════════════╝
"""
        current_info = ""
        if self.current_user:
            current_info += f"Current User: {self.users[self.current_user]['username']}\n"
        if self.current_project:
            current_info += f"Current Project: {self.current_project.name} (Phase: {self.current_project.phase})\n"

        return current_info + menu

    def get_database_stats(self) -> str:
        """Get vector database statistics"""
        try:
            knowledge_count = self.vector_store.knowledge_collection.count()
            conversation_count = self.vector_store.conversation_collection.count()
            project_count = self.vector_store.project_collection.count()

            return f"""
📊 Vector Database Statistics:
═══════════════════════════════
Knowledge Entries: {knowledge_count}
Conversation Records: {conversation_count}
Stored Projects: {project_count}
Total Users: {len(self.users)}

Database Location: {self.vector_store.persist_directory}
"""
        except Exception as e:
            return f"Could not retrieve database statistics: {str(e)}"

    def export_project_data(self, project_id: str) -> str:
        """Export project data to JSON (enhanced with vector data)"""
        project = self.vector_store.get_project(project_id)
        if not project:
            return "Project not found"

        # Get related conversations
        conversations = self.vector_store.search_conversations(
            "", project_id=project_id, n_results=100
        )

        export_data = project.to_dict()
        export_data["vector_conversations"] = conversations

        filename = f"project_{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            return f"Enhanced project data exported to {filename}"
        except Exception as e:
            return f"Export failed: {str(e)}"


def main():
    """Main application loop with vector database"""
    print("🤖 Enhanced Socratic Counselor with Vector Database")
    print("=" * 60)

    # Get API key
    api_key = os.getenv('API_KEY_CLAUDE')

    if not api_key:
        print("❌ API key is required to run the system.")
        return

    try:
        # Initialize the enhanced RAG system
        rag = SocraticRAG(api_key)
        print("✅ Enhanced Socratic RAG system initialized successfully!")
        print(f"📁 Vector database location: {rag.vector_store.persist_directory}")
        print()

    except Exception as e:
        print(f"❌ Failed to initialize system: {str(e)}")
        return

        # Main interaction loop

    while True:
        try:
            print(rag.show_menu())

            if not rag.current_user:
                # Authentication menu
                choice = input("👉 Select an option (0-3): ").strip()

                if choice == "0":
                    print("                         Goodbye!")
                    print("..τω Ασκληπιώ οφείλομεν αλετρυόνα, απόδοτε και μη αμελήσετε..")
                    break

                elif choice == "1":
                    # Login
                    username = input("Enter your username: ").strip()
                    if username:
                        if rag.login_user(username):
                            print(f"✅ Welcome back, {username}!")
                        else:
                            print("❌ Username not found.")
                    else:
                        print("❌ Username cannot be empty.")

                elif choice == "2":
                    # Create user
                    username = input("Enter new username: ").strip()
                    if username:
                        # Check if username already exists
                        if any(user_data["username"] == username for user_data in rag.users.values()):
                            print("❌ Username already exists.")
                        else:
                            user_id = rag.create_user(username)
                            rag.current_user = user_id
                            print(f"✅ User '{username}' created and logged in successfully!")
                    else:
                        print("❌ Username cannot be empty.")

                elif choice == "3":
                    # List users
                    if rag.users:
                        print("\n👥 Existing Users:")
                        for user_data in rag.users.values():
                            print(f"  • {user_data['username']}")
                    else:
                        print("❌ No users found.")

                else:
                    print("❌ Invalid option. Please select 0-3.")

            else:
                # Main menu (user is logged in)
                choice = input("👉 Select an option (0-17): ").strip()

                if choice == "0":
                    print("👋 Goodbye! Your data is safely stored in the vector database.")
                    break

                elif choice == "1":
                    # Create new project
                    name = input("Enter project name: ").strip()
                    if name:
                        try:
                            project_id = rag.create_project(name)
                            rag.set_current_project(project_id)
                            print(f"✅ Project '{name}' created successfully! You are the owner.")
                        except Exception as e:
                            print(f"❌ Failed to create project: {str(e)}")
                    else:
                        print("❌ Project name cannot be empty.")

                elif choice == "2":
                    # Select project from list
                    projects = rag.list_user_projects()
                    if projects:
                        print("\n📋 Your Projects:")
                        print("=" * 60)
                        for i, project in enumerate(projects, 1):
                            owner_indicator = " (Owner)" if project.get("is_owner") else " (Member)"
                            print(f"  {i}. {project['name']}{owner_indicator}")
                            print(f"     Phase: {project['phase']} | Updated: {project['last_updated'][:10]}")
                            print()

                        try:
                            choice_num = int(input(f"Select project (1-{len(projects)}): ").strip())
                            if 1 <= choice_num <= len(projects):
                                selected_project = projects[choice_num - 1]
                                rag.set_current_project(selected_project['project_id'])
                                print(f"✅ Selected project: {selected_project['name']}")
                            else:
                                print("❌ Invalid choice.")
                        except ValueError:
                            print("❌ Please enter a number.")
                        except Exception as e:
                            print(f"❌ Error selecting project: {str(e)}")
                    else:
                        print("📭 No projects found. Create one with option 1!")

                elif choice == "3":
                    # Delete project (owner only)
                    if not rag.current_project:
                        print("❌ No project selected.")
                        continue

                    if rag.current_project.owner != rag.current_user:
                        print("❌ Only the project owner can delete projects.")
                        continue

                    confirm = input(
                        f"⚠️  Are you sure you want to delete '{rag.current_project.name}'? (yes/no): ").strip().lower()
                    if confirm == "yes":
                        if rag.delete_project(rag.current_project.project_id):
                            print("✅ Project deleted successfully.")
                        else:
                            print("❌ Failed to delete project.")
                    else:
                        print("❌ Deletion cancelled.")

                elif choice == "4":
                    # Project summary
                    if rag.current_project:
                        project = rag.current_project
                        owner_name = rag.users.get(project.owner, {}).get('username', 'Unknown')

                        print(f"\n📊 Project Summary: {project.name}")
                        print("=" * 50)
                        print(f"Owner: {owner_name}")
                        print(f"Phase: {project.phase}")

                        # Show authorized users
                        if hasattr(project, 'authorized_users') and len(project.authorized_users) > 1:
                            other_users = [rag.users.get(uid, {}).get('username', 'Unknown')
                                           for uid in project.authorized_users if uid != project.owner]
                            print(f"Other Users: {', '.join(other_users)}")

                        print(f"Goals: {', '.join(project.goals) if project.goals else 'None specified'}")
                        print(
                            f"Requirements: {', '.join(project.requirements) if project.requirements else 'None specified'}")
                        print(
                            f"Tech Stack: {', '.join(project.tech_stack) if project.tech_stack else 'None specified'}")
                        print(
                            f"Constraints: {', '.join(project.constraints) if project.constraints else 'None specified'}")
                        print(f"Created: {project.created_at}")
                        print(f"Last Updated: {project.last_updated}")
                        print(f"Conversation History: {len(project.conversation_history)} exchanges")
                    else:
                        print("❌ No project selected. Use option 2 to select a project.")

                elif choice == "5":
                    # Manage project users (owner only)
                    if not rag.current_project:
                        print("❌ No project selected.")
                        continue

                    if rag.current_project.owner != rag.current_user:
                        print("❌ Only the project owner can manage users.")
                        continue

                    print(f"\n👥 Manage Users - {rag.current_project.name}")
                    print("1. Add user to project")
                    print("2. Remove user from project")
                    print("3. List project users")

                    sub_choice = input("Select option (1-3): ").strip()

                    if sub_choice == "1":
                        username = input("Enter username to add: ").strip()
                        if rag.add_user_to_project(rag.current_project.project_id, username):
                            print(f"✅ User '{username}' added to project.")
                        else:
                            print(f"❌ Failed to add user (user not found or already has access).")

                    elif sub_choice == "2":
                        username = input("Enter username to remove: ").strip()
                        if rag.remove_user_from_project(rag.current_project.project_id, username):
                            print(f"✅ User '{username}' removed from project.")
                        else:
                            print(f"❌ Failed to remove user (user not found, not in project, or is owner).")

                    elif sub_choice == "3":
                        if hasattr(rag.current_project, 'authorized_users'):
                            print(f"\n👥 Project Users:")
                            for user_id in rag.current_project.authorized_users:
                                username = rag.users.get(user_id, {}).get('username', 'Unknown')
                                role = "(Owner)" if user_id == rag.current_project.owner else "(Member)"
                                print(f"  • {username} {role}")
                        else:
                            print("No users found.")

                elif choice == "6":
                    # Start/Continue Socratic dialogue
                    if not rag.current_project:
                        print("❌ No project selected. Use option 2 to select a project or option 1 to create one.")
                        continue

                    print(f"\n🧠 Socratic Dialogue - Project: {rag.current_project.name}")
                    print("=" * 60)
                    print("💡 Tip: Say 'exit' to return to menu, 'I don't know' for suggestions")
                    print()

                    while True:
                        user_input = input("You: ").strip()
                        if user_input.lower() in ['exit', 'quit', 'back']:
                            break
                        if not user_input:
                            continue

                        response = rag.generate_socratic_question(user_input)
                        print(f"🤔 Counselor: {response}")
                        print()

                elif choice == "7":
                    # Ask for suggestions
                    if not rag.current_project:
                        print("❌ No project selected.")
                        continue

                    suggestion = rag._generate_enhanced_suggestion("I need suggestions")
                    print(f"\n💡 Suggestion for {rag.current_project.phase} phase:")
                    print("=" * 50)
                    print(suggestion)

                elif choice == "8":
                    # Change project phase
                    if not rag.current_project:
                        print("❌ No project selected.")
                        continue

                    phases = ["discovery", "analysis", "design", "implementation"]
                    print("\n🔄 Available Phases:")
                    for i, phase in enumerate(phases, 1):
                        marker = "👈 Current" if phase == rag.current_project.phase else ""
                        print(f"  {i}. {phase.capitalize()} {marker}")

                    try:
                        choice_num = int(input("Select phase (1-4): ").strip())
                        if 1 <= choice_num <= 4:
                            new_phase = phases[choice_num - 1]
                            rag.current_project.update_phase(new_phase)
                            rag.vector_store.store_project(rag.current_project)
                            print(f"✅ Project phase updated to: {new_phase}")
                        else:
                            print("❌ Invalid choice.")
                    except ValueError:
                        print("❌ Please enter a number.")

                elif choice == "9":
                    # Search conversation history
                    if not rag.current_project:
                        print("❌ No project selected.")
                        continue

                    query = input("Enter search query for conversation history: ").strip()
                    if query:
                        results = rag.vector_store.search_conversations(
                            query, project_id=rag.current_project.project_id, n_results=5
                        )

                        if results:
                            print(f"\n💬 Conversation Search Results for '{query}':")
                            print("=" * 60)
                            for i, result in enumerate(results, 1):
                                print(f"{i}. Phase: {result['phase']} | {result['timestamp']}")
                                print(f"   You: {result['user_input'][:80]}...")
                                print(f"   Counselor: {result['assistant_response'][:80]}...")
                                print()
                        else:
                            print("❌ No matching conversations found.")
                    else:
                        print("❌ Search query cannot be empty.")

                elif choice == "10":
                    # Generate code
                    requirements = input("Enter code requirements: ").strip()
                    if requirements:
                        print("\n⚙️  Generating code...")
                        code = rag.generate_code(requirements)
                        print("=" * 60)
                        print(code)
                        print("=" * 60)
                    else:
                        print("❌ Requirements cannot be empty.")

                elif choice == "11":
                    # Add knowledge entry
                    content = input("Enter knowledge content: ").strip()
                    category = input("Enter category (e.g., methodology, development, architecture): ").strip()
                    context = input("Enter additional context (optional): ").strip()

                    if content and category:
                        result = rag.add_knowledge(content, category, context)
                        print(f"✅ {result}")
                    else:
                        print("❌ Content and category are required.")

                elif choice == "12":
                    # Search knowledge base
                    query = input("Enter search query for knowledge base: ").strip()
                    category = input("Enter category filter (optional): ").strip()

                    if query:
                        results = rag.vector_store.search_knowledge(
                            query, category=category if category else None, n_results=5
                        )

                        if results:
                            print(f"\n📚 Knowledge Search Results for '{query}':")
                            print("=" * 60)
                            for i, result in enumerate(results, 1):
                                print(f"{i}. Category: {result['category']}")
                                print(f"   Content: {result['content']}")
                                if result['context']:
                                    print(f"   Context: {result['context']}")
                                print(f"   Relevance: {1 - result['distance']:.2%}")
                                print()
                        else:
                            print("❌ No matching knowledge found.")
                    else:
                        print("❌ Search query cannot be empty.")

                elif choice == "13":
                    # Search project insights
                    query = input("Enter search query for insights: ").strip()
                    if query:
                        insights = rag.search_project_insights(query)
                        print(f"\n🔍 Insights for '{query}':")
                        print("=" * 50)
                        print(insights)
                    else:
                        print("❌ Search query cannot be empty.")

                elif choice == "14":
                    # Export project data
                    if not rag.current_project:
                        print("❌ No project selected.")
                        continue

                    result = rag.export_project_data(rag.current_project.project_id)
                    print(f"📁 {result}")

                elif choice == "15":
                    # Database statistics
                    stats = rag.get_database_stats()
                    print(stats)

                elif choice == "16":
                    # Logout
                    username = rag.users[rag.current_user]['username']
                    rag.logout_user()
                    print(f"👋 Goodbye, {username}! You have been logged out.")

                elif choice == "17":
                    # Help
                    print("""
    🆘 ENHANCED SOCRATIC RAG SYSTEM HELP
    ═══════════════════════════════════════════════════════════
    
    🔐 AUTHENTICATION & SECURITY:
      • Must login before accessing projects
      • Project owners control access permissions  
      • Only owners can delete projects or manage users
    
    📋 GETTING STARTED:
      1. Login or create a user account
      2. Create a project (you become the owner)
      3. Start a Socratic dialogue to explore your ideas
      4. Invite team members if needed (owners only)
    
    🤔 SOCRATIC METHOD:
      The system uses guided questioning through four phases:
      • Discovery: Understanding what you want to build
      • Analysis: Examining challenges and approaches  
      • Design: Planning the structure and flow
      • Implementation: Deciding how to build and deploy
    
    👥 PROJECT COLLABORATION:
      • Project owners can add/remove team members
      • All authorized users can participate in conversations
      • Project data is shared among team members
      • Vector search works across all accessible projects
    
    🔍 VECTOR SEARCH FEATURES:
      • Knowledge base learns from conversations
      • Smart suggestions based on similar situations
      • Cross-project insights and patterns
      • Semantic search across all stored data
    
    💡 TIPS:
      • Be specific in your responses for better questions
      • Say "I don't know" to get targeted suggestions
      • Change phases as your understanding evolves
      • Export projects to save your progress
      • Use insights search to find relevant patterns
    
    🔧 ADVANCED FEATURES:
      • Vector database stores semantic relationships
      • Code generation with project context
      • Knowledge base grows with each conversation
      • Multi-user project management with permissions
      • Comprehensive search and export capabilities
    
    For more help, visit the project repository or contact support.
    """)

                else:
                    print("❌ Invalid option. Please select a number from 0-17.")

            print("\n" + "─" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Your data is safely stored in the vector database.")
            break
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")
            print("Please try again or contact support if the problem persists.")


if __name__ == "__main__":
    main()
