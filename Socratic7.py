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
import hashlib


class ProjectContext:
    """Manages project context and state"""

    def __init__(self, project_id: str = None):
        self.project_id = project_id or str(uuid.uuid4())
        self.name = "Unnamed Project"
        self.owner = None  # User ID of the project owner
        self.authorized_users = []  # List of user IDs who can access this project
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
            "owner": self.owner,
            "authorized_users": self.authorized_users,
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
                    "owner": project.owner,
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
                    "owner": project.owner,
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
                "owner": metadata.get("owner"),
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

        # Core Socratic questioning templates
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

    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        return hashlib.sha256((password + "socratic_salt").encode()).hexdigest()

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

    def create_user(self, username: str, password: str) -> str:
        """Create new user with password"""
        # Check if username already exists
        for user_data in self.users.values():
            if user_data["username"] == username:
                return None  # Username already exists

        user_id = str(uuid.uuid4())
        self.users[user_id] = {
            "username": username,
            "password_hash": self._hash_password(password),
            "created_at": datetime.now().isoformat(),
            "projects": [],
            "preferences": {}
        }
        self._save_users()
        return user_id

    def login_user(self, username: str, password: str) -> bool:
        """Login user with username and password"""
        password_hash = self._hash_password(password)
        for user_id, user_data in self.users.items():
            if user_data["username"] == username and user_data["password_hash"] == password_hash:
                self.current_user = user_id
                return True
        return False

    def logout_user(self):
        """Logout current user"""
        self.current_user = None
        self.current_project = None

    def delete_user(self, user_id: str) -> bool:
        """Delete user and all their projects"""
        if user_id not in self.users:
            return False

        # Delete all user's projects where they are the owner
        for project in self.list_projects():
            if project.get("owner") == user_id:
                self.vector_store.delete_project(project["project_id"])

        # Delete user
        del self.users[user_id]
        if self.current_user == user_id:
            self.current_user = None

        self._save_users()
        return True

    def create_project(self, project_name: str) -> str:
        """Create new project"""
        if not self.current_user:
            return None

        project = ProjectContext()
        project.name = project_name
        project.owner = self.current_user
        project.authorized_users = [self.current_user]

        # Store in vector database
        self.vector_store.store_project(project)

        # Add to user's project list
        self.users[self.current_user]["projects"].append(project.project_id)
        self._save_users()

        return project.project_id

    def delete_project(self, project_id: str) -> bool:
        """Delete project (owner only)"""
        project = self.vector_store.get_project(project_id)
        if not project or project.owner != self.current_user:
            return False

        # Remove from all users' project lists
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
        if project and self.current_user in project.authorized_users:
            self.current_project = project
            return True
        return False

    def list_user_projects(self) -> List[Dict]:
        """List projects accessible to current user"""
        if not self.current_user:
            return []

        all_projects = self.list_projects()
        user_projects = []

        for project in all_projects:
            project_obj = self.vector_store.get_project(project["project_id"])
            if project_obj and self.current_user in project_obj.authorized_users:
                project["is_owner"] = (project_obj.owner == self.current_user)
                user_projects.append(project)

        return user_projects

    def list_projects(self) -> List[Dict]:
        """List all projects"""
        return self.vector_store.list_projects()

    def add_user_to_project(self, project_id: str, username: str) -> bool:
        """Add user to project (owner only)"""
        project = self.vector_store.get_project(project_id)
        if not project or project.owner != self.current_user:
            return False

        # Find user by username
        target_user_id = None
        for user_id, user_data in self.users.items():
            if user_data["username"] == username:
                target_user_id = user_id
                break

        if not target_user_id:
            return False

        # Add user to project if not already authorized
        if target_user_id not in project.authorized_users:
            project.authorized_users.append(target_user_id)
            self.users[target_user_id]["projects"].append(project_id)

            # Save changes
            self.vector_store.store_project(project)
            self._save_users()
            return True

        return False

    def remove_user_from_project(self, project_id: str, username: str) -> bool:
        """Remove user from project (owner only)"""
        project = self.vector_store.get_project(project_id)
        if not project or project.owner != self.current_user:
            return False

        # Find user by username
        target_user_id = None
        for user_id, user_data in self.users.items():
            if user_data["username"] == username:
                target_user_id = user_id
                break

        if not target_user_id or target_user_id == project.owner:
            return False  # Can't remove owner

        # Remove user from project
        if target_user_id in project.authorized_users:
            project.authorized_users.remove(target_user_id)
            if project_id in self.users[target_user_id]["projects"]:
                self.users[target_user_id]["projects"].remove(project_id)

            # Save changes
            self.vector_store.store_project(project)
            self._save_users()
            return True

        return False

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
                insights.append(f"  • In {result['phase']} phase: {result['user_input'][:100]}...")

            if project_results:
                insights.append("\n🔍 Similar Projects:")
                for result in project_results:
                    insights.append(f"  • {result['name']} (Phase: {result['phase']})")

            return "\n".join(insights) if insights else "No relevant insights found."

    def update_project_context(self, field: str, value: str):
        """Update project context field"""
        if not self.current_project:
            return "No active project selected."

        if field == "name":
            self.current_project.name = value
        elif field == "team_structure":
            self.current_project.team_structure = value
        elif field == "language_preference":
            self.current_project.language_preference = value
        elif field == "deployment_target":
            self.current_project.deployment_target = value
        elif field == "code_style":
            self.current_project.code_style = value
        elif field == "phase":
            self.current_project.update_phase(value)
        else:
            # Handle adding to list fields
            self.current_project.add_context_item(field, value)

        # Save updated project
        self.vector_store.store_project(self.current_project)
        return f"Updated {field}: {value}"

    def get_project_summary(self) -> str:
        """Get comprehensive project summary"""
        if not self.current_project:
            return "No active project selected."

        summary = [f"📋 Project: {self.current_project.name}"]
        summary.append(f"Phase: {self.current_project.phase}")
        summary.append(f"Owner: {self.users.get(self.current_project.owner, {}).get('username', 'Unknown')}")

        if len(self.current_project.authorized_users) > 1:
            usernames = []
            for user_id in self.current_project.authorized_users:
                if user_id != self.current_project.owner:
                    username = self.users.get(user_id, {}).get('username', 'Unknown')
                    usernames.append(username)
            if usernames:
                summary.append(f"Collaborators: {', '.join(usernames)}")

        if self.current_project.goals:
            summary.append(f"\n🎯 Goals:\n  • " + "\n  • ".join(self.current_project.goals))

        if self.current_project.requirements:
            summary.append(f"\n📋 Requirements:\n  • " + "\n  • ".join(self.current_project.requirements))

        if self.current_project.tech_stack:
            summary.append(f"\n💻 Tech Stack:\n  • " + "\n  • ".join(self.current_project.tech_stack))

        if self.current_project.constraints:
            summary.append(f"\n⚠️ Constraints:\n  • " + "\n  • ".join(self.current_project.constraints))

        if self.current_project.team_structure:
            summary.append(f"\n👥 Team Structure: {self.current_project.team_structure}")

        if self.current_project.language_preference:
            summary.append(f"\n🗣️ Language: {self.current_project.language_preference}")

        if self.current_project.deployment_target:
            summary.append(f"\n🚀 Deployment: {self.current_project.deployment_target}")

        if self.current_project.code_style:
            summary.append(f"\n✨ Code Style: {self.current_project.code_style}")

        # Add conversation stats
        conv_count = len(self.current_project.conversation_history)
        if conv_count > 0:
            summary.append(f"\n📊 Conversations: {conv_count} exchanges")
            summary.append(f"Last Updated: {self.current_project.last_updated[:19]}")

        return "\n".join(summary)

    def export_project_data(self, project_id: str = None) -> Optional[Dict]:
        """Export project data for backup or transfer"""
        project = self.current_project if not project_id else self.vector_store.get_project(project_id)
        if not project:
            return None

        # Check permissions
        if self.current_user not in project.authorized_users:
            return None

        # Get related conversations
        conversations = self.vector_store.search_conversations(
            query="*",  # Get all conversations
            project_id=project.project_id,
            n_results=1000  # Large number to get all
        )

        return {
            "project": project.to_dict(),
            "conversations": conversations,
            "exported_at": datetime.now().isoformat(),
            "exported_by": self.current_user
        }

    def import_project_data(self, project_data: Dict) -> bool:
        """Import project data from export"""
        if not self.current_user:
            return False

        try:
            # Create project from imported data
            project_dict = project_data["project"]
            project_dict["project_id"] = str(uuid.uuid4())  # Generate new ID
            project_dict["owner"] = self.current_user
            project_dict["authorized_users"] = [self.current_user]
            project_dict["imported_at"] = datetime.now().isoformat()

            project = ProjectContext.from_dict(project_dict)

            # Store project
            self.vector_store.store_project(project)

            # Import conversations (they'll get new IDs automatically)
            for conv in project_data.get("conversations", []):
                self.vector_store.add_conversation(
                    project.project_id,
                    conv["user_input"],
                    conv["assistant_response"],
                    conv["phase"]
                )

            # Add to user's project list
            self.users[self.current_user]["projects"].append(project.project_id)
            self._save_users()

            return True

        except Exception as e:
            print(f"Import failed: {e}")
            return False

    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history for current project"""
        if not self.current_project:
            return []

        history = self.current_project.conversation_history[-limit:]
        return [
            {
                "user_input": conv["user"],
                "assistant_response": conv["assistant"],
                "timestamp": conv["timestamp"],
                "phase": conv["phase"]
            }
            for conv in history
        ]

    def backup_all_data(self, backup_path: str = None) -> str:
        """Create backup of all user data"""
        if not backup_path:
            backup_path = f"socratic_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            os.makedirs(backup_path, exist_ok=True)

            # Backup ChromaDB
            if os.path.exists(self.vector_store.persist_directory):
                shutil.copytree(
                    self.vector_store.persist_directory,
                    os.path.join(backup_path, "chroma_db"),
                    dirs_exist_ok=True
                )

            # Backup users file
            users_file = os.path.join(self.vector_store.persist_directory, "users.json")
            if os.path.exists(users_file):
                shutil.copy2(users_file, os.path.join(backup_path, "users.json"))

            # Create backup manifest
            manifest = {
                "backup_created": datetime.now().isoformat(),
                "version": "1.0",
                "description": "Socratic RAG system backup"
            }

            with open(os.path.join(backup_path, "manifest.json"), 'w') as f:
                json.dump(manifest, f, indent=2)

            return f"Backup created successfully at: {backup_path}"

        except Exception as e:
            return f"Backup failed: {str(e)}"

    def restore_from_backup(self, backup_path: str) -> str:
        """Restore data from backup"""
        try:
            # Check if backup exists and is valid
            manifest_file = os.path.join(backup_path, "manifest.json")
            if not os.path.exists(manifest_file):
                return "Invalid backup: manifest.json not found"

            # Restore ChromaDB
            backup_chroma = os.path.join(backup_path, "chroma_db")
            if os.path.exists(backup_chroma):
                if os.path.exists(self.vector_store.persist_directory):
                    shutil.rmtree(self.vector_store.persist_directory)
                shutil.copytree(backup_chroma, self.vector_store.persist_directory)

            # Restore users file
            backup_users = os.path.join(backup_path, "users.json")
            if os.path.exists(backup_users):
                shutil.copy2(backup_users,
                             os.path.join(self.vector_store.persist_directory, "users.json"))

            # Reinitialize vector store
            self.vector_store = VectorStore(self.vector_store.persist_directory)
            self._load_users()

            return "Restore completed successfully"

        except Exception as e:
            return f"Restore failed: {str(e)}"

    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            "total_users": len(self.users),
            "total_projects": len(self.list_projects()),
            "current_user": self.users.get(self.current_user, {}).get('username') if self.current_user else None,
            "current_project": self.current_project.name if self.current_project else None,
            "knowledge_entries": self.vector_store.knowledge_collection.count(),
            "total_conversations": self.vector_store.conversation_collection.count(),
        }

        if self.current_project:
            stats["project_phase"] = self.current_project.phase
            stats["project_conversations"] = len(self.current_project.conversation_history)

        return stats


def main():
    """Main CLI interface"""
    import getpass

    print("🤖 Socratic RAG System v7.0")
    print("=" * 40)

    # Initialize system
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your Anthropic API key: ")

    rag = SocraticRAG(api_key)

    while True:
        print("\n📋 Main Menu:")
        print("1. Login")
        print("2. Create Account")
        print("3. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "1":
            username = input("Username: ")
            password = getpass.getpass("Password: ")

            if rag.login_user(username, password):
                print(f"✅ Logged in as {username}")
                user_session(rag)
            else:
                print("❌ Invalid credentials")

        elif choice == "2":
            username = input("Choose username: ")
            password = getpass.getpass("Choose password: ")

            user_id = rag.create_user(username, password)
            if user_id:
                print(f"✅ Account created successfully!")
                rag.login_user(username, password)
                user_session(rag)
            else:
                print("❌ Username already exists")

        elif choice == "3":
            print("👋 Goodbye!")
            break


def user_session(rag):
    """User session menu"""
    while True:
        print(f"\n👤 User Menu ({rag.users[rag.current_user]['username']}):")
        print("1. List Projects")
        print("2. Create Project")
        print("3. Select Project")
        print("4. Chat (Socratic Mode)")
        print("5. Generate Code")
        print("6. Project Summary")
        print("7. Search Insights")
        print("8. System Stats")
        print("9. Backup Data")
        print("10. Logout")

        choice = input("\nChoice: ").strip()

        if choice == "1":
            projects = rag.list_user_projects()
            if projects:
                print("\n📁 Your Projects:")
                for i, project in enumerate(projects, 1):
                    owner_mark = "👑" if project.get("is_owner") else "👥"
                    print(f"  {i}. {owner_mark} {project['name']} ({project['phase']})")
            else:
                print("No projects found.")

        elif choice == "2":
            name = input("Project name: ")
            project_id = rag.create_project(name)
            if project_id:
                print(f"✅ Project '{name}' created!")
                rag.set_current_project(project_id)
                print(f"🎯 Current project set to: {name}")
            else:
                print("❌ Failed to create project")

        elif choice == "3":
            projects = rag.list_user_projects()
            if projects:
                print("\n📁 Available Projects:")
                for i, project in enumerate(projects, 1):
                    owner_mark = "👑" if project.get("is_owner") else "👥"
                    print(f"  {i}. {owner_mark} {project['name']} ({project['phase']})")

                try:
                    idx = int(input("Select project (number): ")) - 1
                    if 0 <= idx < len(projects):
                        selected = projects[idx]
                        if rag.set_current_project(selected['project_id']):
                            print(f"🎯 Current project: {selected['name']}")
                        else:
                            print("❌ Cannot access project")
                    else:
                        print("❌ Invalid selection")
                except ValueError:
                    print("❌ Please enter a number")
            else:
                print("No projects available.")

        elif choice == "4":
            if not rag.current_project:
                print("❌ Please select a project first")
                continue

            print(f"\n💬 Socratic Chat - {rag.current_project.name}")
            print("Type 'quit' to return to menu")
            print("-" * 40)

            while True:
                user_input = input("\n🧑 You: ").strip()
                if user_input.lower() == 'quit':
                    break

                if user_input:
                    response = rag.generate_socratic_question(user_input)
                    print(f"🤖 Socrates: {response}")

        elif choice == "5":
            if not rag.current_project:
                print("❌ Please select a project first")
                continue

            requirements = input("Code requirements: ")
            if requirements:
                print("\n💻 Generating code...")
                code = rag.generate_code(requirements)
                print(f"\n{code}")

        elif choice == "6":
            summary = rag.get_project_summary()
            print(f"\n{summary}")

        elif choice == "7":
            query = input("Search query: ")
            if query:
                insights = rag.search_project_insights(query)
                print(f"\n{insights}")

        elif choice == "8":
            stats = rag.get_system_stats()
            print("\n📊 System Statistics:")
            for key, value in stats.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")

        elif choice == "9":
            backup_path = rag.backup_all_data()
            print(f"\n{backup_path}")

        elif choice == "10":
            rag.logout_user()
            print("👋 Logged out successfully")
            break


if __name__ == "__main__":
    main()
