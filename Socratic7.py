import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from anthropic import Anthropic
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import uuid
import shutil
import hashlib
import sqlite3
import requests
import base64
from urllib.parse import urlparse
import threading
import time


class DatabaseManager:
    """SQLite database manager for user data"""

    def __init__(self, db_path: str = "./socratic_users.db"):
        self.db_path = db_path
        self._init_database()
        self._lock = threading.Lock()

    def _init_database(self):
        """Initialize SQLite database with user tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    preferences TEXT DEFAULT '{}',
                    total_tokens_used INTEGER DEFAULT 0,
                    last_login TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_projects (
                    user_id TEXT,
                    project_id TEXT,
                    role TEXT DEFAULT 'member',
                    added_at TEXT,
                    PRIMARY KEY (user_id, project_id),
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    project_id TEXT,
                    tokens_used INTEGER,
                    cost_estimate REAL,
                    timestamp TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            conn.commit()

    def create_user(self, username: str, password_hash: str) -> Optional[str]:
        """Create new user, returns user_id or None if username exists"""
        user_id = str(uuid.uuid4())

        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO users (user_id, username, password_hash, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (user_id, username, password_hash, datetime.now().isoformat()))
                    conn.commit()
                return user_id
            except sqlite3.IntegrityError:
                return None  # Username already exists

    def authenticate_user(self, username: str, password_hash: str) -> Optional[str]:
        """Authenticate user, returns user_id or None"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id FROM users 
                WHERE username = ? AND password_hash = ?
            """, (username, password_hash))

            result = cursor.fetchone()
            if result:
                # Update last login
                conn.execute("""
                    UPDATE users SET last_login = ? WHERE user_id = ?
                """, (datetime.now().isoformat(), result[0]))
                conn.commit()
                return result[0]
            return None

    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """Get user information"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT username, created_at, total_tokens_used, last_login, preferences
                FROM users WHERE user_id = ?
            """, (user_id,))

            result = cursor.fetchone()
            if result:
                return {
                    "username": result[0],
                    "created_at": result[1],
                    "total_tokens_used": result[2],
                    "last_login": result[3],
                    "preferences": json.loads(result[4]) if result[4] else {}
                }
            return None

    def add_user_to_project(self, user_id: str, project_id: str, role: str = "member"):
        """Add user to project"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO user_projects (user_id, project_id, role, added_at)
                    VALUES (?, ?, ?, ?)
                """, (user_id, project_id, role, datetime.now().isoformat()))
                conn.commit()

    def get_user_projects(self, user_id: str) -> List[str]:
        """Get project IDs for user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT project_id FROM user_projects WHERE user_id = ?
            """, (user_id,))
            return [row[0] for row in cursor.fetchall()]

    def remove_user_from_project(self, user_id: str, project_id: str):
        """Remove user from project"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM user_projects WHERE user_id = ? AND project_id = ?
                """, (user_id, project_id))
                conn.commit()

    def log_api_usage(self, user_id: str, project_id: str, tokens_used: int, cost_estimate: float):
        """Log API usage for billing/tracking"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Add to usage log
                conn.execute("""
                    INSERT INTO api_usage (user_id, project_id, tokens_used, cost_estimate, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, project_id, tokens_used, cost_estimate, datetime.now().isoformat()))

                # Update user total
                conn.execute("""
                    UPDATE users SET total_tokens_used = total_tokens_used + ?
                    WHERE user_id = ?
                """, (tokens_used, user_id))

                conn.commit()

    def get_usage_stats(self, user_id: str) -> Dict:
        """Get usage statistics for user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get total usage
            cursor.execute("""
                SELECT SUM(tokens_used), SUM(cost_estimate)
                FROM api_usage WHERE user_id = ?
            """, (user_id,))
            total_result = cursor.fetchone()

            # Get recent usage (last 24 hours)
            cursor.execute("""
                SELECT SUM(tokens_used), SUM(cost_estimate)
                FROM api_usage 
                WHERE user_id = ? AND datetime(timestamp) > datetime('now', '-1 day')
            """, (user_id,))
            recent_result = cursor.fetchone()

            return {
                "total_tokens": total_result[0] or 0,
                "total_cost": total_result[1] or 0.0,
                "recent_tokens": recent_result[0] or 0,
                "recent_cost": recent_result[1] or 0.0
            }

    def delete_user(self, user_id: str):
        """Delete user and all related data"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM api_usage WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM user_projects WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                conn.commit()


class GitHubReader:
    """GitHub repository reader"""

    def __init__(self, token: str = None):
        self.token = token
        self.headers = {"Authorization": f"token {token}"} if token else {}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def parse_github_url(self, url: str) -> Optional[Dict]:
        """Parse GitHub URL to extract owner, repo, and path"""
        try:
            parsed = urlparse(url)
            if 'github.com' not in parsed.netloc:
                return None

            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) < 2:
                return None

            owner, repo = path_parts[0], path_parts[1]

            # Handle different URL formats
            file_path = ""
            if len(path_parts) > 2:
                if path_parts[2] in ['tree', 'blob']:
                    # URL like github.com/owner/repo/tree/branch/path or github.com/owner/repo/blob/branch/file
                    if len(path_parts) > 4:
                        file_path = '/'.join(path_parts[4:])
                else:
                    file_path = '/'.join(path_parts[2:])

            return {
                "owner": owner,
                "repo": repo,
                "path": file_path
            }
        except Exception:
            return None

    def get_file_content(self, owner: str, repo: str, path: str, branch: str = "main") -> Optional[str]:
        """Get file content from GitHub"""
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
            params = {"ref": branch}

            response = self.session.get(url, params=params)
            if response.status_code == 404 and branch == "main":
                # Try master branch
                params["ref"] = "master"
                response = self.session.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                if data.get("type") == "file" and data.get("content"):
                    content = base64.b64decode(data["content"]).decode('utf-8')
                    return content

            return None
        except Exception as e:
            print(f"Error fetching file: {e}")
            return None

    def get_repository_structure(self, owner: str, repo: str, path: str = "", branch: str = "main") -> List[Dict]:
        """Get repository structure"""
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
            params = {"ref": branch}

            response = self.session.get(url, params=params)
            if response.status_code == 404 and branch == "main":
                params["ref"] = "master"
                response = self.session.get(url, params=params)

            if response.status_code == 200:
                return response.json()

            return []
        except Exception:
            return []

    def read_repository(self, github_url: str, file_extensions: List[str] = None) -> Dict:
        """Read repository files based on extensions"""
        if file_extensions is None:
            file_extensions = ['.py', '.js', '.md', '.txt', '.json', '.yaml', '.yml']

        parsed = self.parse_github_url(github_url)
        if not parsed:
            return {"error": "Invalid GitHub URL"}

        owner, repo = parsed["owner"], parsed["repo"]
        start_path = parsed["path"]

        results = {
            "repository": f"{owner}/{repo}",
            "files": [],
            "structure": []
        }

        def scan_directory(path: str):
            items = self.get_repository_structure(owner, repo, path)

            for item in items:
                if item["type"] == "file":
                    file_path = item["path"]
                    file_ext = os.path.splitext(file_path)[1].lower()

                    if file_ext in file_extensions:
                        content = self.get_file_content(owner, repo, file_path)
                        if content:
                            results["files"].append({
                                "path": file_path,
                                "content": content,
                                "size": item["size"]
                            })

                    results["structure"].append({
                        "path": file_path,
                        "type": "file",
                        "size": item["size"]
                    })

                elif item["type"] == "dir":
                    results["structure"].append({
                        "path": item["path"],
                        "type": "directory"
                    })
                    # Recursively scan subdirectories (with limit)
                    if len(results["files"]) < 50:  # Limit to prevent too many files
                        scan_directory(item["path"])

        scan_directory(start_path)
        return results


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
        self.github_repositories = []  # Store linked GitHub repos
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
        elif category == "github_repositories":
            self.github_repositories.append(item)
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
            "github_repositories": self.github_repositories,
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
        self.github_collection = self._get_or_create_collection("github_code")

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

    def add_github_code(self, project_id: str, repository: str, file_data: List[Dict]):
        """Add GitHub repository code to vector store"""
        for file_info in file_data:
            file_id = f"github_{project_id}_{hash(file_info['path'])}"

            # Create searchable document from code content
            document = f"File: {file_info['path']}\n{file_info['content']}"

            self.github_collection.add(
                documents=[document],
                metadatas=[{
                    "project_id": project_id,
                    "repository": repository,
                    "file_path": file_info["path"],
                    "type": "github_code",
                    "size": file_info["size"],
                    "added_at": datetime.now().isoformat()
                }],
                ids=[file_id]
            )

    def search_github_code(self, query: str, project_id: str = None, n_results: int = 5) -> List[Dict]:
        """Search GitHub code"""
        where_clause = {"type": "github_code"}
        if project_id:
            where_clause["project_id"] = project_id

        try:
            results = self.github_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause
            )

            return [{
                "file_path": metadata["file_path"],
                "repository": metadata["repository"],
                "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                "distance": distance
            } for metadata, doc, distance in
                zip(results["metadatas"][0], results["documents"][0], results["distances"][0])]
        except Exception:
            return []

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

            # Delete related GitHub code
            github_results = self.github_collection.get(
                where={"project_id": project_id},
                include=["ids"]
            )
            if github_results["ids"]:
                self.github_collection.delete(ids=github_results["ids"])

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
    """Enhanced Socratic RAG system with database, GitHub integration, and token tracking"""

    def __init__(self, api_key: str, persist_directory: str = "./chroma_db", github_token: str = None):
        self.client = Anthropic(api_key=api_key)
        self.vector_store = VectorStore(persist_directory)
        self.db = DatabaseManager()
        self.github_reader = GitHubReader(github_token)
        self.current_project = None
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

    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        return hashlib.sha256((password + "socratic_salt").encode()).hexdigest()

    def create_user(self, username: str, password: str) -> Optional[str]:
        """Create new user with password"""
        password_hash = self._hash_password(password)
        return self.db.create_user(username, password_hash)

    def login_user(self, username: str, password: str) -> bool:
        """Login user with username and password"""
        password_hash = self._hash_password(password)
        user_id = self.db.authenticate_user(username, password_hash)
        if user_id:
            self.current_user = user_id
            return True
        return False

    def logout_user(self):
        """Logout current user"""
        self.current_user = None
        self.current_project = None

    def get_current_user_info(self) -> Optional[Dict]:
        """Get current user information"""
        if not self.current_user:
            return None
        return self.db.get_user_info(self.current_user)

    def get_token_usage_info(self) -> Dict:
        """Get token usage information for current user"""
        if not self.current_user:
            return {"error": "No user logged in"}

        usage_stats = self.db.get_usage_stats(self.current_user)
        user_info = self.db.get_user_info(self.current_user)

        return {
            "username": user_info["username"] if user_info else "Unknown",
            "total_tokens_used": usage_stats["total_tokens"],
            "total_estimated_cost": usage_stats["total_cost"],
            "recent_tokens_24h": usage_stats["recent_tokens"],
            "recent_cost_24h": usage_stats["recent_cost"],
            "account_created": user_info["created_at"] if user_info else "Unknown"
        }

    def estimate_token_count(self, text: str) -> int:
        """Rough estimation of token count (Claude uses ~4 chars per token)"""
        return len(text) // 4

    def calculate_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost based on Claude pricing"""
        # Claude Sonnet pricing (approximate)
        input_cost_per_token = 0.000003  # $3 per million input tokens
        output_cost_per_token = 0.000015  # $15 per million output tokens

        return (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)

    def create_project(self, name: str) -> str:
        """Create new project"""
        if not self.current_user:
            raise ValueError("Must be logged in to create project")

        project = ProjectContext()
        project.name = name
        project.owner = self.current_user
        project.authorized_users = [self.current_user]

        # Store in vector database
        self.vector_store.store_project(project)

        # Add user to project in database
        self.db.add_user_to_project(self.current_user, project.project_id, "owner")

        self.current_project = project
        return project.project_id

    def load_project(self, project_id: str) -> bool:
        """Load existing project"""
        if not self.current_user:
            return False

        # Check if user has access to this project
        user_projects = self.db.get_user_projects(self.current_user)
        if project_id not in user_projects:
            return False

        project = self.vector_store.get_project(project_id)
        if project:
            self.current_project = project
            return True
        return False

    def list_user_projects(self) -> List[Dict]:
        """List projects accessible to current user"""
        if not self.current_user:
            return []

        user_project_ids = self.db.get_user_projects(self.current_user)
        all_projects = self.vector_store.list_projects()

        # Filter projects user has access to
        accessible_projects = [
            proj for proj in all_projects
            if proj["project_id"] in user_project_ids
        ]

        return accessible_projects

    def add_github_repository(self, github_url: str) -> Dict:
        """Add GitHub repository to current project"""
        if not self.current_project:
            return {"error": "No project selected"}

        if not self.current_user:
            return {"error": "Must be logged in"}

        try:
            # Read repository
            repo_data = self.github_reader.read_repository(github_url)

            if "error" in repo_data:
                return repo_data

            # Add to vector store
            self.vector_store.add_github_code(
                self.current_project.project_id,
                repo_data["repository"],
                repo_data["files"]
            )

            # Update project context
            self.current_project.add_context_item("github_repositories", {
                "url": github_url,
                "repository": repo_data["repository"],
                "files_count": len(repo_data["files"]),
                "added_at": datetime.now().isoformat()
            })

            # Save updated project
            self.vector_store.store_project(self.current_project)

            return {
                "success": True,
                "repository": repo_data["repository"],
                "files_processed": len(repo_data["files"]),
                "structure_items": len(repo_data["structure"])
            }

        except Exception as e:
            return {"error": f"Failed to process repository: {str(e)}"}

    def search_project_code(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search code in current project's GitHub repositories"""
        if not self.current_project:
            return []

        return self.vector_store.search_github_code(
            query,
            self.current_project.project_id,
            n_results
        )

    def get_contextual_knowledge(self, query: str, phase: str) -> dict[str, list[dict] | list[Any]]:
        """Get relevant knowledge based on current phase and query"""
        # Search knowledge base
        knowledge_results = self.vector_store.search_knowledge(query, n_results=3)

        # Search conversation history if project exists
        conversation_results = []
        if self.current_project:
            conversation_results = self.vector_store.search_conversations(
                query,
                self.current_project.project_id,
                n_results=2
            )

        # Search similar projects
        similar_projects = self.vector_store.search_similar_projects(query, n_results=2)

        return {
            "knowledge": knowledge_results,
            "conversations": conversation_results,
            "similar_projects": similar_projects
        }

    def generate_socratic_response(self, user_input: str) -> Dict:
        """Generate Socratic response with RAG context"""
        if not self.current_user:
            return {"error": "Must be logged in to use the system"}

        if not self.current_project:
            return {"error": "No project selected. Please create or load a project first."}

        try:
            phase = self.current_project.phase

            # Get contextual information
            context_data = self.get_contextual_knowledge(user_input, phase)

            # Search project code if relevant
            code_context = []
            if any(keyword in user_input.lower() for keyword in
                   ['code', 'implementation', 'function', 'class', 'method']):
                code_context = self.search_project_code(user_input, n_results=3)

            # Build context for Claude
            context_parts = []

            # Add knowledge base context
            if context_data["knowledge"]:
                context_parts.append("Relevant software development knowledge:")
                for item in context_data["knowledge"]:
                    context_parts.append(f"- {item['content']} ({item['category']})")

            # Add conversation history context
            if context_data["conversations"]:
                context_parts.append("\nRelevant previous discussions:")
                for conv in context_data["conversations"]:
                    context_parts.append(f"- User asked: {conv['user_input'][:100]}...")
                    context_parts.append(f"  Response was: {conv['assistant_response'][:100]}...")

            # Add code context if available
            if code_context:
                context_parts.append("\nRelevant code from project repositories:")
                for code in code_context:
                    context_parts.append(f"- {code['file_path']}: {code['content_preview']}")

            # Add project context
            project_context = f"""
Current Project: {self.current_project.name}
Phase: {self.current_project.phase}
Goals: {', '.join(self.current_project.goals) if self.current_project.goals else 'Not defined yet'}
Requirements: {', '.join(self.current_project.requirements) if self.current_project.requirements else 'Not defined yet'}
Tech Stack: {', '.join(self.current_project.tech_stack) if self.current_project.tech_stack else 'Not defined yet'}
"""

            # Get phase-appropriate questions and suggestions
            questions = self.socratic_templates.get(phase, [])
            suggestions = self.suggestion_templates.get(phase, [])

            # Build the prompt
            system_prompt = f"""You are a Socratic AI assistant helping with software development projects. Your role is to guide users through thoughtful questioning rather than giving direct answers.

{project_context}

Context from knowledge base and previous conversations:
{chr(10).join(context_parts) if context_parts else "No additional context available."}

Current phase: {phase}

Guidelines:
1. Ask probing questions that help users think deeper
2. Reference the project context when relevant
3. Use the provided knowledge and conversation history to inform your responses
4. Guide users toward discovering solutions themselves
5. If code context is available, reference it to ground discussions in their actual codebase
6. Provide 1-2 follow-up questions at the end
7. Keep responses concise but thoughtful

Available phase-appropriate questions you might adapt:
{chr(10).join(f"- {q}" for q in questions[:3])}

Suggestions you might reference:
{chr(10).join(f"- {s}" for s in suggestions[:2])}
"""

            # Estimate input tokens
            input_text = system_prompt + user_input
            estimated_input_tokens = self.estimate_token_count(input_text)

            # Make API call to Claude
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_input}]
            )

            assistant_response = response.content[0].text

            # Estimate output tokens and calculate cost
            estimated_output_tokens = self.estimate_token_count(assistant_response)
            estimated_cost = self.calculate_cost_estimate(estimated_input_tokens, estimated_output_tokens)

            # Log API usage
            self.db.log_api_usage(
                self.current_user,
                self.current_project.project_id,
                estimated_input_tokens + estimated_output_tokens,
                estimated_cost
            )

            # Store conversation in vector database
            self.vector_store.add_conversation(
                self.current_project.project_id,
                user_input,
                assistant_response,
                phase
            )

            # Update project conversation history
            self.current_project.conversation_history.append({
                "user_input": user_input,
                "assistant_response": assistant_response,
                "timestamp": datetime.now().isoformat(),
                "phase": phase
            })

            # Save updated project
            self.vector_store.store_project(self.current_project)

            # Get updated usage info
            usage_info = self.get_token_usage_info()

            return {
                "response": assistant_response,
                "phase": phase,
                "token_usage": {
                    "input_tokens": estimated_input_tokens,
                    "output_tokens": estimated_output_tokens,
                    "total_tokens": estimated_input_tokens + estimated_output_tokens,
                    "estimated_cost": estimated_cost,
                    "user_total_tokens": usage_info["total_tokens_used"],
                    "user_total_cost": usage_info["total_estimated_cost"]
                },
                "context_used": {
                    "knowledge_items": len(context_data["knowledge"]),
                    "conversation_items": len(context_data["conversations"]),
                    "code_items": len(code_context),
                    "similar_projects": len(context_data["similar_projects"])
                }
            }

        except Exception as e:
            return {"error": f"Failed to generate response: {str(e)}"}

    def update_project_phase(self, new_phase: str) -> bool:
        """Update current project phase"""
        if not self.current_project:
            return False

        valid_phases = ["discovery", "analysis", "design", "implementation"]
        if new_phase not in valid_phases:
            return False

        self.current_project.update_phase(new_phase)
        self.vector_store.store_project(self.current_project)
        return True

    def add_project_context(self, category: str, item: str) -> bool:
        """Add context item to current project"""
        if not self.current_project:
            return False

        valid_categories = ["goals", "requirements", "tech_stack", "constraints"]
        if category not in valid_categories:
            return False

        self.current_project.add_context_item(category, item)
        self.vector_store.store_project(self.current_project)
        return True

    def get_project_summary(self) -> Dict:
        """Get current project summary"""
        if not self.current_project:
            return {"error": "No project selected"}

        return {
            "project_id": self.current_project.project_id,
            "name": self.current_project.name,
            "phase": self.current_project.phase,
            "goals": self.current_project.goals,
            "requirements": self.current_project.requirements,
            "tech_stack": self.current_project.tech_stack,
            "constraints": self.current_project.constraints,
            "github_repositories": self.current_project.github_repositories,
            "conversation_count": len(self.current_project.conversation_history),
            "created_at": self.current_project.created_at,
            "last_updated": self.current_project.last_updated
        }

    def export_project_data(self) -> Optional[Dict]:
        """Export current project data"""
        if not self.current_project:
            return None

        return self.current_project.to_dict()

    def delete_current_project(self) -> bool:
        """Delete current project"""
        if not self.current_project or not self.current_user:
            return False

        # Check if user is owner
        if self.current_project.owner != self.current_user:
            return False

        # Delete from vector store
        success = self.vector_store.delete_project(self.current_project.project_id)

        if success:
            # Remove user associations
            self.db.remove_user_from_project(self.current_user, self.current_project.project_id)
            self.current_project = None

        return success

    def share_project(self, username: str) -> bool:
        """Share current project with another user"""
        if not self.current_project or not self.current_user:
            return False

        # Check if user is owner
        if self.current_project.owner != self.current_user:
            return False

        # Find user by username
        # Note: We'd need a method to get user_id by username in DatabaseManager
        # For now, assuming username is the user_id
        try:
            self.db.add_user_to_project(username, self.current_project.project_id, "member")
            return True
        except Exception:
            return False

    def get_system_status(self) -> Dict:
        """Get system status and user information"""
        status = {
            "user_logged_in": self.current_user is not None,
            "current_project": self.current_project.name if self.current_project else None,
            "available_projects": len(self.list_user_projects()),
        }

        if self.current_user:
            usage_info = self.get_token_usage_info()
            status.update({
                "user_info": usage_info,
                "project_phase": self.current_project.phase if self.current_project else None,
            })

        return status


def main():
    """Main CLI interface for the Socratic RAG system"""
    import getpass

    # Initialize system
    api_key = os.getenv("API_KEY_CLAUDE")
    github_token = os.getenv("GITHUB_TOKEN")

    if not api_key:
        print("Please set API_KEY_CLAUDE environment variable")
        return

    rag = SocraticRAG(api_key, github_token=github_token)

    print("🎓 Welcome to Socratic RAG - Your AI Software Development Mentor")
    print("=" * 60)

    # User authentication
    while not rag.current_user:
        print("\n1. Login")
        print("2. Create Account")
        choice = input("\nChoose option (1-2): ").strip()

        if choice == "1":
            username = input("Username: ")
            password = getpass.getpass("Password: ")

            if rag.login_user(username, password):
                print(f"✅ Welcome back, {username}!")

                # Show user stats
                usage_info = rag.get_token_usage_info()
                print(f"📊 Total tokens used: {usage_info['total_tokens_used']:,}")
                print(f"💰 Estimated total cost: ${usage_info['total_estimated_cost']:.4f}")

                if usage_info['recent_tokens_24h'] > 0:
                    print(
                        f"🕒 Last 24h: {usage_info['recent_tokens_24h']} tokens (${usage_info['recent_cost_24h']:.4f})")
            else:
                print("❌ Invalid credentials")

        elif choice == "2":
            username = input("Choose username: ")
            password = getpass.getpass("Choose password: ")

            user_id = rag.create_user(username, password)
            if user_id:
                print(f"✅ Account created! Logging you in...")
                rag.login_user(username, password)
            else:
                print("❌ Username already exists")

    # Project selection/creation
    projects = rag.list_user_projects()

    if projects:
        print(f"\n📁 Your Projects ({len(projects)}):")
        for i, project in enumerate(projects):
            print(f"{i + 1}. {project['name']} (Phase: {project['phase']})")

        print(f"{len(projects) + 1}. Create New Project")

        try:
            choice = int(input(f"\nSelect project (1-{len(projects) + 1}): "))
            if 1 <= choice <= len(projects):
                project = projects[choice - 1]
                if rag.load_project(project['project_id']):
                    print(f"✅ Loaded project: {project['name']}")
                else:
                    print("❌ Failed to load project")
                    return
            elif choice == len(projects) + 1:
                name = input("Project name: ")
                project_id = rag.create_project(name)
                print(f"✅ Created project: {name}")
            else:
                print("Invalid selection")
                return
        except ValueError:
            print("Invalid selection")
            return
    else:
        print("\n🆕 No projects found. Let's create your first project!")
        name = input("Project name: ")
        project_id = rag.create_project(name)
        print(f"✅ Created project: {name}")

    # Main interaction loop
    print(f"\n🤖 Ready to help with '{rag.current_project.name}'")
    print("💡 Commands: /help, /phase, /github, /status, /context, /export, /quit")
    print("-" * 60)

    while True:
        try:
            user_input = input(f"\n[{rag.current_project.phase}] You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()

                if command == '/quit' or command == '/exit':
                    print("👋 Goodbye!")
                    break

                elif command == '/help':
                    print("""
🤖 Available Commands:
/help     - Show this help
/phase    - Change project phase
/github   - Add GitHub repository
/status   - Show system status
/context  - Add project context
/export   - Export project data
/projects - Switch projects
/quit     - Exit system
""")

                elif command == '/phase':
                    current_phase = rag.current_project.phase
                    phases = ["discovery", "analysis", "design", "implementation"]
                    print(f"Current phase: {current_phase}")
                    print("Available phases:", ", ".join(phases))

                    new_phase = input("New phase: ").lower()
                    if rag.update_project_phase(new_phase):
                        print(f"✅ Updated to {new_phase} phase")
                    else:
                        print("❌ Invalid phase")

                elif command == '/github':
                    url = input("GitHub repository URL: ")
                    print("🔄 Processing repository...")
                    result = rag.add_github_repository(url)

                    if "error" in result:
                        print(f"❌ {result['error']}")
                    else:
                        print(f"✅ Added {result['repository']}")
                        print(f"📁 Processed {result['files_processed']} files")

                elif command == '/status':
                    status = rag.get_system_status()
                    print(f"""
📊 System Status:
User: {status['user_info']['username']}
Project: {status['current_project']} (Phase: {status.get('project_phase', 'N/A')})
Available Projects: {status['available_projects']}

💰 Token Usage:
Total: {status['user_info']['total_tokens_used']:,} tokens
Cost: ${status['user_info']['total_estimated_cost']:.4f}
Last 24h: {status['user_info']['recent_tokens_24h']} tokens (${status['user_info']['recent_cost_24h']:.4f})
""")

                elif command == '/context':
                    category = input("Category (goals/requirements/tech_stack/constraints): ")
                    item = input(f"Add to {category}: ")

                    if rag.add_project_context(category, item):
                        print(f"✅ Added to {category}")
                    else:
                        print("❌ Invalid category")

                elif command == '/export':
                    data = rag.export_project_data()
                    if data:
                        filename = f"{rag.current_project.name.replace(' ', '_')}_export.json"
                        with open(filename, 'w') as f:
                            json.dump(data, f, indent=2)
                        print(f"✅ Exported to {filename}")
                    else:
                        print("❌ Export failed")

                elif command == '/projects':
                    projects = rag.list_user_projects()
                    if projects:
                        for i, project in enumerate(projects):
                            print(f"{i + 1}. {project['name']} (Phase: {project['phase']})")

                        try:
                            choice = int(input(f"Select project (1-{len(projects)}): "))
                            if 1 <= choice <= len(projects):
                                project = projects[choice - 1]
                                if rag.load_project(project['project_id']):
                                    print(f"✅ Switched to: {project['name']}")
                                else:
                                    print("❌ Failed to load project")
                        except ValueError:
                            print("Invalid selection")

                else:
                    print("❌ Unknown command. Type /help for available commands.")

                continue

            # Generate Socratic response
            print("🤔 Thinking...")
            result = rag.generate_socratic_response(user_input)

            if "error" in result:
                print(f"❌ {result['error']}")
                continue

            # Display response
            print(f"\n🎓 Mentor: {result['response']}")

            # Show token usage info
            usage = result['token_usage']
            print(f"\n📊 Token Usage: {usage['total_tokens']} tokens (${usage['estimated_cost']:.4f})")
            print(f"💰 Your Total: {usage['user_total_tokens']:,} tokens (${usage['user_total_cost']:.4f})")

            # Show context used
            context = result['context_used']
            if any(context.values()):
                context_info = []
                if context['knowledge_items']: context_info.append(f"{context['knowledge_items']} knowledge")
                if context['conversation_items']: context_info.append(f"{context['conversation_items']} conversations")
                if context['code_items']: context_info.append(f"{context['code_items']} code files")
                if context['similar_projects']: context_info.append(f"{context['similar_projects']} similar projects")

                print(f"🔍 Context: {', '.join(context_info)}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()