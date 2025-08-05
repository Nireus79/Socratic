#!/usr/bin/env python3
"""
CodeCraft Agent - Agentic RAG Code Generator
Companion to Socratic6.1 for context-driven code generation

This system takes project context and specifications from a database
and generates actual code using an agentic RAG approach.
"""

import json
import sqlite3
import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import hashlib
import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings

# External dependencies (install with pip)
# try:
#     import openai
#     from sentence_transformers import SentenceTransformer
#     import numpy as np
#     from sklearn.metrics.pairwise import cosine_similarity
#     import chromadb
#     from chromadb.config import Settings
# except ImportError as e:
#     print(f"Missing dependency: {e}")
#     print("Install with: pip install openai sentence-transformers scikit-learn chromadb")


@dataclass
class CodeArtifact:
    """Represents a generated code artifact"""
    filename: str
    content: str
    language: str
    description: str
    dependencies: List[str]
    tests: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ProjectSpec:
    """Project specification from Socratic context"""
    project_id: str
    name: str
    description: str
    requirements: List[str]
    architecture: Dict[str, Any]
    technologies: List[str]
    context_summary: str
    constraints: List[str] = None

    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []


class CodeCraftAgent:
    """
    Agentic RAG system for code generation based on project context
    """

    def __init__(self,
                 context_db_path: str = "socratic_context.db",
                 vector_db_path: str = "./chroma_db",
                 openai_api_key: str = None,
                 model_name: str = "gpt-4"):

        self.context_db_path = context_db_path
        self.vector_db_path = vector_db_path
        self.model_name = model_name

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client
        openai.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key required")

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize vector database
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        self.code_collection = self.chroma_client.get_or_create_collection(
            name="code_patterns",
            metadata={"hnsw:space": "cosine"}
        )

        # Code generation patterns and templates
        self.code_patterns = self._load_code_patterns()

        # Agent tools
        self.tools = {
            "analyze_requirements": self._analyze_requirements,
            "retrieve_patterns": self._retrieve_code_patterns,
            "generate_architecture": self._generate_architecture,
            "generate_code": self._generate_code,
            "validate_code": self._validate_code,
            "create_tests": self._create_tests,
            "optimize_code": self._optimize_code
        }

    def _load_code_patterns(self) -> Dict[str, str]:
        """Load common code patterns and templates"""
        return {
            "python_class": """
class {class_name}:
    \"\"\"
    {description}
    \"\"\"

    def __init__(self, {init_params}):
        {init_body}

    {methods}
""",
            "python_function": """
def {function_name}({parameters}) -> {return_type}:
    \"\"\"
    {description}

    Args:
        {args_docs}

    Returns:
        {return_docs}
    \"\"\"
    {function_body}
""",
            "api_endpoint": """
@app.route('/{endpoint}', methods=['{method}'])
def {function_name}():
    \"\"\"
    {description}
    \"\"\"
    try:
        {endpoint_body}
        return jsonify({{"status": "success", "data": result}})
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500
""",
            "database_model": """
class {model_name}(db.Model):
    \"\"\"
    {description}
    \"\"\"
    __tablename__ = '{table_name}'

    {fields}

    def to_dict(self):
        return {{
            {to_dict_fields}
        }}
""",
            "react_component": """
import React, {{ useState{additional_imports} }} from 'react';
{external_imports}

interface {component_name}Props {{
    {prop_types}
}}

const {component_name}: React.FC<{component_name}Props> = ({{ {props} }}) => {{
    {state_declarations}

    {component_body}

    return (
        {jsx_return}
    );
}};

export default {component_name};
""",
            "test_template": """
import unittest
from unittest.mock import Mock, patch
from {module_path} import {class_or_function}

class Test{test_class_name}(unittest.TestCase):
    \"\"\"
    Test cases for {class_or_function}
    \"\"\"

    def setUp(self):
        {setup_code}

    def tearDown(self):
        {teardown_code}

    {test_methods}

if __name__ == '__main__':
    unittest.main()
"""
        }

    def load_project_context(self, project_id: str) -> Optional[ProjectSpec]:
        """Load project context from Socratic database"""
        try:
            conn = sqlite3.connect(self.context_db_path)
            cursor = conn.cursor()

            # Assuming Socratic6.1 stores context in a structured format
            cursor.execute("""
                SELECT name, description, requirements, architecture, 
                       technologies, context_summary, constraints
                FROM projects WHERE id = ?
            """, (project_id,))

            row = cursor.fetchone()
            if not row:
                self.logger.warning(f"Project {project_id} not found in context database")
                return None

            # Parse JSON fields if they exist
            requirements = json.loads(row[2]) if row[2] else []
            architecture = json.loads(row[3]) if row[3] else {}
            technologies = json.loads(row[4]) if row[4] else []
            constraints = json.loads(row[6]) if row[6] else []

            return ProjectSpec(
                project_id=project_id,
                name=row[0],
                description=row[1],
                requirements=requirements,
                architecture=architecture,
                technologies=technologies,
                context_summary=row[5],
                constraints=constraints
            )

        except Exception as e:
            self.logger.error(f"Error loading project context: {e}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()

    def _analyze_requirements(self, project_spec: ProjectSpec) -> Dict[str, Any]:
        """Analyze project requirements and break them down into actionable tasks"""

        analysis_prompt = f"""
        Analyze the following project requirements and provide a structured breakdown:

        Project: {project_spec.name}
        Description: {project_spec.description}
        Requirements: {project_spec.requirements}
        Technologies: {project_spec.technologies}
        Architecture: {project_spec.architecture}
        Constraints: {project_spec.constraints}

        Provide:
        1. Core components needed
        2. Data models required
        3. API endpoints (if applicable)
        4. Key algorithms or business logic
        5. Testing strategy
        6. Deployment considerations

        Format as JSON with clear categories.
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "You are a senior software architect. Analyze requirements and provide structured technical breakdowns."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3
            )

            analysis_text = response.choices[0].message.content

            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback: create structured analysis from text
                return {"analysis": analysis_text}

        except Exception as e:
            self.logger.error(f"Error analyzing requirements: {e}")
            return {"error": str(e)}

    def _retrieve_code_patterns(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant code patterns from vector database"""
        try:
            # Generate embedding for query
            query_embedding = self.embedding_model.encode([query])

            # Search vector database
            results = self.code_collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )

            patterns = []
            for i, doc in enumerate(results['documents'][0]):
                patterns.append({
                    "content": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'][0] else {},
                    "distance": results['distances'][0][i] if results['distances'][0] else 0
                })

            return patterns

        except Exception as e:
            self.logger.error(f"Error retrieving code patterns: {e}")
            return []

    def _generate_architecture(self, project_spec: ProjectSpec, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed technical architecture"""

        architecture_prompt = f"""
        Based on the project analysis, generate a detailed technical architecture:

        Project: {project_spec.name}
        Analysis: {json.dumps(analysis, indent=2)}
        Technologies: {project_spec.technologies}
        Existing Architecture Ideas: {project_spec.architecture}

        Generate:
        1. Directory structure
        2. Module organization
        3. Database schema (if applicable)
        4. API design (if applicable)
        5. Component relationships
        6. Configuration management
        7. Error handling strategy

        Format as detailed JSON structure.
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "You are a senior software architect specializing in scalable system design."},
                    {"role": "user", "content": architecture_prompt}
                ],
                temperature=0.2
            )

            architecture_text = response.choices[0].message.content

            # Extract JSON
            json_match = re.search(r'\{.*\}', architecture_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"architecture": architecture_text}

        except Exception as e:
            self.logger.error(f"Error generating architecture: {e}")
            return {"error": str(e)}

    def _generate_code(self,
                       component: str,
                       project_spec: ProjectSpec,
                       architecture: Dict[str, Any],
                       patterns: List[Dict[str, Any]]) -> CodeArtifact:
        """Generate code for a specific component"""

        # Select appropriate template
        template = self._select_template(component, project_spec.technologies)

        # Build context from patterns
        pattern_context = "\n".join([p["content"] for p in patterns[:3]])

        code_prompt = f"""
        Generate production-ready code for the following component:

        Component: {component}
        Project: {project_spec.name}
        Technologies: {project_spec.technologies}
        Architecture: {json.dumps(architecture, indent=2)}

        Similar patterns:
        {pattern_context}

        Requirements:
        - Follow best practices for {project_spec.technologies}
        - Include proper error handling
        - Add comprehensive docstrings/comments
        - Ensure code is testable
        - Include type hints where applicable
        - Follow SOLID principles

        Generate only the code, no explanations.
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "You are an expert software engineer. Generate clean, production-ready code with best practices."},
                    {"role": "user", "content": code_prompt}
                ],
                temperature=0.1
            )

            code_content = response.choices[0].message.content

            # Clean up code (remove markdown formatting if present)
            code_content = re.sub(r'^```\w*\n', '', code_content)
            code_content = re.sub(r'\n```$', '', code_content)

            # Determine file extension
            language = self._detect_language(project_spec.technologies)
            extension = self._get_file_extension(language)
            filename = f"{component.lower().replace(' ', '_')}{extension}"

            # Extract dependencies
            dependencies = self._extract_dependencies(code_content, language)

            return CodeArtifact(
                filename=filename,
                content=code_content,
                language=language,
                description=f"Generated {component} component",
                dependencies=dependencies
            )

        except Exception as e:
            self.logger.error(f"Error generating code for {component}: {e}")
            return CodeArtifact(
                filename=f"{component}.error",
                content=f"# Error generating code: {e}",
                language="text",
                description=f"Error in {component}",
                dependencies=[]
            )

    def _select_template(self, component: str, technologies: List[str]) -> str:
        """Select appropriate code template based on component and technologies"""
        component_lower = component.lower()

        if "class" in component_lower or "model" in component_lower:
            if "python" in [t.lower() for t in technologies]:
                return self.code_patterns["python_class"]
            elif "database" in component_lower:
                return self.code_patterns["database_model"]

        elif "api" in component_lower or "endpoint" in component_lower:
            return self.code_patterns["api_endpoint"]

        elif "react" in [t.lower() for t in technologies]:
            return self.code_patterns["react_component"]

        else:
            return self.code_patterns["python_function"]

    def _detect_language(self, technologies: List[str]) -> str:
        """Detect primary programming language from technologies"""
        tech_lower = [t.lower() for t in technologies]

        if any(t in tech_lower for t in ["python", "django", "flask", "fastapi"]):
            return "python"
        elif any(t in tech_lower for t in ["javascript", "react", "node", "express"]):
            return "javascript"
        elif any(t in tech_lower for t in ["java", "spring"]):
            return "java"
        elif any(t in tech_lower for t in ["csharp", "c#", ".net"]):
            return "csharp"
        else:
            return "python"  # default

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language"""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "java": ".java",
            "csharp": ".cs",
            "typescript": ".ts",
            "html": ".html",
            "css": ".css"
        }
        return extensions.get(language, ".txt")

    def _extract_dependencies(self, code: str, language: str) -> List[str]:
        """Extract dependencies from generated code"""
        dependencies = []

        if language == "python":
            import_pattern = r'(?:from|import)\s+(\w+)'
            matches = re.findall(import_pattern, code)
            dependencies.extend(matches)

        elif language == "javascript":
            import_pattern = r'import.*from\s+[\'"]([^\'"]+)[\'"]'
            matches = re.findall(import_pattern, code)
            dependencies.extend(matches)

        return list(set(dependencies))  # Remove duplicates

    def _validate_code(self, artifact: CodeArtifact) -> Dict[str, Any]:
        """Validate generated code for syntax and basic quality"""
        validation_results = {
            "syntax_valid": False,
            "issues": [],
            "suggestions": []
        }

        try:
            if artifact.language == "python":
                # Basic Python syntax validation
                compile(artifact.content, artifact.filename, 'exec')
                validation_results["syntax_valid"] = True

            # Check for common issues
            if len(artifact.content.split('\n')) < 5:
                validation_results["issues"].append("Code seems too short")

            if "TODO" in artifact.content or "FIXME" in artifact.content:
                validation_results["issues"].append("Contains TODO/FIXME comments")

            # Check for documentation
            if '"""' not in artifact.content and "'''" not in artifact.content:
                validation_results["suggestions"].append("Consider adding docstrings")

        except SyntaxError as e:
            validation_results["issues"].append(f"Syntax error: {e}")
        except Exception as e:
            validation_results["issues"].append(f"Validation error: {e}")

        return validation_results

    def _create_tests(self, artifact: CodeArtifact, project_spec: ProjectSpec) -> CodeArtifact:
        """Generate tests for the code artifact"""

        test_prompt = f"""
        Generate comprehensive unit tests for the following code:

        Filename: {artifact.filename}
        Language: {artifact.language}
        Description: {artifact.description}

        Code:
        {artifact.content}

        Generate tests that cover:
        - Happy path scenarios
        - Edge cases
        - Error conditions
        - Boundary conditions

        Use appropriate testing framework for {artifact.language}.
        Include setup and teardown if needed.
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a senior QA engineer. Generate comprehensive test suites."},
                    {"role": "user", "content": test_prompt}
                ],
                temperature=0.2
            )

            test_content = response.choices[0].message.content

            # Clean up test code
            test_content = re.sub(r'^```\w*\n', '', test_content)
            test_content = re.sub(r'\n```$', '', test_content)

            test_filename = f"test_{artifact.filename}"

            return CodeArtifact(
                filename=test_filename,
                content=test_content,
                language=artifact.language,
                description=f"Tests for {artifact.description}",
                dependencies=artifact.dependencies + ["unittest", "pytest"]
            )

        except Exception as e:
            self.logger.error(f"Error creating tests: {e}")
            return CodeArtifact(
                filename=f"test_{artifact.filename}",
                content=f"# Error generating tests: {e}",
                language="text",
                description="Test generation error",
                dependencies=[]
            )

    def _optimize_code(self, artifact: CodeArtifact) -> CodeArtifact:
        """Optimize generated code for performance and readability"""

        optimization_prompt = f"""
        Optimize the following code for performance, readability, and maintainability:

        Current code:
        {artifact.content}

        Optimize for:
        - Performance improvements
        - Memory efficiency
        - Code readability
        - Maintainability
        - Best practices adherence

        Return only the optimized code.
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "You are a senior software engineer specializing in code optimization."},
                    {"role": "user", "content": optimization_prompt}
                ],
                temperature=0.1
            )

            optimized_content = response.choices[0].message.content

            # Clean up code
            optimized_content = re.sub(r'^```\w*\n', '', optimized_content)
            optimized_content = re.sub(r'\n```$', '', optimized_content)

            # Create optimized artifact
            optimized_artifact = CodeArtifact(
                filename=artifact.filename,
                content=optimized_content,
                language=artifact.language,
                description=f"Optimized {artifact.description}",
                dependencies=artifact.dependencies
            )

            return optimized_artifact

        except Exception as e:
            self.logger.error(f"Error optimizing code: {e}")
            return artifact  # Return original if optimization fails

    def generate_project_code(self, project_id: str) -> Dict[str, Any]:
        """
        Main method to generate complete project code
        """
        self.logger.info(f"Starting code generation for project: {project_id}")

        # Load project context
        project_spec = self.load_project_context(project_id)
        if not project_spec:
            return {"error": "Project context not found"}

        # Step 1: Analyze requirements
        self.logger.info("Analyzing requirements...")
        analysis = self._analyze_requirements(project_spec)

        # Step 2: Generate architecture
        self.logger.info("Generating architecture...")
        architecture = self._generate_architecture(project_spec, analysis)

        # Step 3: Identify components to generate
        components = self._identify_components(analysis, architecture)

        # Step 4: Generate code for each component
        generated_artifacts = []
        for component in components:
            self.logger.info(f"Generating code for: {component}")

            # Retrieve relevant patterns
            patterns = self._retrieve_code_patterns(component)

            # Generate code
            artifact = self._generate_code(component, project_spec, architecture, patterns)

            # Validate code
            validation = self._validate_code(artifact)
            artifact.metadata = {"validation": validation}

            # Generate tests
            test_artifact = self._create_tests(artifact, project_spec)

            # Optimize code
            optimized_artifact = self._optimize_code(artifact)

            generated_artifacts.extend([optimized_artifact, test_artifact])

        # Step 5: Save artifacts
        self._save_artifacts(project_id, generated_artifacts)

        return {
            "project_id": project_id,
            "status": "success",
            "artifacts_generated": len(generated_artifacts),
            "artifacts": [asdict(artifact) for artifact in generated_artifacts],
            "analysis": analysis,
            "architecture": architecture
        }

    def _identify_components(self, analysis: Dict[str, Any], architecture: Dict[str, Any]) -> List[str]:
        """Identify components to generate based on analysis and architecture"""
        components = []

        # Extract from analysis
        if "core_components" in analysis:
            components.extend(analysis["core_components"])

        if "data_models" in analysis:
            components.extend([f"{model} Model" for model in analysis["data_models"]])

        if "api_endpoints" in analysis:
            components.extend([f"{endpoint} API" for endpoint in analysis["api_endpoints"]])

        # Extract from architecture
        if "modules" in architecture:
            components.extend(architecture["modules"])

        # Remove duplicates and return
        return list(set(components))

    def _save_artifacts(self, project_id: str, artifacts: List[CodeArtifact]):
        """Save generated artifacts to database and files"""
        try:
            # Create project directory
            project_dir = Path(f"generated_projects/{project_id}")
            project_dir.mkdir(parents=True, exist_ok=True)

            # Save artifacts to files
            for artifact in artifacts:
                file_path = project_dir / artifact.filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(artifact.content)

                self.logger.info(f"Saved: {file_path}")

            # Save metadata to database
            conn = sqlite3.connect("generated_code.db")
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generated_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT,
                    filename TEXT,
                    language TEXT,
                    description TEXT,
                    dependencies TEXT,
                    timestamp TEXT,
                    content_hash TEXT
                )
            """)

            # Insert artifacts
            for artifact in artifacts:
                content_hash = hashlib.md5(artifact.content.encode()).hexdigest()
                cursor.execute("""
                    INSERT INTO generated_artifacts 
                    (project_id, filename, language, description, dependencies, timestamp, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    project_id,
                    artifact.filename,
                    artifact.language,
                    artifact.description,
                    json.dumps(artifact.dependencies),
                    artifact.timestamp,
                    content_hash
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error saving artifacts: {e}")

    def add_code_pattern(self, pattern_name: str, code: str, description: str, tags: List[str]):
        """Add a new code pattern to the vector database"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode([code])

            # Add to vector database
            self.code_collection.add(
                documents=[code],
                metadatas=[{
                    "name": pattern_name,
                    "description": description,
                    "tags": json.dumps(tags)
                }],
                ids=[f"pattern_{pattern_name}_{datetime.now().timestamp()}"]
            )

            self.logger.info(f"Added code pattern: {pattern_name}")

        except Exception as e:
            self.logger.error(f"Error adding code pattern: {e}")


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="CodeCraft Agent - Generate code from project context")
    parser.add_argument("--project-id", required=True, help="Project ID from Socratic context")
    parser.add_argument("--context-db", default="socratic_context.db", help="Path to Socratic context database")
    parser.add_argument("--api-key", help="OpenAI API key")

    args = parser.parse_args()

    try:
        # Initialize CodeCraft Agent
        agent = CodeCraftAgent(
            context_db_path=args.context_db,
            openai_api_key=args.api_key
        )

        # Generate code
        result = agent.generate_project_code(args.project_id)

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Successfully generated {result['artifacts_generated']} artifacts")
            print(f"Project files saved to: generated_projects/{args.project_id}/")

    except Exception as e:
        print(f"Failed to generate code: {e}")


if __name__ == "__main__":
    main()
