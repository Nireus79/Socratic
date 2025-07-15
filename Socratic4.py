#!/usr/bin/env python3
"""
Enhanced Socratic RAG System with Code Generation
After building context through Socratic questioning, generates actual code/scripts
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic


@dataclass
class ProjectContext:
    """Enhanced project context with code generation fields"""
    goals_requirements: List[str]
    technical_stack: List[str]
    constraints: List[str]
    team_structure: str
    language_preferences: List[str]
    deployment_target: str
    code_style: str
    architecture_pattern: str = ""
    database_schema: List[str] = None
    api_endpoints: List[str] = None
    ui_components: List[str] = None
    business_logic: List[str] = None

    def __post_init__(self):
        if self.database_schema is None:
            self.database_schema = []
        if self.api_endpoints is None:
            self.api_endpoints = []
        if self.ui_components is None:
            self.ui_components = []
        if self.business_logic is None:
            self.business_logic = []


@dataclass
class CodeTemplate:
    """Template for generating specific types of code"""
    name: str
    description: str
    tech_stack: List[str]
    template_type: str  # 'web_app', 'api', 'database', 'config', 'script'
    base_structure: str
    dependencies: List[str]


class CodeGenerator:
    """Generates code based on project context"""

    def __init__(self, client):
        self.client = client
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, CodeTemplate]:
        """Load code templates for different project types"""
        templates = {}

        # Web App Templates
        templates['flask_web_app'] = CodeTemplate(
            name="Flask Web Application",
            description="Basic Flask web application structure",
            tech_stack=['python', 'flask', 'web'],
            template_type='web_app',
            base_structure="""
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Models will be generated based on context
{models}

# Routes will be generated based on context
{routes}

# Business logic will be generated based on context
{business_logic}

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
""",
            dependencies=['flask', 'flask-sqlalchemy']
        )

        templates['fastapi_app'] = CodeTemplate(
            name="FastAPI Application",
            description="FastAPI REST API structure",
            tech_stack=['python', 'fastapi', 'api'],
            template_type='api',
            base_structure="""
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime
import os

app = FastAPI(title="{app_name}", version="1.0.0")

# Database setup
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///./app.db')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
{models}

# Pydantic schemas
{schemas}

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API routes
{routes}

# Business logic
{business_logic}
""",
            dependencies=['fastapi', 'sqlalchemy', 'pydantic']
        )

        templates['react_component'] = CodeTemplate(
            name="React Component",
            description="React component structure",
            tech_stack=['javascript', 'react', 'frontend'],
            template_type='ui',
            base_structure="""
import React, {{ useState, useEffect }} from 'react';
import axios from 'axios';

const {component_name} = () => {{
    {state_variables}

    {use_effects}

    {event_handlers}

    return (
        <div className="{component_name.lower()}">
            {jsx_structure}
        </div>
    );
}};

export default {component_name};
""",
            dependencies=['react', 'axios']
        )

        return templates

    def generate_code_from_context(self, context: ProjectContext) -> Dict[str, str]:
        """Generate complete code based on project context"""
        generated_files = {}

        # Determine primary template based on tech stack
        primary_template = self._select_primary_template(context)

        if primary_template:
            # Generate main application file
            main_code = self._generate_main_application(context, primary_template)
            generated_files['main.py'] = main_code

            # Generate models based on database schema
            if context.database_schema:
                models_code = self._generate_models(context)
                generated_files['models.py'] = models_code

            # Generate API routes
            if context.api_endpoints:
                routes_code = self._generate_routes(context)
                generated_files['routes.py'] = routes_code

            # Generate configuration files
            config_code = self._generate_config(context)
            generated_files['config.py'] = config_code

            # Generate requirements.txt
            requirements = self._generate_requirements(context, primary_template)
            generated_files['requirements.txt'] = requirements

            # Generate Docker configuration if deployment target includes containers
            if 'docker' in context.deployment_target.lower():
                dockerfile = self._generate_dockerfile(context)
                generated_files['Dockerfile'] = dockerfile

            # Generate frontend components if needed
            if any('react' in tech.lower() for tech in context.technical_stack):
                frontend_code = self._generate_frontend_components(context)
                generated_files.update(frontend_code)

        return generated_files

    def _select_primary_template(self, context: ProjectContext) -> Optional[CodeTemplate]:
        """Select the most appropriate template based on context"""
        tech_stack_lower = [tech.lower() for tech in context.technical_stack]

        if 'fastapi' in tech_stack_lower:
            return self.templates['fastapi_app']
        elif 'flask' in tech_stack_lower:
            return self.templates['flask_web_app']
        elif 'react' in tech_stack_lower:
            return self.templates['react_component']

        # Default to Flask for web applications
        if any(web in tech_stack_lower for web in ['web', 'webapp', 'website']):
            return self.templates['flask_web_app']

        return None

    def _generate_main_application(self, context: ProjectContext, template: CodeTemplate) -> str:
        """Generate the main application code"""
        # Use Claude to generate contextual code
        prompt = f"""
        Generate a complete {template.name} based on this project context:

        Goals: {', '.join(context.goals_requirements)}
        Tech Stack: {', '.join(context.technical_stack)}
        Architecture: {context.architecture_pattern}
        Database Schema: {', '.join(context.database_schema)}
        API Endpoints: {', '.join(context.api_endpoints)}

        Use this template structure:
        {template.base_structure}

        Generate production-ready code with proper error handling, logging, and security considerations.
        Include detailed comments explaining the code structure.
        """

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _generate_models(self, context: ProjectContext) -> str:
        """Generate database models based on schema"""
        prompt = f"""
        Generate database models for this project:

        Database Schema: {', '.join(context.database_schema)}
        Tech Stack: {', '.join(context.technical_stack)}
        Goals: {', '.join(context.goals_requirements)}

        Create SQLAlchemy models with proper relationships, constraints, and indexes.
        Include validation and helper methods where appropriate.
        """

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _generate_routes(self, context: ProjectContext) -> str:
        """Generate API routes based on endpoints"""
        prompt = f"""
        Generate API routes for this project:

        API Endpoints: {', '.join(context.api_endpoints)}
        Tech Stack: {', '.join(context.technical_stack)}
        Business Logic: {', '.join(context.business_logic)}

        Create complete route handlers with proper HTTP methods, validation, and error handling.
        Include authentication and authorization where needed.
        """

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _generate_config(self, context: ProjectContext) -> str:
        """Generate configuration files"""
        config_template = f"""
import os
from typing import Dict, Any

class Config:
    '''Application configuration'''

    # Basic settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///app.db')

    # Deployment: {context.deployment_target}
    # Tech Stack: {', '.join(context.technical_stack)}

    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True

config = {{
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}}
"""
        return config_template

    def _generate_requirements(self, context: ProjectContext, template: CodeTemplate) -> str:
        """Generate requirements.txt based on dependencies"""
        base_requirements = template.dependencies.copy()

        # Add context-specific requirements
        tech_stack_lower = [tech.lower() for tech in context.technical_stack]

        if 'postgresql' in tech_stack_lower:
            base_requirements.append('psycopg2-binary')
        if 'mysql' in tech_stack_lower:
            base_requirements.append('PyMySQL')
        if 'redis' in tech_stack_lower:
            base_requirements.append('redis')
        if 'celery' in tech_stack_lower:
            base_requirements.append('celery')

        return '\n'.join(sorted(set(base_requirements)))

    def _generate_dockerfile(self, context: ProjectContext) -> str:
        """Generate Dockerfile for containerization"""
        dockerfile_template = f"""
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
"""
        return dockerfile_template

    def _generate_frontend_components(self, context: ProjectContext) -> Dict[str, str]:
        """Generate frontend components if React is in tech stack"""
        components = {}

        if context.ui_components:
            for component in context.ui_components:
                component_name = component.replace(' ', '').replace('-', '')

                prompt = f"""
                Generate a React component for: {component}

                Project context:
                - Goals: {', '.join(context.goals_requirements)}
                - UI Components needed: {', '.join(context.ui_components)}
                - API Endpoints: {', '.join(context.api_endpoints)}

                Create a functional component with hooks, proper state management, and API integration.
                Include proper error handling and loading states.
                """

                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )

                components[f"{component_name}.jsx"] = response.content[0].text

        return components


class EnhancedSocraticRAG:
    """Enhanced Socratic RAG system with code generation capabilities"""

    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get('API_KEY_CLAUDE'))
        self.code_generator = CodeGenerator(self.client)
        self.context = ProjectContext(
            goals_requirements=[],
            technical_stack=[],
            constraints=[],
            team_structure="",
            language_preferences=[],
            deployment_target="",
            code_style=""
        )
        self.conversation_history = []
        self.current_phase = "discovery"
        self.ready_for_code_generation = False

    def chat(self, user_input: str) -> str:
        """Main chat interface with code generation capability"""
        if user_input.lower() in ['quit', 'exit', 'end']:
            return "Goodbye! Your project context has been saved."

        if user_input.lower() == 'generate' or user_input.lower() == 'code':
            return self._generate_project_code()

        if user_input.lower() == 'summary':
            return self._generate_context_summary()

        # Process user input and update context
        self._update_context_from_input(user_input)

        # Generate next Socratic question
        next_question = self._generate_next_question()

        # Check if ready for code generation
        if self._is_ready_for_code_generation():
            self.ready_for_code_generation = True
            next_question += ("\n\n🎯 Your project context looks complete! Type 'generate' or 'code' to create your "
                              "project files.")

        return next_question

    def _generate_project_code(self) -> str:
        """Generate complete project code based on context"""
        if not self.ready_for_code_generation:
            return "❌ Project context is not yet complete. Please continue the conversation to build more context."

        try:
            print("\n🔄 Generating your project code...")
            generated_files = self.code_generator.generate_code_from_context(self.context)

            # Save files to disk
            project_dir = "generated_project"
            os.makedirs(project_dir, exist_ok=True)

            for filename, content in generated_files.items():
                filepath = os.path.join(project_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

            result = f"""
🎉 SUCCESS! Your project has been generated!

📁 Files created in '{project_dir}/' directory:
{chr(10).join(f'   • {filename}' for filename in generated_files.keys())}

🚀 Next steps:
1. Navigate to the project directory: cd {project_dir}
2. Install dependencies: pip install -r requirements.txt
3. Run your application: python main.py

💡 Your project is ready to customize and extend!
"""
            return result

        except Exception as e:
            return f"❌ Error generating code: {str(e)}"

    def _update_context_from_input(self, user_input: str):
        """Update project context based on user input using Claude"""
        prompt = f"""
        Analyze this user input and extract relevant project context information:

        User input: "{user_input}"

        Current context: {json.dumps(asdict(self.context), indent=2)}

        Extract and categorize any new information about:
        - Goals and requirements
        - Technical stack (frameworks, languages, databases)
        - Architecture patterns
        - Database schema needs
        - API endpoints
        - UI components
        - Business logic
        - Constraints
        - Deployment preferences

        Return a JSON object with the extracted information.
        """

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            # Parse the response and update context
            # This is a simplified version - in practice, you'd want more robust parsing
            content = response.content[0].text
            self._merge_context_updates(content, user_input)
        except Exception as e:
            print(f"Error updating context: {e}")

    def _merge_context_updates(self, claude_response: str, user_input: str):
        """Merge context updates from Claude's analysis"""
        # Simple keyword-based context updating
        user_lower = user_input.lower()

        # Technical stack detection
        if any(tech in user_lower for tech in ['flask', 'fastapi', 'django']):
            if 'flask' in user_lower and 'flask' not in self.context.technical_stack:
                self.context.technical_stack.append('flask')
            if 'fastapi' in user_lower and 'fastapi' not in self.context.technical_stack:
                self.context.technical_stack.append('fastapi')

        if any(db in user_lower for db in ['postgresql', 'mysql', 'sqlite']):
            if 'postgresql' in user_lower and 'postgresql' not in self.context.technical_stack:
                self.context.technical_stack.append('postgresql')

        if any(frontend in user_lower for frontend in ['react', 'vue', 'angular']):
            if 'react' in user_lower and 'react' not in self.context.technical_stack:
                self.context.technical_stack.append('react')

        # Goals and requirements
        if 'manage' in user_lower or 'management' in user_lower:
            if 'task management' not in self.context.goals_requirements:
                self.context.goals_requirements.append('task management')

        # Add more sophisticated parsing here

    def _generate_next_question(self) -> str:
        """Generate the next Socratic question based on current context"""
        prompt = f"""
        You are a Socratic counselor helping a developer plan their project. 

        Current project context:
        - Goals: {', '.join(self.context.goals_requirements) if self.context.goals_requirements else 'None specified'}
        - Tech Stack: {', '.join(self.context.technical_stack) if self.context.technical_stack else 'None specified'}
        - Architecture: {self.context.architecture_pattern if self.context.architecture_pattern else 'None specified'}
        - Database Schema: {', '.join(self.context.database_schema) if self.context.database_schema else 'None specified'}
        - API Endpoints: {', '.join(self.context.api_endpoints) if self.context.api_endpoints else 'None specified'}

        Current phase: {self.current_phase}

        Generate the next most important Socratic question to help them discover what they need for their project.
        Focus on areas where context is missing or needs clarification.

        Ask only ONE question. Be specific and actionable.
        """

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _is_ready_for_code_generation(self) -> bool:
        """Check if we have enough context to generate code"""
        return (
                len(self.context.goals_requirements) > 0 and
                len(self.context.technical_stack) > 0 and
                (len(self.context.api_endpoints) > 0 or len(self.context.ui_components) > 0)
        )

    def _generate_context_summary(self) -> str:
        """Generate a summary of the current project context"""
        summary = f"""
📋 PROJECT CONTEXT SUMMARY
=============================

🎯 Goals & Requirements:
{chr(10).join(f'   • {goal}' for goal in self.context.goals_requirements) if self.context.goals_requirements else '   • None specified'}

🛠️ Technical Stack:
{chr(10).join(f'   • {tech}' for tech in self.context.technical_stack) if self.context.technical_stack else '   • None specified'}

🏗️ Architecture Pattern:
   • {self.context.architecture_pattern if self.context.architecture_pattern else 'None specified'}

🗄️ Database Schema:
{chr(10).join(f'   • {schema}' for schema in self.context.database_schema) if self.context.database_schema else '   • None specified'}

🌐 API Endpoints:
{chr(10).join(f'   • {endpoint}' for endpoint in self.context.api_endpoints) if self.context.api_endpoints else '   • None specified'}

🎨 UI Components:
{chr(10).join(f'   • {component}' for component in self.context.ui_components) if self.context.ui_components else '   • None specified'}

🚀 Deployment Target:
   • {self.context.deployment_target if self.context.deployment_target else 'None specified'}

⚡ Ready for Code Generation: {'✅ Yes' if self.ready_for_code_generation else '❌ No - need more context'}
"""
        return summary


def main():
    """Main function to run the enhanced Socratic RAG system"""
    print("🤖 Enhanced Socratic Counselor with Code Generation")
    print("=" * 55)
    print("I'll help you plan your project and then generate the actual code!")
    print("Commands: 'summary' (show context), 'generate' (create code), 'quit' (exit)")
    print()

    # Initialize the system
    api_key = os.environ.get('API_KEY_CLAUDE')
    if not api_key:
        api_key = input("Enter your Claude API key: ")

    try:
        rag = EnhancedSocraticRAG(api_key)

        # Start the conversation
        print("Assistant: What exactly do you want to achieve with this project?")

        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            response = rag.chat(user_input)
            print(f"\nAssistant: {response}")

            if user_input.lower() in ['quit', 'exit', 'end']:
                break

    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure your Claude API key is correct and you have internet connection.")


if __name__ == "__main__":
    main()
